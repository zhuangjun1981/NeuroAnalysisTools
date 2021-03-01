'''
to read and analyze swc files from neuron
morphology reconstruction
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
from mpl_toolkits.mplot3d import Axes3D

COLOR_DICT = {
    2 : '#ff0000', # axon ?
    3 : '#0000ff', # apical dendrite ?
    4 : '#ff0000', # basal dendrite ?
    7 : '#000000', # unknown ?
              }


def read_swc(file_path, vox_size_x=None, vox_size_y=None, vox_size_z=None,
             name='', comment='', unit=''):
    """

    :param file_path: str, path to the swc file
    :param vox_size_x: float, voxel size in x
    :param vox_size_y: float, voxel size in y
    :param vox_size_z: float, voxel size in z
    :param unit: str, unit of voxel sizes
    :return: AxonTree object
    """
    n_skip = 0

    columns = ["##n", "type", "z", "y", "x", "r", "parent"]

    with open(file_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("#"):
                n_skip += 1

                if line.startswith("#n"):
                    columns = line[1:].split(' ')[0:7]
                elif line.startswith("##n"):
                    columns = line[2:].split(',')[0:7]
                elif line.startswith("#name"):
                    if len(line) > 5:
                        name = line[5:]
                elif line.startswith("#comment"):
                    if len(line) > 8:
                        comment = line[8:]
            else:
                break
    f.close()

    # print(f'columns: {columns}')
    # print(f'name: {name}')
    # print(f'comments: {comment}')

    swc = pd.read_csv(file_path, index_col=0, skiprows=n_skip, sep=" ",
                      usecols=[0, 1, 2, 3, 4, 5, 6],
                      names=columns)

    if vox_size_x is not None:
        swc.x = swc.x * vox_size_x

    if vox_size_y is not None:
        swc.y = swc.y * vox_size_y

    if vox_size_z is not None:
        swc.z = swc.z * vox_size_z

    swc_f = AxonTree(data=swc.copy(deep=True), name=name, comment=comment,
                     unit=unit)

    return swc_f


def get_segments_in_zrange(segments, zrange):
    """
    for a set of segments, get all the segments / segment chunks that
    are sliced within a certain z range
    :param segments: SegmentSet object
    :param zrange: array-like, 1d, two float, [zstart, zend]
                   zend should be larger than zstart
    :return: SegmentSet object, each segment is a component of self
             that lies in the zrange
    """
    segs_new = []

    for seg in segments:
        seg_new = Segment(seg).get_chunk_in_zrange(zrange=zrange)
        if seg_new is not None:
            segs_new.append(seg_new)

    if segs_new:
        return SegmentSet(np.array(segs_new))
    else:
        return None


class AxonTree(pd.DataFrame):
    """
    a swc like dataframe storing the structure of an
    axon tree, subclass from pandas.Dataframe

    column names:
        n: node index
        type: node type
        x: coordinate in x
        y: coordinate in y
        z: coordinate in z
        radius: radius of the node
        parent: index of the parent of this node

    with two extra attributes:
    name: str
    comment: str
    """

    _metadata = ['name', 'comment', 'unit']

    def __init__(self, name='', comment='', unit='', *args, **kwargs):

        super(AxonTree, self).__init__(*args, **kwargs)

        self.name = name
        self.comment = comment
        self.unit = unit

    def sort_node(self):

        parent = self['parent']

        min_n = min([n for n in self.index if n >= 0])
        self.index = self.index - min_n

        new_parent = []
        for p in parent:
            if p >= 0:
                new_parent.append(p - min_n)
            else:
                new_parent.append(p)

        self['parent'] = new_parent

        self.sort_index(inplace=True)

    def get_children(self, node_id):
        """
        return index of all children of the given node
        :param node_id: int
        :return: list of integers (index of all children)
        """
        return list(self[self['parent'] == node_id].index)

    def save_swc(self, save_path):

        if os.path.isfile(save_path):
            raise IOError('Cannot save swc file. File already exists.')

        self.to_csv(save_path, sep=" ")

        name_line = f'#name{self.name}'
        comment_line = f'#comment{self.comment}'
        column_line = f'##n,{",".join(self.columns)}'

        with open(save_path, 'r+') as f:
            content = f.readline()
            content = f.read()
            f.seek(0, 0)
            f.write(f'{name_line}\n{comment_line}\n{column_line}\n{content}')

        f.close()
        return

    def get_min_distances(self, swc):
        """
        for the every node in the input swc file, calculate the
        minimum distance from that node to all nodes in self.

        return a list of all these minimum distances.

        used for matching a small axon segment (swc, usually the
        segment with in vivo imaging data) to a fully reconstructed
        axon arbor (self).

        :param swc: AxonTree
        :return min_diss: 1d array
        """

        min_diss = []

        for _, seg_node in swc.iterrows():

            dx = self['x'] - seg_node['x']
            dy = self['y'] - seg_node['y']
            dz = self['z'] - seg_node['z']

            diss = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            min_diss.append(np.min(diss))

        return np.array(min_diss)

    def move_to_origin(self, new_origin):
        """
        move the structure to a new origin
        this happen in place

        :param new_origin: 1d array, [x, y, z] for the new origion
        :return: None
        """
        self.x = self.x - new_origin[0]
        self.y = self.y - new_origin[1]
        self.z = self.z - new_origin[2]

    def get_center(self):
        """
        return the center coordinates of all nodes
        :return center: 1d array, [x, y, z] of the average
                        coordinates of all nodes
        """

        return np.array([self.x.mean(), self.y.mean(), self.z.mean()])

    def scale(self, scale_x, scale_y, scale_z, unit=''):
        """
        scale self to standard unit basd on voxel size
        :return: None
        """

        self.x = self.x * scale_x
        self.y = self.y * scale_y
        self.z = self.z * scale_z

        self.unit = unit

    def rotate_xy(self, angle, is_rad=False, rot_center=(0.0, 0.0), direction='CW'):
        """
        rotate the structure around z axis (in xy plane) to align orientation

        note: the rotation is applied in the space with inverted y axis to
        match the plotting functions.

        :param angle: float, rotation angle
        :param is_rad: bool, if true, the "angle" is in radians
                             if false, the "angle" is in degrees
                       default False
        :param rot_center: list of two floats, [x, y] of rotation center, default is origin
        :param direction: str, "CW" (clockwise) or "CCW" (counterclockwise)
        :return: None
        """

        if not is_rad:
            ang_rad = angle * np.pi / 180
        else:
            ang_rad = angle

        xr = self.x - rot_center[0]
        yr = self.y - rot_center[1]
        dis_r = np.sqrt(xr ** 2 + yr ** 2)

        curr_ang = np.arctan2(yr, xr)

        if direction == 'CW':
            new_ang = curr_ang + ang_rad
        elif direction == 'CCW':
            new_ang = curr_ang - ang_rad
        else:
            raise ValueError(f'input "direction" should be "CW" or "CCW". got {direction}.')

        self.x = dis_r * np.cos(new_ang) + rot_center[0]
        self.y = dis_r * np.sin(new_ang) + rot_center[1]

    def get_segments(self):
        """
        get all segments of the tree in array format

        :return: SegmentSet object
                 ndarray, shape: n x 2 x 3.
                 first dimension: segments
                 second dimension: [parent, child]
                 third dimension: [x, y, z]
        """

        segs = []

        for node_i, node_row in self.iterrows():
            if node_row['parent'] != -1: # the nodes have a parent
                childxyz = np.array([node_row.x, node_row.y, node_row.z])
                parent_id = node_row['parent']
                parentxyz = np.array([self.loc[parent_id, 'x'],
                                      self.loc[parent_id, 'y'],
                                      self.loc[parent_id, 'z']])
                seg = np.array([parentxyz, childxyz])
                segs.append(seg)

        return SegmentSet(np.array(segs))

    # def get_segments(self):
    #     """
    #     :return: a list of Segment objects, all segments of self
    #     """
    #     segs = []
    #
    #     for node_i, node_row in self.iterrows():
    #         if node_row['parent'] != -1:  # the nodes have a parent
    #             childxyz = np.array([node_row.x, node_row.y, node_row.z])
    #             parent_id = node_row['parent']
    #             parentxyz = np.array([self.loc[parent_id, 'x'],
    #                                   self.loc[parent_id, 'y'],
    #                                   self.loc[parent_id, 'z']])
    #             seg = np.array([parentxyz, childxyz])
    #             segs.append(Segment(seg))
    #
    #     return segs
    #
    # def get_convex_hull_in_zrange(self, zrange):
    #     """
    #     this is not very precise ...
    #
    #     return the volume of a 3d convex hull between
    #     two different z depths, with points as nodes
    #
    #     :param zrange: array-like, two floats,
    #                     starting and end depth
    #     :return: scipy.spatial.ConvexHull object
    #     """
    #
    #     points = self[(self.z >= zrange[0]) & (self.z < zrange[1])][['x', 'y', 'z']]
    #     # print(points)
    #
    #     if len(points) <= 3:
    #         return None
    #     else:
    #         return sp.ConvexHull(points)

    def get_segments_in_z(self, zsteps):
        """
        get the segments / segment chunks that are sliced by a series
        of z ranges. zranges are defined by zsteps
        :param zsteps: array-like, 1d, float, bin edges in z
        :return segs_in_z: list of dictionaries
            each dictionay: {'zbin_name': str,
                             'zstart': float, start of zbin
                             'zend': float, end of zbin
                             'segments': list of Segment objects}
        """
        segs = self.get_segments()
        zbins = np.array([zsteps[:-1], zsteps[1:]])

        segs_in_z = []

        for bin_i in range(zbins.shape[1]):
            zrange = zbins[:, bin_i]
            curr_segs = get_segments_in_zrange(segments=segs, zrange=zrange)
            curr_dict = {'zbin_name': f'zbin{bin_i:06d}',
                         'zstart': zrange[0],
                         'zend': zrange[1],
                         'segments': curr_segs}
            segs_in_z.append(curr_dict)

        return segs_in_z

    def plot_3d_mpl(self, ax=None, color_dict=COLOR_DICT, *args, **kwargs):

        if ax is None:
            f = plt.figure(figsize=(8, 8))
            ax = f.add_subplot(111, projection='3d')

        for node_i, node_row in self.iterrows():
            if node_row.parent != -1:

                if color_dict is None:
                    curr_color = '#ff0000'
                else:
                    curr_color = color_dict[node_row.type]

                ax.plot([self.loc[node_row.parent, 'x'], node_row.x],
                        [self.loc[node_row.parent, 'y'], node_row.y],
                        [self.loc[node_row.parent, 'z'], node_row.z],
                        color=curr_color, *args, **kwargs)

        ax.invert_zaxis() # z from small to large (superficial to deep)
        ax.invert_yaxis() # y from small to large (anterior to posterior)
        # do not invert x axis, x from small to large (lateral to medial)
        ax.set_xlabel(f'x ({self.unit})')
        ax.set_ylabel(f'y ({self.unit})')
        ax.set_zlabel(f'z ({self.unit})')

        return ax

    def plot_xy_mpl(self, ax=None, color_dict=COLOR_DICT, *args, **kwargs):

        if ax is None:
            f= plt.figure(figsize=(8, 8))
            ax = f.add_subplot(111)

        for node_i, node_row in self.iterrows():
            if node_row.parent != -1:

                if color_dict is None:
                    curr_color = '#ff0000'
                else:
                    curr_color = color_dict[node_row.type]

                ax.plot([self.loc[node_row.parent, 'x'], node_row.x],
                        [self.loc[node_row.parent, 'y'], node_row.y],
                        color=curr_color, *args, **kwargs)

        ax.set_xlabel(f'x ({self.unit})')
        ax.set_ylabel(f'y ({self.unit})')
        ax.invert_yaxis()

        return ax

    def plot_xz_mpl(self, ax=None, color_dict=COLOR_DICT, *args, **kwargs):

        if ax is None:
            f= plt.figure(figsize=(8, 8))
            ax = f.add_subplot(111)

        for node_i, node_row in self.iterrows():
            if node_row.parent != -1:

                if color_dict is None:
                    curr_color = '#ff0000'
                else:
                    curr_color = color_dict[node_row.type]

                ax.plot([self.loc[node_row.parent, 'x'], node_row.x],
                        [self.loc[node_row.parent, 'z'], node_row.z],
                        color=curr_color, *args, **kwargs)

        ax.set_xlabel(f'x ({self.unit})')
        ax.set_ylabel(f'z ({self.unit})')
        ax.invert_yaxis()

        return ax

    def plot_yz_mpl(self, ax=None, color_dict=COLOR_DICT, *args, **kwargs):

        if ax is None:
            f= plt.figure(figsize=(8, 8))
            ax = f.add_subplot(111)

        for node_i, node_row in self.iterrows():
            if node_row.parent != -1:

                if color_dict is None:
                    curr_color = '#ff0000'
                else:
                    curr_color = color_dict[node_row.type]

                ax.plot([self.loc[node_row.parent, 'y'], node_row.y],
                        [self.loc[node_row.parent, 'z'], node_row.z],
                        color=curr_color, *args, **kwargs)

        ax.set_xlabel(f'y ({self.unit})')
        ax.set_ylabel(f'z ({self.unit})')
        ax.invert_xaxis()
        ax.invert_yaxis()

        return ax


class SegmentSet(np.ndarray):
    """subclass of np.ndarray representing a set of segments

    shape = (n, 2, 3)
    n = number of axon segments
    each segment is represented as:
    [[parent.x, parent.y, parent.z],
     [child.x,  child.y,  child.z ]]
    """

    def __new__(cls, input_array):

        if len(input_array.shape) != 3:
            raise ValueError('The input_array should be 3d.')

        if input_array.shape[1] != 2:
            raise ValueError('The input_array.shape[1] should be 2.')

        if input_array.shape[2] != 3:
            raise ValueError('The input_array.shape[2] should be 3.')

        obj = np.asarray(input_array.astype(np.float64)).view(cls)

        return obj

    def __array_finalize(self, obj):
        if obj is None:
            return

    def get_total_length(self):
        """
        :return: float, total length of this set of segments
        """

        return np.sum(
                    np.sqrt(
                        np.sum(
                            np.square(self[:, 0, :] -
                                      self[:, 1, :]), axis=1)))

    def get_xy_2d_hull(self):
        """

        :return: xyhull, 2d scipy.spatial.Qhull object, convex hull
            encompassed by all the nodes in this segment set
        """

        points = np.concatenate(self[:, :, 0:2], axis=0)

        # print(points)

        if points.shape[0] < 3:
            return None
        else:
            try:
                hull = sp.ConvexHull(points)
                return hull
            except sp.qhull.QhullError:
                return None

    def get_3d_hull(self):
        points = np.concatenate(self, axis=0)
        if points.shape[0] < 4:
            return None
        else:
            try:
                hull = sp.ConvexHull(points)
                return hull
            except sp.qhull.QhullError:
                return None


class Segment(np.ndarray):
    """
    subclass of np.ndarray representing a single linear segment

    shape = (2, 3)
    [[parent.x, parent.y, parent.z],
     [child.x,  child.y,  child.z ]]
    """

    def __new__(cls, input_array):

        if input_array.shape != (2, 3):
            raise ValueError('The shape of an Segment should be (2, 3).')

        obj = np.asarray(input_array.astype(np.float64)).view(cls)

        return obj

    def __array_finalize(self, obj):
        if obj is None:
            return

    @property
    def length(self):
        return np.sqrt(np.sum(np.square(self[0, :] - self[1, :]), axis=0))

    @property
    def xrange(self):
        return np.abs(self[0, 0] - self[1, 0])

    @property
    def yrange(self):
        return np.abs(self[0, 1] - self[1, 1])

    @property
    def zrange(self):
        return np.abs(self[0, 2] - self[1, 2])

    def get_z_ratio(self):
        """
        return the ration between the z span for each segment over
        segment length. (in this order it is more likely to avoid
        divided by zero error)

        this is for precise measurement of segment length at
        different depth.

        :return: float, ratio: length / z_span
        """
        return np.abs(self[0, 2] - self[1, 2]) / self.length

    def get_z_length_distribution(self, z_start, z_end, z_step):
        """
        given a set of bins in z (depth), return the length
        the segment in each bin
        :param z_start: float, starting depth in z
        :param z_end: float, ending depth in z
        :param z_step: float, bin width in z
                         bin_edges in depth is defined by
                         np.arange(z_start, z_end + z_step, z_step)
        :return bin_edges: 1d array, depth value of all bin edges
        :return z_dist: 1d array, length of this segment in each bin.
                        the length of z_dist should be 1 less than the
                        length of bin_edges
        """

        bin_edges = np.arange(z_start, z_end + z_step, z_step)
        z_dist = np.zeros(len(bin_edges) - 1)

        z_ratio = self.get_z_ratio()
        # if z_ratio == 0:
        #     print(self)

        ztop = np.min(self[:, 2])
        if ztop < bin_edges[0]:
            ztop = bin_edges[0]
        zbot = np.max(self[:, 2])
        if zbot > z_end:
            zbot = z_end - 1e-11 # to deal with edge cases


        topi = int((ztop - z_start) // z_step)
        boti = int((zbot - z_start) // z_step)

        if topi >= len(z_dist) or boti < 0:
            return bin_edges, z_dist

        if topi == boti: # the whole segment is in one bin
            z_dist[topi] = self.length
        else:
            # deal with the incomplete most superficial bin
            z_dist[topi] = (bin_edges[topi + 1] - ztop) / z_ratio

            # deal with the incomplete deepest bin
            z_dist[boti] = (zbot - bin_edges[boti]) / z_ratio

            # deal with all the complete bins in the middle
            z_dist[topi + 1:boti] = z_step / z_ratio

        return bin_edges, z_dist

    def get_chunk_in_zrange(self, zrange):
        """
        return the chunk of self that overlaps with the z range,
        z range: [zstart, zend],
                 inclusive on top, exclusive on end

        :param zrange: array-like, 1d, two floats, start and end of z range
        :return: None, if self is not in zrange
                 new Segment object (the part of self that is in z range)
                 if self has overlap with the z range
        """

        if zrange[1] <= zrange[0]:
            raise ValueError(f"input 'zrange' should be "
                             f"[zstart, zend] and zend should be larger"
                             f"than zstart. getting: {zrange}")

        # get ordered the segment object:
        # the first node on the top and the second node on the bottom
        if self[0, 2] > self[1, 2]:
            seg_ord = Segment(self[::-1, :])
        else:
            seg_ord = Segment(np.array(self))

        ztop = seg_ord[0, 2]
        zbot = seg_ord[1, 2]

        if zbot < zrange[0]:
            return None
        elif zbot == zrange[0]:
            if zbot == ztop:
                return seg_ord
            else:
                new_seg = Segment(np.array(seg_ord))
                new_seg[0, :] = new_seg[1, :]
                new_seg[1, 2] = new_seg[1, 2] + 1e-11
                return new_seg
        elif zbot < zrange[1]:
            if ztop < zrange[0]:
                new_x0 = (zrange[0] - ztop) * seg_ord.xrange / seg_ord.zrange + seg_ord[0, 0]
                new_y0 = (zrange[0] - ztop) * seg_ord.yrange / seg_ord.zrange + seg_ord[0, 1]
                return Segment(np.array([[new_x0, new_y0, zrange[0]],
                                             seg_ord[1, :]]))
            elif ztop >= zrange[0]:
                return seg_ord
        elif zbot == zrange[1]:
            if ztop < zrange[0]:
                new_x0 = (zrange[0] - ztop) * seg_ord.xrange / seg_ord.zrange + seg_ord[0, 0]
                new_y0 = (zrange[0] - ztop) * seg_ord.yrange / seg_ord.zrange + seg_ord[0, 1]
                return Segment(np.array([[new_x0, new_y0, zrange[0]],
                                             seg_ord[1, :]]))
            elif zrange[0] <= ztop < zbot:
                return seg_ord
            else:
                return None
        else:
            if ztop < zrange[0]:
                new_x0 = (zrange[0] - ztop) * seg_ord.xrange / seg_ord.zrange + seg_ord[0, 0]
                new_y0 = (zrange[0] - ztop) * seg_ord.yrange / seg_ord.zrange + seg_ord[0, 1]
                new_x1 = (zrange[1] - ztop) * seg_ord.xrange / seg_ord.zrange + seg_ord[0, 0]
                new_y1 = (zrange[1] - ztop) * seg_ord.yrange / seg_ord.zrange + seg_ord[0, 1]
                return Segment(np.array([[new_x0, new_y0, zrange[0]],
                                             [new_x1, new_y1, zrange[1]]]))
            elif zrange[0] <= ztop < zrange[1]:
                new_x1 = (zrange[1] - ztop) * seg_ord.xrange / seg_ord.zrange + seg_ord[0, 0]
                new_y1 = (zrange[1] - ztop) * seg_ord.yrange / seg_ord.zrange + seg_ord[0, 1]
                return Segment(np.array([seg_ord[0, :],
                                             [new_x1, new_y1, zrange[1]]]))
            else:
                return None
