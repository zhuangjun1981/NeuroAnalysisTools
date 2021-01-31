'''
to read and analyze swc files from neuron
morphology reconstruction
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

COLOR_DICT = {
    2 : '#ff0000', # axon?
    3 : '#0000ff', # apical dendrite?
    4 : '#ff0000', # basal dendrite?
              }


def read_swc(file_path, vox_size_x=None, vox_size_y=None, vox_size_z=None, unit=''):
    """

    :param file_path: str, path to the swc file
    :param vox_size_x: float, voxel size in x
    :param vox_size_y: float, voxel size in y
    :param vox_size_z: float, voxel size in z
    :param unit: str, unit of voxel sizes
    :return: SwcFile object
    """
    n_skip = 0

    columns = ["##n", "type", "z", "y", "x", "r", "parent"]
    name = ''
    comment = ''
    unit = ''

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

    swc_f = SwcFile(data=swc.copy(deep=True), name=name, comment=comment,
                    unit=unit)

    return swc_f


class SwcFile(pd.DataFrame):
    """
    swc file object, subclass from pandas.Dataframe

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

        super(SwcFile, self).__init__(*args, **kwargs)

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

    # def sort_node_deep(self, start_node_ind):
    #     """
    #     resort the node of the tree using
    #     breadth first search, assuming no directions between nodes
    #     :param start_node_ind: int, the index of start node
    #     :return: a new SwcFile object with sorted node
    #     """
    #     #todo: finish this
    #
    #     # initiate first node
    #     new_node_id = 0
    #     new_node_lst = [list(self[start_node_ind, :])]
    #     new_node_lst[0][0] = new_node_id
    #     new_node_lst[0][-1] = -1
    #
    #     self.go_through_next_level(
    #         df=self,
    #         node_id=start_node_ind,
    #         new_node_id=new_node_id,
    #         new_node_lst=new_node_lst)

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

        :param swc: SwcFile
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

    def scale(self, vox_size_x, vox_size_y, vox_size_z, unit=''):
        """
        scale self to standard unit basd on voxel size
        :return: None
        """

        self.x = self.x * vox_size_x
        self.y = self.y * vox_size_y
        self.z = self.z * vox_size_z

        self.unit = unit

    def plot_3d_mpl(self, ax=None, color_dict=COLOR_DICT):

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
                        color=curr_color)

        ax.invert_zaxis()
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
        ax.invert_yaxis()

        return ax

