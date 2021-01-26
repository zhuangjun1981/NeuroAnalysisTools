'''
to read and analyze swc files from neuron
morphology reconstruction
'''

import os
import pandas as pd
import numpy as np


def read_swc(file_path, vox_size_x=None, vox_size_y=None, vox_size_z=None):
    """

    :param file_path: str, path to the swc file
    :param vox_size_x: float, voxel size in x (um)
    :param vox_size_y: float, voxel size in y (um)
    :param vox_size_z: float, voxel size in z (um)
    :return: SwcFile object
    """
    n_skip = 0

    columns = ["##n", "type", "z", "y", "x", "r", "parent"]
    name = ''
    comment = ''

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

    swc_f = SwcFile(data=swc.copy(deep=True), name=name, comment=comment)

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

    _metadata = ['name', 'comment']

    def __init__(self, name='', comment='', *args, **kwargs):

        super(SwcFile, self).__init__(*args, **kwargs)

        self.name = name
        self.comment = comment

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

