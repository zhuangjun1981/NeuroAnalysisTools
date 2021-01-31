import os
import unittest
from .. import SwcTools as st

class TestTimingAnalysis(unittest.TestCase):

    def setUp(self):
        self.data_dir =os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    def test_read_swc(self):
        fpath = os.path.join(self.data_dir, 'test_swc3.swc')
        swc_f = st.read_swc(fpath)
        assert(len(swc_f) == 4)

    def test_plot(self):

        import matplotlib.pyplot as plt
        fpath = os.path.join(self.data_dir, 'test_swc3.swc')
        swc_f = st.read_swc(fpath)
        swc_f.plot_3d_mpl()
        swc_f.plot_xy_mpl()
        swc_f.plot_xz_mpl()
        swc_f.plot_yz_mpl()

        plt.show()


# curr_folder = os.path.dirname(os.path.realpath(__file__))
# os.chdir(curr_folder)
#
# fpath = os.path.join(os.path.dirname(os.path.dirname(curr_folder)), 'test', 'data', 'test_swc.swc')
#
# swc_f = st.read_swc(fpath)
# swc_f.scale(vox_size_x=0.414, vox_size_y=0.414, vox_size_z=0.5, unit='um') # scale to um
# cen = swc_f.get_center()
# swc_f.move_to_origin([cen[0], cen[1], 0]) # center x and y
# swc_f.type = 2 # change all nodes to be axon
# swc_f.radius = 1 # set radius of all nodes to be 1
# swc_f.y = -swc_f.y # match vaa3d orientation at "zero" rotation
#
# swc_f.plot_3d_mpl()
# swc_f.plot_xy_mpl()
# swc_f.plot_xz_mpl()
# swc_f.plot_yz_mpl()
#
# plt.show()