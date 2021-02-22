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

    def test_get_segments(self):
        fpath = os.path.join(self.data_dir, 'test_swc_simple.swc')
        at = st.read_swc(fpath)
        segments = at.get_segments()
        print(segments)
        # print(segments.shape)
        assert (segments.shape == (4, 2, 3))

    def test_get_zratio(self):
        import numpy as np
        fpath = os.path.join(self.data_dir, 'test_swc_simple.swc')
        at = st.read_swc(fpath)
        segments = at.get_segments()

        lengths = []
        for segment in segments:
            seg = st.AxonSegment(segment)
            lengths.append(seg.length)

        # print(lengths)
        assert(np.allclose(lengths, [1, 1, 1, np.sqrt(3)], rtol=1e-16))

        zrs = []
        for segment in segments:
            seg = st.AxonSegment(segment)
            zrs.append(seg.get_z_ratio())

        # print(zrs)
        assert(np.allclose(zrs, [0, 0, 1, 1/np.sqrt(3)], rtol=1e-16))

    def test_get_z_length_distribution(self):

        import numpy as np
        fpath = os.path.join(self.data_dir, 'test_swc_simple.swc')
        at = st.read_swc(fpath)
        segments = at.get_segments()

        seg = st.AxonSegment(segments[0])
        # print(seg)

        bin_edges, z_dist = seg.get_z_length_distribution(z_start=-1.5, z_end=2.5,
                                                          z_step=1)
        # print(bin_edges)
        # print(z_dist)
        assert(np.allclose(bin_edges, (-1.5, -0.5, 0.5, 1.5, 2.5), rtol=1e-10))
        assert(np.allclose(z_dist, (0, 1, 0, 0), rtol=1e-10))

        seg2 = st.AxonSegment(segments[2])
        bin_edges, z_dist = seg2.get_z_length_distribution(z_start=0.2, z_end=0.8, z_step=0.1)
        # print(bin_edges)
        # print(z_dist)
        assert(np.allclose(bin_edges, (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), rtol=1e-10))
        assert (np.allclose(z_dist, (0.1, 0.1, 0.1, 0.1, 0.1, 0.1), rtol=1e-10))

        seg3 = st.AxonSegment(segments[3])
        bin_edges, z_dist = seg3.get_z_length_distribution(z_start=-0.5, z_end=3.5, z_step=1)
        # print(bin_edges)
        # print(z_dist)
        assert(np.allclose(z_dist, (0, np.sqrt(3)/2, np.sqrt(3)/2, 0), rtol=1e-10))
