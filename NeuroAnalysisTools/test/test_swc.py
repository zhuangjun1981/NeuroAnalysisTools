import os, unittest
import numpy as np
from .. import SwcTools as st

class TestTimingAnalysis(unittest.TestCase):

    def setUp(self):
        self.data_dir =os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    def test_read_swc(self):
        fpath = os.path.join(self.data_dir, 'test_swc_simple.swc')
        swc_f = st.read_swc(fpath)
        assert(len(swc_f) == 5)

    def test_plot(self):

        import matplotlib.pyplot as plt
        fpath = os.path.join(self.data_dir, 'test_swc_simple.swc')
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
        # print(segments)
        # print(segments.shape)
        assert (segments.shape == (4, 2, 3))

    def test_get_zratio(self):
        fpath = os.path.join(self.data_dir, 'test_swc_simple.swc')
        at = st.read_swc(fpath)
        segments = at.get_segments()

        lengths = []
        for segment in segments:
            seg = st.Segment(segment)
            lengths.append(seg.length)

        # print(lengths)
        assert(np.allclose(lengths, [1, 1, 1, np.sqrt(3)], rtol=1e-16))

        zrs = []
        for segment in segments:
            seg = st.Segment(segment)
            zrs.append(seg.get_z_ratio())

        # print(zrs)
        assert(np.allclose(zrs, [0, 0, 1, 1/np.sqrt(3)], rtol=1e-16))

    def test_get_z_length_distribution(self):

        fpath = os.path.join(self.data_dir, 'test_swc_simple.swc')
        at = st.read_swc(fpath)
        segments = at.get_segments()

        seg = st.Segment(segments[0])
        # print(seg)

        bin_edges, z_dist = seg.get_z_length_distribution(z_start=-1.5, z_end=2.5,
                                                          z_step=1)
        # print(bin_edges)
        # print(z_dist)
        assert(np.allclose(bin_edges, (-1.5, -0.5, 0.5, 1.5, 2.5), rtol=1e-10))
        assert(np.allclose(z_dist, (0, 1, 0, 0), rtol=1e-10))

        seg2 = st.Segment(segments[2])
        bin_edges, z_dist = seg2.get_z_length_distribution(z_start=0.2, z_end=0.8, z_step=0.1)
        # print(bin_edges)
        # print(z_dist)
        assert(np.allclose(bin_edges, (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), rtol=1e-10))
        assert (np.allclose(z_dist, (0.1, 0.1, 0.1, 0.1, 0.1, 0.1), rtol=1e-10))

        seg3 = st.Segment(segments[3])
        bin_edges, z_dist = seg3.get_z_length_distribution(z_start=-0.5, z_end=3.5, z_step=1)
        # print(bin_edges)
        # print(z_dist)
        assert(np.allclose(z_dist, (0, np.sqrt(3)/2, np.sqrt(3)/2, 0), rtol=1e-10))

    # def test_convex_hull_between_depths(self):
    #
    #     fpath = os.path.join(self.data_dir, 'test_swc3.swc')
    #     at = st.read_swc(fpath)
    #     hull = at.get_convex_hull_between_depths(z_range=[-1, 200])
    #     # print(hull)
    #     assert(hull.volume == 500000)

    def test_get_chunk_in_z_range(self):

        seg = st.Segment(np.array([[1, 2, 3],
                                       [0, 0, 0]], dtype=np.float64))

        seg0 = seg.get_chunk_in_zrange(zrange=[-0.5, -0.1])
        assert(seg0 is None)

        seg1 = seg.get_chunk_in_zrange(zrange=[-0.5, 0])
        assert(seg1 is None)

        seg2 = seg.get_chunk_in_zrange(zrange=[-0.5, 1])
        assert(np.allclose(seg2, [[0, 0, 0], [1/3, 2/3, 1]], rtol=1e-10))

        seg3 = seg.get_chunk_in_zrange(zrange=[0, 1])
        assert(np.allclose(seg3, [[0, 0, 0], [1/3, 2/3, 1]], rtol=1e-10))

        seg4 = seg.get_chunk_in_zrange(zrange=[0.5, 1])
        assert(np.allclose(seg4, [[1/6, 1/3, 0.5], [1/3, 2/3, 1]], rtol=1e-10))

        seg5 = seg.get_chunk_in_zrange(zrange=[1, 3])
        assert (np.allclose(seg5, [[1/3, 2/3, 1], [1, 2, 3]], rtol=1e-10))

        seg6 = seg.get_chunk_in_zrange(zrange=[2, 4])
        assert (np.allclose(seg6, [[2/3, 4/3, 2], [1, 2, 3]], rtol=1e-10))

        seg7 = seg.get_chunk_in_zrange(zrange=[3, 4])
        assert (np.allclose(seg7, [[1, 2, 3], [1, 2, 3]], rtol=1e-10))

        seg8 = seg.get_chunk_in_zrange(zrange=[3.5, 4])
        assert (seg8 is None)

        seg9 = st.Segment(np.array([[1, 2, 0],
                                        [0, 0, 0]], dtype=np.float64))

        seg10 = seg9.get_chunk_in_zrange(zrange=[-1, -0.5])
        assert(seg10 is None)

        seg11 = seg9.get_chunk_in_zrange(zrange=[-1, 0])
        assert(seg11 is None)

        seg12 = seg9.get_chunk_in_zrange(zrange=[-1, 1])
        assert (np.allclose(seg12, [[1, 2, 0], [0, 0, 0]], rtol=1e-10))

        seg13 = seg9.get_chunk_in_zrange(zrange=[0, 1])
        assert (np.allclose(seg13, [[1, 2, 0], [0, 0, 0]], rtol=1e-10))

        seg14 = seg9.get_chunk_in_zrange(zrange=[1, 2])
        assert (seg14 is None)

    def test_get_segments_in_z(self):
        fpath = os.path.join(self.data_dir, 'test_swc_simple.swc')
        at = st.read_swc(fpath)
        # print(at)
        zsteps = np.arange(-1, 3, 0.5, dtype=np.float64)
        # print(f'zsteps: {zsteps}')
        segs_in_z = at.get_segments_in_z(zsteps=zsteps)

        # _ = [print(f'\n{segs["zbin_name"]}: '
        #            f'[{segs["zstart"]}, {segs["zend"]}], '
        #            f'\n\t{segs["segments"]}')
        #      for segs in segs_in_z]

        assert (segs_in_z[0]['segments'] is None)
        assert (segs_in_z[1]['segments'] is None)
        assert (np.allclose(segs_in_z[2]['segments'],
                            np.array([[[0., 0., 0.],
                                       [1., 0., 0.]],
                                      [[1., 0., 0.],
                                       [1., 1., 0.]],
                                      [[1., 1., 0.],
                                       [1., 1., 0.5]]]),
                            rtol=1e-10))
        assert (np.allclose(segs_in_z[3]['segments'],
                            np.array([[[1., 1., 0.5],
                                       [1., 1., 1. ]]]),
                            rtol=1e-10))
        assert (np.allclose(segs_in_z[4]['segments'],
                            np.array([[[1., 1., 1.],
                                       [1., 1., 1.]],
                                      [[1., 1., 1.],
                                       [1.5, 1.5, 1.5]]]),
                            rtol=1e-10))
        assert (np.allclose(segs_in_z[5]['segments'],
                            np.array([[[1.5, 1.5, 1.5],
                                       [2. , 2. , 2. ]]]),
                            rtol=1e-10))
        assert (np.allclose(segs_in_z[6]['segments'],
                            np.array([[[2., 2., 2.],
                                       [2., 2., 2.]]]),
                            rtol=1e-10))

    def test_get_lengths(self):
        fpath = os.path.join(self.data_dir, 'test_swc_simple.swc')
        at = st.read_swc(fpath)
        segs = at.get_segments()
        lens = segs.get_lengths()
        assert(np.allclose(lens, [1, 1, 1, np.sqrt(3)], rtol=1e-10))

    def test_get_centers(self):
        fpath = os.path.join(self.data_dir, 'test_swc_simple.swc')
        at = st.read_swc(fpath)
        segs = at.get_segments()
        cens = segs.get_centers()
        assert(np.allclose(cens, [[0.5, 0, 0],
                                  [1, 0.5, 0],
                                  [1, 1, 0.5],
                                  [1.5, 1.5, 1.5]], rtol=1e-10))

    def test_get_weighted_center(self):
        fpath = os.path.join(self.data_dir, 'test_swc_simple.swc')
        at = st.read_swc(fpath)
        segs = at.get_segments()
        cen = segs.get_weighted_center()

        sq3 = np.sqrt(3)

        x = (2.5 + 1.5 * sq3) / (3 + sq3)
        y = (1.5 + 1.5 * sq3) / (3 + sq3)
        z = (0.5 + 1.5 * sq3) / (3 + sq3)

        assert(np.allclose(cen, [x, y, z], rtol=1e-10))

    def test_get_distances_to_weighted_center(self):
        arr = [[[0., 0., 0.],
                [1., 0., 0.]],
               [[1., 0., 0.],
                [1., 1., 0.]],
               [[1., 1., 0.],
                [1., 1., 1.]]]

        segs = st.SegmentSet(np.array(arr))
        cens = segs.get_centers()
        cen = segs.get_weighted_center()

        dis0 = np.sqrt((0.5 - 2.5 / 3) ** 2 + (0.0 - 1.5 / 3) ** 2)
        dis1 = np.sqrt((1.0 - 2.5 / 3) ** 2 + (0.5 - 1.5 / 3) ** 2)
        dis2 = np.sqrt((1.0 - 2.5 / 3) ** 2 + (1.0 - 1.5 / 3) ** 2)

        diss = segs.get_distances_to_weighted_center_xy()
        # print(diss)
        # print([dis0, dis1, dis2])
        assert(np.allclose(diss, [dis0, dis1, dis2], rtol=1e-10))

    def test_get_total_length(self):

        axonss = st.SegmentSet(np.array([[[0, 0, 0],
                                              [1, 1, 1]],
                                             [[0, 1, 2],
                                              [3, 4, 5]]]))
        assert(np.allclose(axonss.get_total_length(),
                           np.sqrt(3) + np.sqrt(27),
                           rtol=1e-10))

    def test_get_xy_2d_hull(self):

        fpath = os.path.join(self.data_dir, 'test_swc_simple.swc')
        at = st.read_swc(fpath)

        # print(at)

        segs_dicts = at.get_segments_in_z(zsteps=[-5, -1, 3, 7])
        segs = segs_dicts[1]['segments']
        # print(segs)

        hull = segs.get_xy_2d_hull()
        assert(np.allclose(hull.volume, 1., rtol=1e-10))
