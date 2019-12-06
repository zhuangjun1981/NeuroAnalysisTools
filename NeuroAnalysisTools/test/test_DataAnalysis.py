__author__ = 'junz'

import unittest
import numpy as np
from ..core import DataAnalysis as da


class TestTimingAnalysis(unittest.TestCase):

    def setUp(self):
        pass

    def test_interpolate_nans(self):
        y = np.array([0., 1., 2., 3., np.nan, np.nan, 6., 7., np.nan, 9.])
        y1 = da.interpolate_nans(y)
        # print(y1)
        # print(np.arange(10.))
        assert (np.array_equal(y1, np.arange(10.)))

    def test_downsample(self):
        y = np.arange(10)
        y1 = da.downsample(arr=y, rate=2, method=np.mean)
        print(y1)
        assert (np.array_equal(y1, np.arange(0.5, 9, 2.)))

    def test_get_clustering_distances(self):

        mat_dis = np.ones((5, 5))
        mat_dis[0, 2] = 0.1
        mat_dis[2, 0] = 0.1
        mat_dis[3, 4] = 0.2
        mat_dis[4, 3] = 0.2
        clu = [1, 0, 1, 2, 2]

        dis_clu, dis_non_clu = da.get_clustering_distances(mat_dis=mat_dis, cluster=clu)

        assert(np.array_equal(dis_non_clu, np.ones((8,), dtype=np.float64)))
        assert(np.array_equal(dis_clu[0], np.array([], dtype=np.float64)))
        assert(np.array_equal(dis_clu[1], np.array([0.1], dtype=np.float64)))
        assert(np.array_equal(dis_clu[2], np.array([0.2], dtype=np.float64)))

