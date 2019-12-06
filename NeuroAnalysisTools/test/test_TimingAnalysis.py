__author__ = 'junz'

import unittest
import numpy as np
from ..core import TimingAnalysis as ta
# import matplotlib.pyplot as plt

class TestTimingAnalysis(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_onset_time_stamps(self):
        trace = np.array(([0.] * 5 + [5.] * 5) * 5)
        ts = ta.get_onset_timeStamps(trace, Fs=10000., onsetType='raising')
        assert(ts[2] == 0.0025)
        ts2 = ta.get_onset_timeStamps(trace, Fs=10000., onsetType='falling')
        assert(ts2[2] == 0.0030)

    def test_get_burst(self):
        spikes = [0.3, 0.5, 0.501, 0.503, 0.505, 0.65, 0.7, 0.73, 0.733, 0.734, 0.735, 0.9, 1.5, 1.6,
                  1.602, 1.603, 1.605, 1.94, 1.942]

        _, burst_ind = ta.get_burst(spikes, pre_isi=(-np.inf, -0.1), inter_isi=0.004, spk_num_thr=2)
        assert(np.array_equal(burst_ind, [[1, 4], [13, 4], [17, 2]]))

        _, burst_ind = ta.get_burst(spikes, pre_isi=(-np.inf, -0.01), inter_isi=0.004, spk_num_thr=2)
        assert (np.array_equal(burst_ind, [[1, 4], [7, 4], [13, 4], [17, 2]]))

        _, burst_ind = ta.get_burst(spikes, pre_isi=(-np.inf, -0.01), inter_isi=0.002, spk_num_thr=2)
        assert (np.array_equal(burst_ind, [[1, 2]]))

        _, burst_ind = ta.get_burst(spikes, pre_isi=(-np.inf, -0.01), inter_isi=0.004, spk_num_thr=3)
        assert (np.array_equal(burst_ind, [[1, 4], [7, 4], [13, 4]]))

    def test_get_event_with_pre_iei(self):
        ts = np.arange(100) + np.random.rand(100) * 0.4
        ts2 = ta.get_event_with_pre_iei(ts, iei=0.8)
        assert(np.min(np.diff(ts2)) > 0.8)

        ts3 = np.arange(20)
        ts3[5] = 6
        ts3[7] = 6
        ts3[1] = 0
        ts4 = ta.get_event_with_pre_iei(ts3, iei=0.5)
        assert(np.array_equal(ts4, np.array([2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])))
        ts5 = ta.get_event_with_pre_iei(ts3, iei=1)
        assert (np.array_equal(ts5, np.array([2, 6, 8])))
        np.random.shuffle(ts3)
        ts6 = ta.get_event_with_pre_iei(ts3, iei=0.5)
        print(ts3)
        print(ts6)
        assert (np.array_equal(ts6, np.array([2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])))
        ts7 = ta.get_event_with_pre_iei(ts3, iei=1)
        assert (np.array_equal(ts7, np.array([2, 6, 8])))

    def test_find_nearest(self):
        trace = np.arange(10)
        assert(ta.find_nearest(trace, 1.4) == 1)
        assert(ta.find_nearest(trace, 1.6) == 2)
        assert(ta.find_nearest(trace, 1.4, -1) == 1)
        assert(ta.find_nearest(trace, 1.6, -1) == 1)
        assert(ta.find_nearest(trace, 1.4, 1) == 2)
        assert(ta.find_nearest(trace, 1.6, 1) == 2)
        assert(ta.find_nearest(trace, -1, -1) is None)
        assert(ta.find_nearest(trace, 11, 1) is None)

    def test_discrete_cross_correlation(self):
        ts1 = np.arange(10)
        ts2 = ts1 + 0.5
        t, cc = ta.discrete_cross_correlation(ts1, ts2, t_range=(-1., 1.), bins=10, isPlot=False)
        assert(np.array_equal(cc, np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0])))

        t, cc = ta.discrete_cross_correlation(ts1, ts2, t_range=(-1., 1.), bins=20, isPlot=False)
        assert(np.array_equal(cc, np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                                            0., 0.])))
        ts2 = np.hstack((ts2, ts2 + 0.01)).flatten()
        t, cc = ta.discrete_cross_correlation(ts1, ts2, t_range=(-1., 1.), bins=20, isPlot=False)
        assert (np.array_equal(cc, np.array([0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0.,
                                             0., 0.])))

    def test_haramp(self):
        t = np.arange(1000) * 0.001
        trace = np.sin(t * 2 * 2 * np.pi) + 1
        har = ta.haramp(trace=trace, periods=1, ceil_f=3)
        assert (len(har) == 3)
        assert (har[2] / har[0] == 1.)

        trace = np.zeros(1000)
        trace[np.array([0, 249, 499, 749])] = 1.
        har = ta.haramp(trace=trace, periods=4, ceil_f=4)
        assert (len(har) == 4)
        assert (round(1000. * har[1] / har[0]) / 1000. == 2.)

    def test_threshold_to_intervals(self):
        trace = np.array([2.3, 4.5, 6.7, 5.5, 3.3, 9.2, 4.4, 3.2, 1.0, 0.8, 5.5])

        intvs1 = ta.threshold_to_intervals(trace=trace, thr=5.0, comparison='>=')
        # print(intvs1)
        for intv in intvs1:
            # print(trace[intv[0]: intv[1]])
            assert(np.min(trace[intv[0]: intv[1]]) >= 5.0)

        intvs2 = ta.threshold_to_intervals(trace=trace, thr=3.0, comparison='<')
        for intv in intvs2:
            # print(trace[intv[0]: intv[1]])
            assert(np.max(trace[intv[0]: intv[1]]) < 3.0)


if __name__ == '__main__':
    TestTimingAnalysis.test_get_onset_time_stamps()
    TestTimingAnalysis.test_get_burst()
    TestTimingAnalysis.test_find_nearest()
    TestTimingAnalysis.test_get_event_with_pre_iei()