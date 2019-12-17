__author__ = 'junz'

import os
import unittest

import h5py
import numpy as np

# from .. import SingleCellAnalysis as sca
# from ..core import FileTools as ft
# from ..core import ImageAnalysis as ia

import NeuroAnalysisTools.core.ImageAnalysis as ia
import NeuroAnalysisTools.core.FileTools as ft
import NeuroAnalysisTools.SingleCellAnalysis as sca


class TestSingleCellAnalysis(unittest.TestCase):

    def setUp(self):

        currFolder = os.path.dirname(os.path.realpath(__file__))
        self.testDataFolder = os.path.join(currFolder, 'data')

        self.sparseNoiseDisplayLogPath = os.path.join(self.testDataFolder, 'SparseNoiseDisplayLog.pkl')
        self.testH5Path = os.path.join(self.testDataFolder, 'test.hdf5')
        self.STRFDataPath = os.path.join(self.testDataFolder, 'cellsSTRF.hdf5')

    def test_mergeROIs(self):
        roi1 = ia.WeightedROI(np.arange(9).reshape((3, 3)))
        roi2 = ia.WeightedROI(np.arange(1, 10).reshape((3, 3)))

        merged_ROI = sca.merge_weighted_rois(roi1, roi2)
        merged_ROI2 = sca.merge_binary_rois(roi1, roi2)

        assert (np.array_equal(merged_ROI.get_weighted_mask(), np.arange(1, 18, 2).reshape((3, 3))))
        assert (np.array_equal(merged_ROI2.get_binary_mask(), np.ones((3, 3))))

    def test_getSparseNoiseOnsetIndex(self):

        display_log = ft.loadFile(self.sparseNoiseDisplayLogPath)

        # print(display_log.keys())

        allOnsetInd, onsetIndWithLocationSign = sca.get_sparse_noise_onset_index(display_log)

        # print(list(allOnsetInd[0:10]))
        # print(onsetIndWithLocationSign[2])

        assert (list(allOnsetInd[0:10]) == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54])
        for probe in onsetIndWithLocationSign:
            if np.array_equal(probe[0], np.array([0., 70.])) and probe[1] == -1.:
                # print(probe[2])
                assert(np.array_equal(probe[2], np.array([960, 1158, 2280, 3816, 4578, 5586, 6546, 7008, 8496, 9270])))

    def test_SpatialTemporalReceptiveField_from_h5_group(self):
        f = h5py.File(self.STRFDataPath, 'r')
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
        trace = np.array(STRF.data['traces'][20])
        assert ((float(trace[4, 8]) + 0.934942364693) < 1e-10)
        # STRF.plot_traces(figSize=(15,10),yRange=[-5,50],columnSpacing=0.002,rowSpacing=0.002)

    def test_SpatialTemporalReceptiveField(self):
        locations = [[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0], [3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0]]
        signs = [1, 1, 1, 1, -1, -1, -1, -1]
        traces = [[np.arange(4)], [np.arange(1, 5)], [np.arange(2, 6)], [np.arange(3, 7)], [np.arange(5, 9)],
                  [np.arange(6, 10)],
                  [np.arange(7, 11)], [np.arange(8, 12)]]
        traces = [np.array(t) for t in traces]
        time = np.arange(4, 8)
        STRF = sca.SpatialTemporalReceptiveField(locations, signs, traces, time)

        # print(signs)
        # print(locations)
        # print(traces)
        # print(STRF.data)

        assert (STRF.data['traces'].iloc[0][0, 1] == 8)
        assert (STRF.data['sign'].loc[4] == -1)
        newLocations = [[location[0] + 1, location[1] + 1] for location in locations[0:4]]
        newSigns = [1, 1, 1, 1]
        STRF.add_traces(newLocations, newSigns, traces[0:4])
        assert (STRF.data['traces'][7][1][2] == 4)
        # _ = STRF.plot_traces()

    def test_SpatialTemporalReceptiveField_IO(self):
        locations = [[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0], [3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0]]
        signs = [1, 1, 1, 1, -1, -1, -1, -1]
        traces = [[np.arange(4)], [np.arange(1, 5)], [np.arange(2, 6)], [np.arange(3, 7)], [np.arange(5, 9)],
                  [np.arange(6, 10)], [np.arange(7, 11)], [np.arange(8, 12)]]
        time = np.arange(4, 8)

        STRF = sca.SpatialTemporalReceptiveField(locations, signs, traces, time)

        # print(STRF.data)

        if os.path.isfile(self.testH5Path):
            os.remove(self.testH5Path)

        testFile = h5py.File(self.testH5Path, 'a')
        STRFGroup = testFile.create_group('spatial_temporal_receptive_field')
        STRF.to_h5_group(STRFGroup)
        testFile.close()

        h5File = h5py.File(self.testH5Path, 'r')
        STRF2 = sca.SpatialTemporalReceptiveField.from_h5_group(h5File['spatial_temporal_receptive_field'])
        h5File.close()

        assert (STRF2.data.altitude.equals(STRF.data.altitude))
        assert (STRF2.data.azimuth.equals(STRF.data.azimuth))
        assert (STRF2.data.sign.equals(STRF.data.sign))

        # print(STRF.data.traces)
        # print(STRF2.data.traces)

        assert (np.array_equal(np.array([np.array(t) for t in STRF.data.traces]),
                               np.array([np.array(t) for t in STRF2.data.traces])))

    def test_SpatialTemporalReceptiveField_getAmpLitudeMap(self):
        f = h5py.File(self.STRFDataPath, 'r')
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
        ampON, ampOFF, altPos, aziPos = STRF.get_amplitude_map(timeWindow=[0, 0.5])
        assert (ampON[7, 10] - (-0.0258248019964) < 1e-10)
        # print(ampOFF[8, 9])
        # print(altPos)
        # print(aziPos)
        assert (ampOFF[8, 9] - (0.8174604177474976) < 1e-10)
        assert (altPos[5] == 30.)
        assert (aziPos[3] == -5.)
        # sca.plot_2d_receptive_field(ampON,altPos,aziPos,cmap='gray_r',interpolation='nearest')

    def test_SpatialTemporalReceptiveField_getZscoreMap(self):
        f = h5py.File(self.STRFDataPath, 'r')
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
        zscoreON, zscoreOFF, altPos, aziPos = STRF.get_zscore_map(timeWindow=[0, 0.5])
        assert (zscoreON[7, 10] - (-0.070735671412) < 1e-10)
        # print(zscoreOFF[8, 9])
        assert (zscoreOFF[8, 9] - (1.09214839758106) < 1e-10)
        # sca.plot_2d_receptive_field(ampON,altPos,aziPos,cmap='gray_r',interpolation='nearest')

    def test_SpatialTemporalReceptiveField_getCenters(self):
        f = h5py.File(self.STRFDataPath, 'r')
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])

        zon, zoff, _, alts, azis = STRF.get_zscore_rois(timeWindow=[0, 0.5], zscoreThr=2)
        # print(zon.weights)
        # print(zon.pixels)
        # print(zoff.weights)
        # print(zoff.pixels)
        # print(alts)
        # print(azis)

        # print(STRF.get_zscore_roi_centers(timeWindow=[0, 0.5], zscoreThr=2)[0][0])
        # print(STRF.get_zscore_roi_centers(timeWindow=[0, 0.5], zscoreThr=2)[0][1])
        assert (STRF.get_zscore_roi_centers()[1][1] == 45.)
        assert (STRF.get_zscore_roi_centers()[1][0] == -30.)

    def test_SpatialTemporalReceptiveField_getAmplitudeReceptiveField(self):
        f = h5py.File(self.STRFDataPath, 'r')
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
        ampRFON, ampRFOFF = STRF.get_amplitude_receptive_field()

        assert (ampRFON.sign == 'ON')
        assert (ampRFOFF.sign == 'OFF')
        assert (ampRFOFF.get_weighted_mask()[7, 9] - 3.2014527 < 1e-7)

    def test_SpatialTemporalReceptiveField_getZscoreReceptiveField(self):
        f = h5py.File(self.STRFDataPath, 'r')
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
        zscoreRFON, zscoreRFOFF = STRF.get_zscore_receptive_field()
        assert (zscoreRFON.sign == 'ON');
        assert (zscoreRFOFF.sign == 'OFF')
        assert (zscoreRFOFF.get_weighted_mask()[7, 9] - 1.3324414 < 1e-7)

    def test_SpatialTemporalReceptiveField_shrink(self):
        f = h5py.File(self.STRFDataPath, 'r')
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
        STRF.shrink([-10, 10], None)
        assert (np.array_equal(np.array(STRF.data.altitude.unique()), np.array([-10., -5., 0., 5., 10.])))
        STRF.shrink(None, [0, 20])
        assert (np.array_equal(np.array(STRF.data.azimuth.unique()), np.array([0., 5., 10., 15., 20.])))

    def test_SpatialReceptiveField(self):
        SRF = sca.SpatialReceptiveField(np.arange(9).reshape((3, 3)), np.arange(3), np.arange(3))
        assert (np.array_equal(SRF.weights, np.arange(1, 9)))

    def test_SpatialReceptiveField_thresholdReceptiveField(self):
        SRF = sca.SpatialReceptiveField(np.arange(9).reshape((3, 3)), np.arange(3), np.arange(3))
        thresholdedSRF = SRF.threshold(thr=4)
        assert (np.array_equal(thresholdedSRF.weights, np.arange(4, 9)))

    def test_SpatialReceptiveField_interpolate(self):
        SRF = sca.SpatialReceptiveField(np.random.rand(5, 5), np.arange(5)[::-1], np.arange(5))
        SRF_i = SRF.interpolate(5)
        assert (SRF_i.get_weighted_mask().shape == (20, 20))

    def test_get_orientation_properties(self):
        import pandas as pd
        dires = np.arange(8) * 45
        resps = np.ones(8)
        resps[2] = 2.
        dire_tuning = pd.DataFrame()
        dire_tuning['dire'] = dires
        dire_tuning['resp_mean'] = resps
        # print(dire_tuning)

        OSI_raw, DSI_raw, gOSI_raw, gDSI_raw, OSI_ele, DSI_ele, gOSI_ele, \
        gDSI_ele, OSI_rec, DSI_rec, gOSI_rec, gDSI_rec, peak_dire_raw, vs_dire_raw, \
        vs_dire_ele, vs_dire_rec = sca.DriftingGratingResponseTable.get_dire_tuning_properties(dire_tuning=dire_tuning,
                                                                                               response_dir='pos',
                                                                                               elevation_bias=0.)

        # print('\nOSI_raw: {}'.format(OSI_raw))
        # print('DSI_raw: {}'.format(DSI_raw))
        # print('gOSI_raw: {}'.format(gOSI_raw))
        # print('gDSI_raw: {}'.format(gDSI_raw))
        # print('\nOSI_ele: {}'.format(OSI_ele))
        # print('DSI_ele: {}'.format(DSI_ele))
        # print('gOSI_ele: {}'.format(gOSI_ele))
        # print('gDSI_ele: {}'.format(gDSI_ele))
        # print('\nOSI_rec: {}'.format(OSI_rec))
        # print('DSI_rec: {}'.format(DSI_rec))
        # print('gOSI_rec: {}'.format(gOSI_rec))
        # print('gDSI_rec: {}'.format(gDSI_rec))
        # print('\npeak_dire_raw: {}'.format(peak_dire_raw))
        # print('\nvs_dire_raw: {}'.format(vs_dire_raw))
        # print('vs_orie_raw: {}'.format(vs_orie_raw))
        # print('\nvs_dire_ele: {}'.format(vs_dire_ele))

        assert (OSI_raw == OSI_ele == OSI_rec)
        assert (abs(OSI_raw - 0.3333333333333333) < 1e-10)
        assert (DSI_raw == DSI_ele == DSI_rec)
        assert (abs(DSI_raw - 0.3333333333333333) < 1e-10)

        assert (gOSI_raw == gOSI_ele == gOSI_rec)
        assert (abs(gOSI_raw - 0.111111111111111) < 1e-10)
        assert (gDSI_raw == gDSI_ele == gDSI_rec)
        assert (abs(gDSI_raw - 0.111111111111111) < 1e-10)

        assert (peak_dire_raw == int(vs_dire_raw) == int(vs_dire_ele) == int(vs_dire_rec) == 90)

        dire_tuning.loc[6, 'resp_mean'] = -1.
        # print(dire_tuning)

        OSI_raw, DSI_raw, gOSI_raw, gDSI_raw, OSI_ele, DSI_ele, gOSI_ele, \
        gDSI_ele, OSI_rec, DSI_rec, gOSI_rec, gDSI_rec, peak_dire_raw, vs_dire_raw, \
        vs_dire_ele, vs_dire_rec = sca.DriftingGratingResponseTable.get_dire_tuning_properties(dire_tuning=dire_tuning,
                                                                                               response_dir='pos',
                                                                                               elevation_bias=0.)

        # print('\nOSI_raw: {}'.format(OSI_raw))
        # print('DSI_raw: {}'.format(DSI_raw))
        # print('gOSI_raw: {}'.format(gOSI_raw))
        # print('gDSI_raw: {}'.format(gDSI_raw))
        # print('\nOSI_ele: {}'.format(OSI_ele))
        # print('DSI_ele: {}'.format(DSI_ele))
        # print('gOSI_ele: {}'.format(gOSI_ele))
        # print('gDSI_ele: {}'.format(gDSI_ele))
        # print('\nOSI_rec: {}'.format(OSI_rec))
        # print('DSI_rec: {}'.format(DSI_rec))
        # print('gOSI_rec: {}'.format(gOSI_rec))
        # print('gDSI_rec: {}'.format(gDSI_rec))
        # print('\npeak_dire_raw: {}'.format(peak_dire_raw))
        # print('\nvs_dire_raw: {}'.format(vs_dire_raw))
        # print('\nvs_dire_ele: {}'.format(vs_dire_ele))
        # print('\nvs_dire_rec: {}'.format(vs_dire_rec))

        assert (OSI_raw == OSI_rec)
        assert(abs(OSI_raw - 1. / 3.) < 1e-10)
        assert (DSI_raw == 3.0)
        assert (DSI_ele == DSI_rec == 1.0)
        assert (OSI_ele == 0.2)
        assert (abs(gOSI_raw - 1. / 7.) < 1e-10)
        assert (abs(gDSI_raw - 3. / 7.) < 1e-10)
        assert (abs(gOSI_ele - 1. / 15.) < 1e-10)
        assert (abs(gDSI_ele - 3. / 15.) < 1e-10)
        assert (gOSI_rec == 0.)
        assert (abs(gDSI_rec -2. / 8.) < 1e-10)

        assert (peak_dire_raw == int(vs_dire_raw) == int(vs_dire_ele) == int(vs_dire_rec) == 90)

    def test_SpatialReceptiveField_reverse_locations(self):

        alts = [-2., -1., 0., 1., 2.]
        azis = [-2., -1., 0., 1., 2.]

        map = np.array([[0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 1., 1., 0.],
                        [0., 0., 1., 1., 0.],
                        [0., 0., 0., 0., 0.]])

        srf = sca.SpatialReceptiveField(mask=map, altPos=alts, aziPos=azis, sign='ON')
        # print(srf.get_weighted_rf_center())
        assert(srf.get_weighted_rf_center() == [0.5, 0.5])

        srf2 = sca.SpatialReceptiveField(mask=map, altPos=alts[::-1], aziPos=azis, sign='ON')
        assert(srf2.get_weighted_rf_center() == [-0.5, 0.5])

        srf_f = srf.gaussian_filter(sigma=1)
        # print(srf_f.get_weighted_rf_center())

        srf2_f = srf2.gaussian_filter(sigma=1)
        # print(srf2_f.get_weighted_rf_center())
        assert (abs(srf_f.get_weighted_rf_center()[0] + srf2_f.get_weighted_rf_center()[0]) < 1e-10)

        srf_fi = srf_f.interpolate(ratio=2, method='linear')
        # print(srf_fi.get_weighted_mask())
        # print(srf_fi.get_weighted_rf_center())

        # import matplotlib.pyplot as plt
        # f=plt.figure()
        # ax=f.add_subplot(111)
        # srf_fi.plot_rf(plot_axis=ax, tick_spacing=1)
        # plt.show()

        srf_fit = srf_fi.threshold(thr=0.4)
        # print(srf_fit.get_weighted_rf_center())
        assert(srf_fit.get_weighted_rf_center()[0] - 0.5 < 0.05)

        srf2_fi = srf2_f.interpolate(ratio=2, method='linear')
        # print(srf2_fi.get_weighted_mask())
        # print(srf2_fi.get_weighted_rf_center())

        assert(np.array_equal(srf_fi.get_weighted_mask(), srf2_fi.get_weighted_mask()))
        assert(abs(srf_fi.get_weighted_rf_center()[0] + srf2_fi.get_weighted_rf_center()[0]) < 1e-10)


if __name__ == '__main__':
    pass

