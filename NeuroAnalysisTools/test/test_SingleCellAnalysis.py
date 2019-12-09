__author__ = 'junz'

import os
import unittest
import h5py
import numpy as np

from ..core import ImageAnalysis as ia
from ..core import FileTools as ft
from .. import SingleCellAnalysis as sca

# import NeuroAnalysisTools.core.ImageAnalysis as ia
# import NeuroAnalysisTools.core.FileTools as ft
# import NeuroAnalysisTools.SingleCellAnalysis as sca


class TestSingleCellAnalysis(unittest.TestCase):

    def setup(self):

        currFolder = os.path.dirname(os.path.realpath(__file__))
        self.testDataFolder = os.path.join(currFolder,'data')

        self.sparseNoiseDisplayLogPath = os.path.join(self.testDataFolder,'SparseNoiseDisplayLog.pkl')
        self.testH5Path = os.path.join(self.testDataFolder,'test.hdf5')
        self.STRFDataPath = os.path.join(self.testDataFolder,'cellsSTRF.hdf5')

    def test_mergeROIs(self):
        roi1 = ia.WeightedROI(np.arange(9).reshape((3, 3)))
        roi2 = ia.WeightedROI(np.arange(1, 10).reshape((3, 3)))

        merged_ROI = sca.merge_weighted_rois(roi1, roi2)
        merged_ROI2 = sca.merge_binary_rois(roi1, roi2)

        assert(np.array_equal(merged_ROI.get_weighted_mask(), np.arange(1, 18, 2).reshape((3, 3))))
        assert(np.array_equal(merged_ROI2.get_binary_mask(), np.ones((3, 3))))

    def test_getSparseNoiseOnsetIndex(self):
        allOnsetInd, onsetIndWithLocationSign = sca.get_sparse_noise_onset_index(ft.loadFile(self.sparseNoiseDisplayLogPath))
        # print list(allOnsetInd[0:10])
        # print onsetIndWithLocationSign[2][0]
        assert(list(allOnsetInd[0:10])==[0, 6, 12, 18, 24, 30, 36, 42, 48, 54])
        assert(np.array_equal(onsetIndWithLocationSign[2][0],np.array([0.,  70.])))

    def test_SpatialTemporalReceptiveField_from_h5_group(self):
        f = h5py.File(self.STRFDataPath)
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
        trace = np.array(STRF.data['traces'][20])
        assert((float(trace[4, 8])+0.934942364693) < 1e-10)
        # STRF.plot_traces(figSize=(15,10),yRange=[-5,50],columnSpacing=0.002,rowSpacing=0.002)

    def test_SpatialTemporalReceptiveField(self):
        locations = [[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0],[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0]]
        signs = [1,1,1,1,-1,-1,-1,-1]
        traces=[[np.arange(4)],[np.arange(1,5)],[np.arange(2,6)],[np.arange(3,7)],[np.arange(5,9)],[np.arange(6,10)],
                [np.arange(7,11)],[np.arange(8,12)]]
        traces=[np.array(t) for t in traces]
        time = np.arange(4,8)
        STRF = sca.SpatialTemporalReceptiveField(locations,signs,traces,time)
        assert(STRF.data['traces'][0][0][1]==8)
        assert(STRF.data['sign'][4]==1)
        assert(np.array_equal(STRF.get_locations()[2], np.array([3., 4., -1.])))
        newLocations = [[location[0]+1,location[1]+1] for location in locations[0:4]]
        newSigns = [1,1,1,1]
        STRF.add_traces(newLocations, newSigns, traces[0:4])
        assert(STRF.data['traces'][7][1][2]==4)
        # _ = STRF.plot_traces()

    def test_SpatialTemporalReceptiveField_IO(self):
        locations = [[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0],[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0]]
        signs = [1,1,1,1,-1,-1,-1,-1]
        traces=[[np.arange(4)],[np.arange(1,5)],[np.arange(2,6)],[np.arange(3,7)],[np.arange(5,9)],[np.arange(6,10)],[np.arange(7,11)],[np.arange(8,12)]]
        time = np.arange(4,8)

        STRF = sca.SpatialTemporalReceptiveField(locations,signs,traces,time)
        if os.path.isfile(self.testH5Path):os.remove(self.testH5Path)
        testFile = h5py.File(self.testH5Path)
        STRFGroup = testFile.create_group('spatial_temporal_receptive_field')
        STRF.to_h5_group(STRFGroup)
        testFile.close()

        h5File = h5py.File(self.testH5Path)
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(h5File['spatial_temporal_receptive_field'])
        h5File.close()
        assert(STRF.data['traces'][3][0][1]==7)

    def test_SpatialTemporalReceptiveField_getAmpLitudeMap(self):
        f = h5py.File(self.STRFDataPath)
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
        ampON, ampOFF, altPos, aziPos = STRF.get_amplitude_map()
        assert(ampON[7,10]-(-0.0258248019964) < 1e-10)
        assert(ampOFF[8,9]-(-0.501572728157) < 1e-10)
        assert(altPos[5]==30.)
        assert(aziPos[3]==-5.)
        # sca.plot_2d_receptive_field(ampON,altPos,aziPos,cmap='gray_r',interpolation='nearest')

    def test_SpatialTemporalReceptiveField_getZscoreMap(self):
        f = h5py.File(self.STRFDataPath)
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
        zscoreON, zscoreOFF, altPos, aziPos = STRF.get_zscore_map()
        assert(zscoreON[7,10]-(-0.070735671412) < 1e-10)
        assert(zscoreOFF[8,9]-(-0.324245551387) < 1e-10)
        # sca.plot_2d_receptive_field(ampON,altPos,aziPos,cmap='gray_r',interpolation='nearest')

    def test_SpatialTemporalReceptiveField_getCenters(self):
        f = h5py.File(self.STRFDataPath)
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
        assert(STRF.get_zscore_roi_centers()[1][1] - (-2.1776047950146622) < 1e-10)

    def test_SpatialTemporalReceptiveField_getAmplitudeReceptiveField(self):
        f = h5py.File(self.STRFDataPath)
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
        ampRFON, ampRFOFF = STRF.get_amplitude_receptive_field()
        assert(ampRFON.sign==1);assert(ampRFOFF.sign==-1)
        assert(ampRFOFF.get_weighted_mask()[7, 9] - 3.2014527 < 1e-7)

    def test_SpatialTemporalReceptiveField_getZscoreReceptiveField(self):
        f = h5py.File(self.STRFDataPath)
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
        zscoreRFON, zscoreRFOFF = STRF.get_zscore_receptive_field()
        assert(zscoreRFON.sign==1);assert(zscoreRFOFF.sign==-1)
        assert(zscoreRFOFF.get_weighted_mask()[7, 9] - 1.3324414 < 1e-7)

    def test_SpatialTemporalReceptiveField_shrink(self):
        f = h5py.File(self.STRFDataPath)
        STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
        STRF.shrink([-10,10],None)
        assert(np.array_equal(np.unique(np.array(STRF.get_locations())[:, 0]), np.array([-10., -5., 0., 5., 10.])))
        STRF.shrink(None,[0,20])
        assert(np.array_equal(np.unique(np.array(STRF.get_locations())[:, 1]), np.array([0., 5., 10., 15., 20.])))

    def test_SpatialReceptiveField(self):
        SRF = sca.SpatialReceptiveField(np.arange(9).reshape((3,3)),np.arange(3),np.arange(3))
        assert(np.array_equal(SRF.weights,np.arange(1,9)))

    def test_SpatialReceptiveField_thresholdReceptiveField(self):
        SRF = sca.SpatialReceptiveField(np.arange(9).reshape((3,3)),np.arange(3),np.arange(3))
        thresholdedSRF=SRF.threshold_receptive_field(4)
        assert(np.array_equal(thresholdedSRF.weights,np.arange(4,9)))

    def test_SpatialReceptiveField_interpolate(self):
        SRF = sca.SpatialReceptiveField(np.random.rand(5,5),np.arange(5)[::-1],np.arange(5))
        SRF.interpolate(5)
        assert(SRF.get_weighted_mask().shape == (20, 20))

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

        assert (OSI_raw == OSI_ele == OSI_rec == 1. / 3.)
        assert (DSI_raw == DSI_ele == DSI_rec == 1. / 3.)

        assert (gOSI_raw == gOSI_ele == gOSI_rec == 1. / 9.)
        assert (gDSI_raw == gDSI_ele == gDSI_rec == 1. / 9.)

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

        assert (OSI_raw == OSI_rec == 1. / 3.)
        assert (DSI_raw == 3.0)
        assert (DSI_ele == DSI_rec == 1.0)
        assert (OSI_ele == 0.2)
        assert (gOSI_raw < (1. / 7. + 1E-7))
        assert (gOSI_raw > (1. / 7. - 1E-7))
        assert (gDSI_raw == 3. / 7.)
        assert (gOSI_ele < (1. / 15. + 1E-7))
        assert (gOSI_ele > (1. / 15. - 1E-7))
        assert (gDSI_ele == 3. / 15.)
        assert (gOSI_rec == 0.)
        assert (gDSI_rec == 2. / 8.)

        assert (peak_dire_raw == int(vs_dire_raw) == int(vs_dire_ele) == int(vs_dire_rec) == 90)


if __name__ == '__main__':
    tests = TestSingleCellAnalysis()

