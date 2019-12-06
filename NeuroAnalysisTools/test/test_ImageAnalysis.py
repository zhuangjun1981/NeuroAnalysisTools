__author__ = 'junz'

import numpy as np
from ..core import ImageAnalysis as ia
import unittest


class TestImageAnalysis(unittest.TestCase):

    def setup(self):
        pass

    def test_getTrace(self):
        mov = np.arange(64).reshape((4,4,4))
        # print mov

        mask1 = np.zeros((4,4)); mask1[2,2]=1; mask1[1,1]=1
        trace1 = ia.get_trace(mov, mask1, maskMode='binary')
        assert(trace1[2] == 39.5)

        mask2 = np.zeros((4,4),dtype=np.float); mask2[:]=np.nan; mask2[2,2]=1; mask2[1,1]=1
        trace2 = ia.get_trace(mov, mask2, maskMode='binaryNan')
        assert(trace2[2] == 39.5)

        mask3 = np.zeros((4,4),dtype=np.float); mask3[2,2]=1; mask3[1,1]=2
        trace3 = ia.get_trace(mov, mask3, maskMode='weighted')
        assert(trace3[2] == 58)

        mask4 = np.zeros((4,4),dtype=np.float); mask4[:]=np.nan; mask4[2,2]=1; mask4[1,1]=2
        trace4 = ia.get_trace(mov, mask4, maskMode='weightedNan')
        assert(trace4[2] == 58)

    def test_ROI_binary_overlap(self):
        roi1 = np.zeros((10, 10))
        roi1[4:8, 3:7] = 1
        roi1 = ia.ROI(roi1)
        roi2 = np.zeros((10, 10))
        roi2[5:9, 5:8] = 1
        roi2 = ia.ROI(roi2)
        assert(roi1.binary_overlap(roi2) == 6)

    def test_ROI(self):
        a = np.zeros((10, 10))
        a[5:7, 3:6] = 1
        a[8:9, 7:10] = np.nan
        roi = ia.ROI(a)
        # plt.imshow(roi.get_binary_mask(),interpolation='nearest')
        assert (list(roi.get_center()) == [5.5, 4.])

    def test_ROI_getBinaryTrace(self):
        mov = np.random.rand(5, 4, 4)
        mask = np.zeros((4, 4))
        mask[2, 3] = 1
        trace1 = mov[:, 2, 3]
        roi = ia.ROI(mask)
        trace2 = roi.get_binary_trace(mov)
        assert (np.array_equal(trace1, trace2))

    def test_WeigthedROI_getWeightedCenter(self):
        aa = np.random.rand(5, 5);
        mask = np.zeros((5, 5))
        mask[2, 3] = aa[2, 3];
        mask[1, 4] = aa[1, 4];
        mask[3, 4] = aa[3, 4]
        roi = ia.WeightedROI(mask)
        center = roi.get_weighted_center()
        assert (center[0] == (2 * aa[2, 3] + 1 * aa[1, 4] + 3 * aa[3, 4]) / (aa[2, 3] + aa[1, 4] + aa[3, 4]))

    def test_plot_ROIs(self):
        aa = np.zeros((50, 50));
        aa[15:20, 30:35] = np.random.rand(5, 5)
        roi1 = ia.ROI(aa)
        _ = roi1.plot_binary_mask_border();
        _ = roi1.plot_binary_mask()
        roi2 = ia.WeightedROI(aa)
        _ = roi2.plot_binary_mask_border();
        _ = roi2.plot_binary_mask();
        _ = roi2.plot_weighted_mask()

    def test_WeightedROI_getWeightedCenterInCoordinate(self):
        aa = np.zeros((5, 5));
        aa[1:3, 2:4] = 0.5
        roi = ia.WeightedROI(aa)
        assert (list(roi.get_weighted_center_in_coordinate(range(2, 7), range(1, 6))) == [3.5, 3.5])

    def test_mergeROIs(self):

        roi1 = ia.WeightedROI(np.arange(9).reshape((3, 3)))
        roi2 = ia.WeightedROI(np.arange(1, 10).reshape((3, 3)))

        merged_ROI = ia.merge_weighted_rois(roi1, roi2)
        merged_ROI2 = ia.merge_binary_rois(roi1, roi2)

        assert (np.array_equal(merged_ROI.get_weighted_mask(), np.arange(1, 18, 2).reshape((3, 3))))
        assert (np.array_equal(merged_ROI2.get_binary_mask(), np.ones((3, 3))))

    def test_get_circularity(self):
        aa = np.zeros((10, 10))
        aa[3:5, 3:5] = 1
        cir1 = ia.get_circularity(aa, is_skimage=False)
        # print(cir1)
        assert(0.7853981633974483 - 1e-15 < cir1 < 0.7853981633974483 + 1e-15)

        print(ia.get_circularity(aa, is_skimage=True))

        aa[3:5, 5] = 1
        cir2 = ia.get_circularity(aa, is_skimage=False)
        # print(cir2)
        assert (0.7539822368615503 - 1e-15 < cir2 < 0.7539822368615503 + 1e-15)

    def test_fit_ellipse(self):

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 30:80] = 255
        mask[40:50, 70:80] = 0
        mask[20:30, 30:40] = 0

        # import matplotlib.pyplot as plt
        # f = plt.figure(figsize=(4, 4))
        # ax = f.add_subplot(111)
        # ax.set_aspect('equal')
        # ax.imshow(mask, interpolation='nearest')
        # plt.show()

        ell = ia.fit_ellipse(mask)
        print(ell.info())

        assert((np.round(ell.angle * 100) / 100) % 180. == 44.16)

        import cv2
        img = np.array([mask, mask, mask]).transpose((1, 2, 0)).copy()
        img = ell.draw(img=img, thickness=1)
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        import matplotlib.pyplot as plt
        plt.imshow(img, interpolation='nearest')
        plt.show()