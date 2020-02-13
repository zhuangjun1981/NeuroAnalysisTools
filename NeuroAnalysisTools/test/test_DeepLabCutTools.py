import os
import unittest

import h5py
import numpy as np
import NeuroAnalysisTools.DeepLabCutTools as dlct

class TestSingleCellAnalysis(unittest.TestCase):

    def setUp(self):
        currFolder = os.path.dirname(os.path.realpath(__file__))
        self.test_data_folder = os.path.join(currFolder, 'data')
        self.test_mov_path = os.path.join(self.test_data_folder, 'test_mov_dlc.avi')
        self.test_h5_path = os.path.join(self.test_data_folder, 'test_mov_dlc.h5')

    def test_read_data_file(self):

        df_pts = dlct.read_data_file(self.test_h5_path, is_verbose=False)
        assert(df_pts.shape == (10, 108))

    def test_get_confidence_dist(self):
        df_pts = dlct.read_data_file(self.test_h5_path, is_verbose=False)
        hist, _ = dlct.get_confidence_dist(df_pts, lev_thr=0.8, obj='pup',
                                           is_plot=False, range=[0, 12],
                                           bins=12)
        assert(np.array_equal(hist, [0,0,0,0,0,0,0,0,0,0,0,10]))

    def test_fit_ellips(self):
        import cv2
        df_pts = dlct.read_data_file(self.test_h5_path, is_verbose=False)
        ells = dlct.fit_ellips(df_pts=df_pts, obj='pup', lev_thr=0.8, num_thr=11, fit_func=cv2.fitEllipse,
                               is_verbose=False)
        # print(ells)
        assert(ells.shape == (10, 5))

    def test_get_all_ellipses(self):
        import cv2
        df_pts = dlct.read_data_file(self.test_h5_path, is_verbose=False)
        df_ell = dlct.get_all_ellips(df_pts=df_pts, lev_thr=0.8, num_thr=11, fit_func=cv2.fitEllipse,
                                     is_verbose=False)
        # print(df_ell)
        assert(df_ell.shape == (10, 15))

    def test_generate_labeled_movie(self):
        import cv2
        save_path = os.path.join(self.test_data_folder, 'test_mov_dlc_ellipse.avi')

        if os.path.isfile(save_path):
            os.remove(save_path)

        is_verbose = False
        df_pts = dlct.read_data_file(self.test_h5_path, is_verbose=is_verbose)
        df_ell = dlct.get_all_ellips(df_pts=df_pts, lev_thr=0.8, num_thr=11, fit_func=cv2.fitEllipse,
                                     is_verbose=is_verbose)
        dlct.generate_labeled_movie(mov_path_raw=self.test_mov_path,
                                    mov_path_lab=save_path,
                                    df_ell=df_ell,
                                    fps=30.,
                                    fourcc='XVID',
                                    is_verbose=is_verbose)

        if os.path.isfile(save_path):
            os.remove(save_path)



