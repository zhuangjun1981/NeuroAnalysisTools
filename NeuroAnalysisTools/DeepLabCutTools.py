import os
import h5py
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from .core import ImageAnalysis as ia
import time


def read_data_file(fpath, is_verbose=True):
    """
    read the resulting .h5 file from Peter Ledochowitsch's code using DeepLabCut to extract
    pupil information from pipeline eyetracking movie.

    :param fpath: str
    :param is_verbose: bool
    :return: dataframe, index is frame id. There are 108 columns representing 36 points extracted
             from the frame. Each point has 3 numbers [column position, row position, confidence level].
             first 12 points are for the edges of the LED corneal reflection.
             second 12 points are for the outline of the eye openning
             third 12 points are for the pupil

             confidence level is a float in the range [0., 1.] with higher value meaning higher confidence
    """

    if is_verbose:
        print('Reading DeepLabCut result file from pipeline eyetracking model.')
        print('\tFile path: {}'.format(fpath))

    f = h5py.File(fpath, 'r')
    keys = list(f.keys())

    if is_verbose:
        print('\tKeys of the data_file: {}. Reading the first one.'.format(keys))
    grp = f[keys[0]]

    grp_keys = list(grp.keys())
    if is_verbose:
        print('\tKeys of the group "/{}": {}. Reading the second one.'.format(keys[0], grp_keys))
    dset = grp[grp_keys[1]][()]
    pts = pts = np.array([p[1] for p in dset])

    col_ns_led = [['led{:02d}_col'.format(i),
                   'led{:02d}_row'.format(i),
                   'led{:02d}_lev'.format(i)] for i in range(12)]

    col_ns_led = np.concatenate(col_ns_led)
    # print(col_ns_led)

    col_ns_eye = [['eye{:02d}_col'.format(i),
                   'eye{:02d}_row'.format(i),
                   'eye{:02d}_lev'.format(i)] for i in range(12)]

    col_ns_eye = np.concatenate(col_ns_eye)
    # print(col_ns_eye)

    col_ns_pup = [['pup{:02d}_col'.format(i),
                   'pup{:02d}_row'.format(i),
                   'pup{:02d}_lev'.format(i)] for i in range(12)]

    col_ns_pup = np.concatenate(col_ns_pup)
    # print(col_ns_pup)

    df_pts = pd.DataFrame(data=pts, columns=np.concatenate([col_ns_led, col_ns_eye, col_ns_pup]))

    if is_verbose:
        print('\tFinal dataframe: shape {}'.format(df_pts.shape))
        print('\n{}'.format(df_pts.head()))

    return df_pts


def get_confidence_dist(df_pts,
                        lev_thr=0.8,
                        obj='pup',
                        is_plot=False,
                        **kwargs):
    """
    get the distribution of the points that pass the confidence threshold for each frame

    :param df_pts: dataframe, result from "read_data_file" function
    :param lev_thr: float, [0., 1.,], threshold for confidence level
    :param obj: str, object of interest, 'led', 'eye' or 'pup' for pupil
    :param kwargs: other inputs to np.histogram() function.
    :return bin_edges: 1d array
    :return hist: 1d array
    """

    col_ns = ['{}{:02d}_lev'.format(obj, i) for i in range(12)]
    # print(col_ns)

    arr = np.array(df_pts[col_ns])
    pts_num = np.sum(arr > lev_thr, axis=1)

    hist, bin_edges = np.histogram(pts_num, **kwargs)

    if is_plot:
        bin_width = np.mean(np.diff(bin_edges))
        bin_middle = bin_edges[:-1] + bin_width / 2
        f = plt.figure(figsize=(5, 5))
        ax = f.add_subplot(111)
        ax.bar(bin_middle, hist, align='center', width=0.8)
        ax.set_xlabel('points per frame that pass threshold')
        ax.set_ylabel('number of frames')
        plt.show()

    return hist, bin_edges


def fit_ellipse(df_pts,
                obj='pup',
                lev_thr=0.8,
                num_thr=11,
                fit_func=cv2.fitEllipse,
                is_verbose=True):
    """
    fit ellipse for a given object in all frames. For a given frame, if the number of points
    with confidence level higher than lev_thr higher than num_thr, a ellipse will be fitted.
    otherwise, np.nan for each ellipse parameter will be returned.

    :param df_pts: dataframe, result from "read_data_file" function
    :param obj: str, object of interest, 'led', 'eye' or 'pup' for pupil
    :param lev_thr: float, [0., 1.,], threshold for confidence level,
                    only the points that have confidence level higher than this value will
                    used for ellipse fitting
    :param num_thr: int, no less than
    :param fit_func: opencv function for ellipse fitting, could be:
                     cv2.fitEllipse, cv2.fitEllipseAMS or cv2.fitEllipseDirect
    :param is_verbose: bool
    :return df_ellipse: dataframe, each line is a frame,
                        columns = [center_row, center_col, long_axis, short_axis, angle]
                        see NeuroAnalysisTools.core.ImageAnalysis.Ellipse class
    """

    if num_thr < 5:
        print('Ellipse fitting require at least 5 points. '
              'The given "num_thr" is less than 5 ({}). Set it to 5.'.format(num_thr))
        num_thr = 5

    if is_verbose:
        print('fitting ellipse for {} ...'.format(obj))

    cns_col = ['{}{:02d}_col'.format(obj, i) for i in range(12)]
    cns_row = ['{}{:02d}_row'.format(obj, i) for i in range(12)]
    cns_lev = ['{}{:02d}_lev'.format(obj, i) for i in range(12)]

    cols = np.round(np.array(df_pts[cns_col])).astype(np.int)
    rows = np.round(np.array(df_pts[cns_row])).astype(np.int)
    levs = np.array(df_pts[cns_lev])

    ellipses = []

    if is_verbose:
        frame_num = len(df_pts)
        frame_num_10th = int(frame_num / 10)

    for frame_i, frame_levs in enumerate(levs):

        if is_verbose:
            if frame_i % frame_num_10th == 0:
                print('\t{:2d}%'.format(int(frame_i * 10 // frame_num_10th)))

        good_pt_msk = frame_levs >= lev_thr

        if np.sum(good_pt_msk) >= num_thr:
            frame_cols = cols[frame_i][good_pt_msk]
            frame_rows = rows[frame_i][good_pt_msk]
            frame_contour = np.array([frame_cols, frame_rows]).T
            frame_ell = ia.Ellipse.from_cv2_box(fit_func(frame_contour))
            ellipses.append([frame_ell.center[0],
                             frame_ell.center[1],
                             frame_ell.axes[0],
                             frame_ell.axes[1],
                             frame_ell.angle])
        else:
            ellipses.append([np.nan] * 5)

    return pd.DataFrame(data=np.array(ellipses, dtype=np.float64),
                        columns=['{}_center_row'.format(obj),
                                 '{}_center_col'.format(obj),
                                 '{}_axis_long'.format(obj),
                                 '{}_axis_short'.format(obj),
                                 '{}_angle_deg'.format(obj)])


def get_all_ellipse(df_pts,
                    lev_thr=0.8,
                    num_thr=11,
                    fit_func=cv2.fitEllipse,
                    is_verbose=True):

    ells_pup = fit_ellipse(df_pts=df_pts, obj='pup', lev_thr=lev_thr, num_thr=num_thr, fit_func=fit_func,
                           is_verbose=is_verbose)
    ells_led = fit_ellipse(df_pts=df_pts, obj='led', lev_thr=lev_thr, num_thr=num_thr, fit_func=fit_func,
                           is_verbose=is_verbose)
    ells_eye = fit_ellipse(df_pts=df_pts, obj='eye', lev_thr=lev_thr, num_thr=num_thr, fit_func=fit_func,
                           is_verbose=is_verbose)

    return pd.concat([ells_led, ells_eye, ells_pup], axis=1)


def label_one_ellipse(frame,
                      ell_param,
                      color=(255, 0, 0),
                      line_width=2):
    """
    label one ellipse to one frame

    :param frame: 3d array, one video frame, from cv2.VideoCapture.read()
    :param ell_param: 1d array, parameters of the ellipse,
                               [center_row, center_col, axis_long, axis_short, anger_degree]
    :param color: list/tuple of 3 unsigned integer numbers, RGB
    :param line_width: float
    :return frame_labelled: 3d array, labeled frame
    """

    if np.isnan(ell_param).any():
        return frame
    else:
        ell = ia.Ellipse(center=[ell_param[0], ell_param[1]],
                         axes=[ell_param[2], ell_param[3]],
                         angle=ell_param[4])
        frame_labelled = ell.draw(img=frame, color=color, thickness=line_width)

        return frame_labelled


def generate_labeled_movie(mov_path_raw,
                           mov_path_lab,
                           df_ell,
                           fourcc='XVID',
                           is_verbose=True,
                           fps=None):

    mov_cv2 = cv2.VideoCapture(mov_path_raw)
    frame_num = int(mov_cv2.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_shape = (int(mov_cv2.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(mov_cv2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if fps is None:
        fps = mov_cv2.get(cv2.CAP_PROP_FPS)

    if frame_num != len(df_ell):
        raise ValueError('The number of frames in {} ({}) does not match the length of the '
                         'ellipse dataframe {}.'.format(os.path.realpath(mov_path_raw),
                                                        frame_num, len(df_ell)))

    if is_verbose:
        print('\nStart to generating labeled movie for {} ...'.format(os.path.realpath(mov_path_raw)))
        print('\tnumber of frames: {}'.format(frame_num))
        print('\tshape of frames (width, height): {}'.format(frame_shape))

    ells_led = np.array(df_ell[['led_center_row', 'led_center_col', 'led_axis_long', 'led_axis_short',
                                'led_angle_deg']])
    ells_eye = np.array(df_ell[['eye_center_row', 'eye_center_col', 'eye_axis_long', 'eye_axis_short',
                                'eye_angle_deg']])
    ells_pup = np.array(df_ell[['pup_center_row', 'pup_center_col', 'pup_axis_long', 'pup_axis_short',
                                'pup_angle_deg']])

    output = cv2.VideoWriter(filename=mov_path_lab,
                             fourcc=cv2.VideoWriter_fourcc(*fourcc),
                             fps=fps,
                             frameSize=frame_shape)

    if is_verbose:
        t0 = time.time()
        frame_num_10th = int(frame_num / 10)

    for i in range(frame_num):

        if is_verbose:
            if i % frame_num_10th == 0:
                print('\t{:6d} seconds: {:2d}%'.format(int(time.time() - t0),
                                                       int(i * 10 // frame_num_10th)))

        _, curr_frame = mov_cv2.read()
        frame_labeled = label_one_ellipse(curr_frame, ell_param=ells_led[i], color=(255, 0, 0), line_width=2)
        frame_labeled = label_one_ellipse(frame_labeled, ell_param=ells_eye[i], color=(0, 255, 0), line_width=2)
        frame_labeled = label_one_ellipse(frame_labeled, ell_param=ells_pup[i], color=(0, 0, 255), line_width=2)

        # plt.imshow(frame_labeled)
        # plt.show()

        output.write(frame_labeled)

    output.release()



