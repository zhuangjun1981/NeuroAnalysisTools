import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
                        thr=0.8,
                        obj='pup',
                        is_plot=False,
                        **kwargs):
    """
    get the distribution of the points that pass the confidence threshold for each frame

    :param df_pts: dataframe, result from "read_data_file" function
    :param thr: float, [0., 1.,], threshold for confidence level
    :param obj: str, object of interest, 'led', 'eye' or 'pup' for pupil
    :param kwargs: other inputs to np.histogram() function.
    :return bin_edges: 1d array
    :return hist: 1d array
    """

    col_ns = ['{}{:02d}_lev'.format(obj, i) for i in range(12)]
    # print(col_ns)

    arr = np.array(df_pts[col_ns])
    pts_num = np.sum(arr > thr, axis=1)

    hist, bin_edges = np.histogram(pts_num, **kwargs)

    if is_plot:
        bin_left = bin_edges[:-1]
        f = plt.figure(figsize=(5, 5))
        ax = f.add_subplot(111)
        ax.bar(bin_left, hist)
        ax.xlabel('points per frame that pass threshold')
        ax.ylabel('number of frames')
        plt.show()

    return hist, bin_edges