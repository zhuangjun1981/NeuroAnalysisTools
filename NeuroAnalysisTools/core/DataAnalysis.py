import numpy as np
import scipy.ndimage as ni
import matplotlib.pyplot as plt


def interpolate_nans(arr):
    """
    fill the nans in a 1d array by interpolating value on both sides
    """

    if len(arr.shape) != 1:
        raise ValueError('input arr should be 1d array.')

    nan_ind = np.isnan(arr)

    nan_pos = nan_ind.nonzero()[0]
    # print(nan_pos)
    data_pos = (~nan_ind).nonzero()[0]
    # print(data_pos)
    data = arr[~nan_ind]
    # print(data)

    arr1 = np.array(arr)
    arr1[nan_ind] = np.interp(nan_pos, data_pos, data)

    return arr1


def downsample(arr, rate, method=np.mean):
    """
    down sample a 1d array by the method

    :param arr: 1d array.
    :param rate: int, larger than 1, downsample rate
    :param method: function that can be applied to one axis of a 2d array
    :return: 1d array downsampled data
    """

    if len(arr.shape) != 1:
        raise ValueError('input arr should be 1d array.')

    rate_int = int(np.round(rate))

    if rate_int < 2:
        raise ValueError('input rate should be a integer larger than 1.')

    if arr.shape[0] < rate_int:
        return np.array([])
    else:
        n_d = arr.shape[0] // rate
        arr_reshape = arr[0: n_d * rate].reshape((n_d, rate))
        arr_d = method(arr_reshape, axis=1)

    return arr_d


def get_pupil_area(pupil_shapes, fs, ell_thr=0.5, median_win=3.):
    """
    from elliptic pupil shapes, calculate pupil areas and filter out outliers.

    step 1: calculate area
    step 2: nan the shape with ellipticity larger than ell_thr
            ellipticity = (a - b) / b
    step 3: interpolate the nans
    step 4: median filter with length of median_win

    :param pupil_shapes: 2d array, each row: each sampling point; column0: axis0; column1: axis1; column2: angle
    :param fs: float, Hz, sampling rate
    :param ell_thr: float, (0. 1.], threshold for ellipticity
    :param median_win: float, sec, window length of median filter

    :return: 1d array of pupil area.
    """

    if len(pupil_shapes.shape) != 2:
        raise ValueError('input pupil_shapes should be 2d array.')

    if pupil_shapes.shape[1] < 2:
        raise ValueError('input pupil_shapes should have at least 2 columns.')

    area = np.pi * pupil_shapes[:, 0] * pupil_shapes[:, 1]
    ax1 = np.nanmax(pupil_shapes[:, 0:2], axis=1)
    ax2 = np.nanmin(pupil_shapes[:, 0:2], axis=1)
    ell = (ax1 - ax2) / ax1
    area[ell > ell_thr] = np.nan
    area = interpolate_nans(area)
    area = ni.median_filter(area, int(fs * median_win))

    return area


def get_running_speed(sig, ts, ref=None, disk_radius=8., fs_final=30., speed_thr_pos=100., speed_thr_neg=-20.,
                      gauss_sig=0.02):
    """
    get downsampled and filtered running speed from raw data.

    the sig/ref defines the running disk angle position.

    :param sig: 1d array, voltage, signal from encoder
    :param ts: 1d array, timestamps
    :param ref: 1d array or None, reference from encoder, if None, assuming 5 vol.
    :param disk_radius: float, mouse running disk radius in cm
    :param fs_final: float, the final sampling rate after downsampling
    :param speed_thr_pos: float, cm/sec, positive speed threshold
    :param speed_thr_neg: float, cm/sec, negative speed threshold
    :param gauss_sig: float, sec, gaussian filter sigma
    :return speed: 1d array, downsampled and filtered speed, cm/sec
    :return speed_ts: 1d array, timestamps of speed
    """

    if ref is not None:
        running = 2 * np.pi * (sig / ref) * disk_radius
    else:
        running = 2 * np.pi * (sig / 5.) * disk_radius

    fs_raw = 1. / np.mean(np.diff(ts))

    rate_d = int(fs_raw / fs_final)
    running_d = downsample(arr=running, rate=rate_d, method=np.mean)
    ts_d = downsample(arr=ts, rate=rate_d, method=np.mean)

    speed = np.diff(running_d)

    speed_ts = ts_d[0:-1]
    speed = speed / np.mean(np.diff(speed_ts))

    speed[speed > speed_thr_pos] = np.nan
    speed[speed < speed_thr_neg] = np.nan

    speed = interpolate_nans(speed)

    sigma_pt = int(gauss_sig / np.mean(np.diff(speed_ts)))
    speed = ni.gaussian_filter1d(input=speed, sigma=sigma_pt)

    return speed, speed_ts


def get_clustering_distances(mat_dis, cluster):
    """
    giving distance matrix and cluster masks, return a list of arrays containing distances among each cluster
    and a array containing distances between not leaves not in same cluster.
    :param mat_dis: 2d array, m x m, distance matrix
    :param cluster: 1d array, cluster masks, usually generated by the scipy.cluster.hierarchy.fcluster()
                    function
    :return dis_clu: list of 1d arrays, each array contains the distances between all pairs of leaves
                     belonging to each cluster.
    :return dis_non_clu: 1d arrays, distances between all pairs of leaves not belonging to same cluster.
    """

    if len(mat_dis.shape) != 2:
        raise ValueError('distance matrix should be a 2d array.')

    if mat_dis.shape[0] != mat_dis.shape[1]:
        raise ValueError('distance matrix should ba square array.')

    if np.min(cluster) == 1:
        clu = np.array(cluster) - 1 # change to 0-based indexing
    elif np.min(cluster) == 0:
        clu = np.array(cluster)
    else:
        raise ValueError("the index of first cluster should be either 0 or 1.")

    # getting cluster dictionary
    cluster_num = np.max(clu) + 1
    clu_dic = {}
    for clu_i in range(cluster_num):
        clu_dic.update({clu_i: np.where(clu==clu_i)[0]})

    # reorganize mat_dis
    mat_dis_tmp = []
    for clu_i in range(cluster_num):
        mat_dis_tmp.append(mat_dis[clu_dic[clu_i], :])
    mat_dis_tmp = np.concatenate(mat_dis_tmp, axis=0).transpose()

    mat_dis_reorg = []
    for clu_i in range(cluster_num):
        mat_dis_reorg.append(mat_dis_tmp[clu_dic[clu_i], :])
    mat_dis_reorg = np.concatenate(mat_dis_reorg, axis=0).transpose()

    # plt.imshow(mat_dis_reorg, interpolation='nearest', cmap='plasma', vmin=0, vmax=1)
    # plt.show()

    # get cluster masks on the reorganized distance matrix
    mask_non_clu = np.ones(mat_dis.shape, dtype=np.bool)
    dis_clu = []
    clu_start_i = 0
    for clu_i in range(cluster_num):
        leaf_num = len(clu_dic[clu_i])
        clu_end_i = clu_start_i + leaf_num
        mat_clu = mat_dis_reorg[clu_start_i: clu_end_i, clu_start_i: clu_end_i]
        mask_non_clu[clu_start_i: clu_end_i, clu_start_i:clu_end_i] = False

        dis_clu.append(mat_clu[np.triu_indices(n=leaf_num, k=1)])
        clu_start_i = clu_end_i

    mask_triu = np.zeros(mat_dis.shape, dtype=np.bool)
    mask_triu[np.triu_indices(n=mat_dis.shape[0], k=1)] = True

    dis_non_clu = mat_dis_reorg[mask_non_clu & mask_triu]

    # print(clu_dic)
    # print(mat_dis_reorg)
    # print(mask_non_clu)
    # print(dis_clu)
    # print(dis_non_clu)

    return dis_clu, dis_non_clu


def calculate_image_selectivity(resp_arr, levels=1000):
    """
    calculate response selectivity to a set of image stimuli
    adapted from Saskia de Vries' code.

    :param resp_arr: 1d array, mean responses to each image
    :param levels: uint, number of levels to set threshold
    :return: float, [-1, 1], small: low selectivity. large: high selectivity
    """

    if len(resp_arr.shape) != 1:
        raise ValueError('Input "resp_arr" should be 1d array.')

    img_num = float(len(resp_arr))

    fmin = resp_arr.min()
    fmax = resp_arr.max()
    rtj = np.empty(levels)
    for l in range(levels):
        thr = fmin + l * ((fmax - fmin) / levels)
        rtj[l] = np.sum(resp_arr > thr) / img_num
    sel = 1 - (2 * rtj.mean())
    return sel


if __name__ == '__main__':

    # ============================================================================================================
    y = np.array([1, 1, 1, np.nan, np.nan, 2, 2, np.nan, 0])
    y1 = interpolate_nans(y)
    print(y1)
    # ============================================================================================================