"""
these are the functions that deal with the .nwb database of GCaMP labelled LGN boutons.
"""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from numbers import Number
import pandas as pd
import scipy.stats as stats
import scipy.ndimage as ni
import scipy.interpolate as ip
import scipy.spatial as spatial
import scipy.cluster as cluster
from matplotlib.backends.backend_pdf import PdfPages

from . import SingleCellAnalysis as sca
from .core import ImageAnalysis as ia
from .core import PlottingTools as pt
from .core import DataAnalysis as da
from .core import TimingAnalysis as ta


ANALYSIS_PARAMS = {
    'trace_type': 'f_center_subtracted',
    'trace_abs_minimum': 1., # float, trace absolute minimum, if the roi trace minimum is lower than this value
                             # it will be added to with a bias to ensure the absolute minimum is no less than
                             # this value for robustness of df/f calculation
    'filter_length_skew_sec': 5., # float, second, the length to filter input trace to get slow trend
    'response_window_positive_rf': [0., 0.5], # list of 2 floats, temporal window to get upwards calcium response for receptive field
    'response_window_negative_rf': [0., 1.], # list of 2 floats, temporal window to get downward calcium response for receptive field
    'gaussian_filter_sigma_rf': 1., # float, in pixels, filtering sigma for z-score receptive fields
    'interpolate_rate_rf': 10., # float, interpolate rate of filtered z-score maps
    # 'peak_z_threshold_rf': 1.5, # float, threshold for significant receptive field of z score after filtering.
    'rf_z_thr_abs': 1.6, # float, absolute threshold for significant zscore receptive field
    'rf_z_thr_rel': 0.4, # float, relative threshold for significant zscore receptive field
    'response_window_dgc': [0., 1.], # list of two floats, temporal window for getting response value for drifting grating
    'baseline_window_dgc': [-0.5, 0.], # list of two floats, temporal window for getting baseline value for drifting grating
    'is_collapse_sf': False, # bool, average across sf or not for direction/tf tuning curve
    'is_collapse_tf': False, # bool, average across tf or not for direction/sf tuning curve
    'is_collapse_dire': False, # bool, average across direction or not for tf/sf tuning curve
    'dgc_elevation_bias': 0., # float, the bias to lift the dgc tuning curves if postprocess is 'elevate'
                   }

PLOTTING_PARAMS = {
    'response_type_for_plot': 'zscore', # str, 'df', 'dff', or 'zscore'
    'fig_size': (8.5, 11),
    'fig_facecolor': "#ffffff",
    'ax_roi_img_coord': [0.01, 0.75, 0.3, 0.24], # coordinates of roi image
    'rf_img_vmin': 0., # reference image min
    'rf_img_vmax': 0.5, # reference image max
    'roi_border_color': '#ff0000',
    'roi_border_width': 2,
    'field_traces_coord': [0.32, 0.75, 0.67, 0.24], # field coordinates of trace plot
    'traces_panels': 4, # number of panels to plot traces
    'traces_color': '#888888',
    'traces_line_width': 0.5,
    'ax_rf_pos_coord': [0.01, 0.535, 0.3, 0.24],
    'ax_rf_neg_coord': [0.32, 0.535, 0.3, 0.24],
    'rf_zscore_vmax': 4.,
    'ax_peak_traces_pos_coord': [0.01, 0.39, 0.3, 0.17],
    'ax_peak_traces_neg_coord': [0.32, 0.39, 0.3, 0.17],
    'blank_traces_color': '#888888',
    'peak_traces_pos_color': '#ff0000',
    'peak_traces_neg_color': '#0000ff',
    'response_window_color': '#ff00ff',
    'baseline_window_color': '#888888',
    'block_face_color': '#cccccc',
    'single_traces_lw': 0.5,
    'mean_traces_lw': 2.,
    'dgc_postprocess': 'elevate',
    'ax_text_coord': [0.63, 0.005, 0.36, 0.74],
    'ax_sftf_pos_coord': [0.01, 0.21, 0.3, 0.17],
    'ax_sftf_neg_coord': [0.32, 0.21, 0.3, 0.17],
    'sftf_cmap': 'RdBu_r',
    'sftf_vmax': 4,
    'sftf_vmin': -4,
    'ax_dire_pos_coord': [0.01, 0.01, 0.28, 0.18],
    'ax_dire_neg_coord': [0.32, 0.01, 0.28, 0.18],
    'dire_color_pos': '#ff0000',
    'dire_color_neg': '#0000ff',
    'dire_line_width': 2,
}


def get_nt_index(dire_lst, weights=None, is_arc=False, half_span=45., sum_thr=10):
    """
    calculate nasal/temporal index: (nasal - temporal) / (nasal + temporal) from a
    list of preferred directions.

    :param dire_lst: 1d array, a list of preferred directions
    :param weights: 1d array, same shape as dire_lst, weights for the preferred directions
    :param is_arc: bool, True: directions are in arc unit; False: directions are in degree unit
    :param half_span: float, positive, in degrees, the half-span to define nasal or temporal direction.
                      nasal directions will be definded as [- half_span, half_span], temporal directions
                      will be defined as [180 - half_span, 180 + half_span].
    :param sum_thr: float, positive, the threshold for the total number of nasal and temporal directions.
                    if num(nasal) + num(temporal) < sum_thr, return np.nan
    :return nt_index: float, [-1., 1.]
    """

    if is_arc:
        dire_lst = np.array(dire_lst) * 180. / np.pi
    else:
        dire_lst = np.array(dire_lst)

    if weights is None:
        weights = np.ones(dire_lst.shape)

    weights = np.array(weights, dtype=np.float64)

    if dire_lst.shape != weights.shape:
        raise ValueError("the shape of dire_lst ({}) should be the same "
                         "as the shape of weights ({})".format(dire_lst.shape, weights.shape))

    dire_lst = dire_lst % 360.

    temp = np.sum(weights[dire_lst >= (360. - half_span)]) + \
           np.sum(weights[dire_lst <= half_span])
    nasa = np.sum(weights[np.logical_and(dire_lst >= (180. - half_span),
                                         dire_lst <= (180. + half_span))])

    if nasa + temp < float(sum_thr):
        return np.nan, nasa, temp
    else:
        return (nasa - temp) / (nasa + temp), nasa, temp


def get_normalized_binary_roi(roi, scope, canvas_size=300., pixel_res=600, is_center=True):
    """
    given an roi, return the

    :param roi: corticalmapping.core.ImagingAnalysis.ROI object, this object should have correct "pixelSize" and
                "pixelSizeUnit" attribute
    :param scope: str, 'deepscope' or 'sutter' in future maybe "scientifica"
    :param canvas_size: float, micron
    :param pixel_res: uint, pixel resolution of the final roi
    :param is_center: bool, if true, center the roi to its center of mass (binary).
    :return: corticalmapping.core.ImagingAnalysis.ROI object, with orientation in the standard way.
             (up: anterior; bottom: posterior; left: lateral; right: medial) and with pixelSizeUnit as microns
    """

    mask = roi.get_binary_mask()

    if roi.pixelSizeX != roi.pixelSizeY:
        raise NotImplementedError('rois with non-square pixels are not supported.')

    if roi.pixelSizeUnit == 'meter':
        pixel_size_s = float(roi.pixelSizeX) * 1e6
    elif roi.pixelSizeUnit == 'micron':
        pixel_size_s = float(roi.pixelSizeX)
    else:
        raise LookupError("the pixelSizeUnit attribute of input roi should be either 'meter' or 'micron'. "
                          "'{}' given.".format(roi.pixelSizeUnit))

    pixel_size_t = float(canvas_size) / pixel_res
    zoom = pixel_size_s / pixel_size_t

    mask_nor = ia.rigid_transform_cv2_2d(mask, zoom=zoom)

    if scope == 'deepscope':
        mask_nor = ia.rigid_transform_cv2_2d(mask_nor,
                                             rotation=140,
                                             outputShape=(pixel_res, pixel_res))[::-1]
    elif scope == 'sutter':
        mask_nor = mask_nor.transpose()[::-1, :]
        mask_nor = ia.rigid_transform_cv2_2d(mask_nor, outputShape=(pixel_res, pixel_res))
    else:
        raise LookupError('Do not understand scope type ({}). Should be "deepscope" or "sutter".')

    mask_nor[mask_nor <= 0] = 0
    mask_nor[mask_nor > 0] = 1
    mask_nor = mask_nor.astype(np.uint8)
    roi_nor = ia.ROI(mask_nor, pixelSize=pixel_size_t, pixelSizeUnit='micron')
    if not is_center:
        return roi_nor
    else:
        center = roi_nor.get_center()
        offset_y = pixel_res / 2. - center[0]
        offset_x = pixel_res / 2. - center[1]

        mask_nor = ia.rigid_transform_cv2_2d(mask_nor, offset=[offset_x, offset_y])

        mask_nor[mask_nor <= 0] = 0
        mask_nor[mask_nor > 0] = 1
        mask_nor = mask_nor.astype(np.uint8)
        roi_nor = ia.ROI(mask_nor, pixelSize=pixel_size_t, pixelSizeUnit='micron')
    return roi_nor


def get_scope(nwb_f):

    try:
        device = nwb_f['general/optophysiology/imaging_plane_1/device'][()]
    except KeyError:
        device = nwb_f['general/optophysiology/imaging_plane_0/device'][()]

    device = device.decode('utf-8')

    if device in ['DeepScope', 'Deep Scope', 'deepscope', 'Deepscope', 'deep scope', 'Deep scope']:
        return 'deepscope'
    elif device in ['Sutter', 'sutter'] or 'Sutter' in device:
        return 'sutter'
    else:
        raise LookupError('Do not understand device ({}). should be "deepscope" or "sutter"'.format(device))


def get_depth(nwb_f, plane_n):
    return nwb_f['processing/rois_and_traces_{}/imaging_depth_micron'.format(plane_n)][()]


def get_background_img(nwb_f, plane_n):

    rf_grp = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane/reference_images'.format(plane_n)]

    if 'max_projection' in rf_grp.keys():
        return rf_grp['max_projection/data'][()]
    elif 'mean_projection' in rf_grp.keys():
        return rf_grp['mean_projection/data'][()]
    else:
        return None


def get_roi_triplets(nwb_f, overlap_ratio=0.9, size_thr=25.):
    """
    for deepscope imaging session with 3 planes, get overlapping roi triplets
    each triplets contain one roi for each plane and they are highly overlapping
    this is to find cells appear in multiple plane, and the results can be passed
    to HighLevel.plot_roi_traces_three_planes to plot and decided if they represent
    same cell.
    :param nwb_f:
    :param overlap_ratio:
    :param size_thr: only rois bigger than this size will be considered, um^2
    :return: list of triplets (tuple of three strings)
    """

    roi_grp0 = nwb_f['processing/rois_and_traces_plane0/ImageSegmentation/imaging_plane']
    roi_lst0 = roi_grp0['roi_list'][()]
    roi_lst0 = [r for r in roi_lst0 if r[0:4] == 'roi_']
    roi_lst0_new = []
    for roi_n0 in roi_lst0:
        curr_roi = get_roi(nwb_f=nwb_f, plane_n='plane0', roi_n=roi_n0)
        if curr_roi.get_pixel_area() * 1e12 >= size_thr:
            roi_lst0_new.append(roi_n0)
    roi_lst0 = roi_lst0_new

    roi_grp1 = nwb_f['processing/rois_and_traces_plane1/ImageSegmentation/imaging_plane']
    roi_lst1 = roi_grp1['roi_list'][()]
    roi_lst1 = [r for r in roi_lst1 if r[0:4] == 'roi_']
    roi_lst1_new = []
    for roi_n1 in roi_lst1:
        curr_roi = get_roi(nwb_f=nwb_f, plane_n='plane1', roi_n=roi_n1)
        if curr_roi.get_pixel_area() * 1e12 >= size_thr:
            roi_lst1_new.append(roi_n1)
    roi_lst1 = roi_lst1_new

    roi_grp2 = nwb_f['processing/rois_and_traces_plane2/ImageSegmentation/imaging_plane']
    roi_lst2 = roi_grp2['roi_list'][()]
    roi_lst2 = [r for r in roi_lst2 if r[0:4] == 'roi_']
    roi_lst2_new = []
    for roi_n2 in roi_lst2:
        curr_roi = get_roi(nwb_f=nwb_f, plane_n='plane2', roi_n=roi_n2)
        if curr_roi.get_pixel_area() * 1e12 >= size_thr:
            roi_lst2_new.append(roi_n2)
    roi_lst2 = roi_lst2_new

    triplets = []

    while roi_lst1: # start from middle plane

        curr_roi1_n = roi_lst1.pop(0)
        curr_triplet = [None, curr_roi1_n, None]

        curr_roi1 = get_roi(nwb_f=nwb_f, plane_n='plane1', roi_n=curr_roi1_n)
        curr_roi1_area = curr_roi1.get_binary_area()

        for curr_roi0_ind, curr_roi0_n in enumerate(roi_lst0):
            # look through rois in plane0, pick the one overlaps with curr_roi1
            curr_roi0 = get_roi(nwb_f=nwb_f, plane_n='plane0', roi_n=curr_roi0_n)
            curr_roi0_area = curr_roi0.get_binary_area()
            curr_overlap = curr_roi1.binary_overlap(curr_roi0)
            if float(curr_overlap) / min([curr_roi1_area, curr_roi0_area]) >= overlap_ratio:
                curr_triplet[0] = roi_lst0.pop(curr_roi0_ind)
                break

        for curr_roi2_ind, curr_roi2_n in enumerate(roi_lst2):
            # look through rois in plane0, pick the one overlaps with curr_roi1
            curr_roi2 = get_roi(nwb_f=nwb_f, plane_n='plane2', roi_n=curr_roi2_n)
            curr_roi2_area = curr_roi2.get_binary_area()
            curr_overlap = curr_roi1.binary_overlap(curr_roi2)
            if float(curr_overlap) / min([curr_roi1_area, curr_roi2_area]) >= overlap_ratio:
                curr_triplet[2] = roi_lst2.pop(curr_roi2_ind)
                break

        print(curr_triplet)
        triplets.append(tuple(curr_triplet))

    while roi_lst2: # next, more superficial plane

        curr_roi2_n = roi_lst2.pop(0)
        curr_triplet = [None, None, curr_roi2_n]

        curr_roi2 = get_roi(nwb_f=nwb_f, plane_n='plane2', roi_n=curr_roi2_n)
        curr_roi2_area = curr_roi2.get_binary_area()

        for curr_roi0_ind, curr_roi0_n in enumerate(roi_lst0):
            # look through rois in plane0, pick the one overlaps with curr_roi2
            curr_roi0 = get_roi(nwb_f=nwb_f, plane_n='plane0', roi_n=curr_roi0_n)
            curr_roi0_area = curr_roi0.get_binary_area()
            curr_overlap = curr_roi2.binary_overlap(curr_roi0)
            if float(curr_overlap) / min([curr_roi2_area, curr_roi0_area]) >= overlap_ratio:
                curr_triplet[0] = roi_lst0.pop(curr_roi0_ind)
                break

        triplets.append(tuple(curr_triplet))

    triplets = triplets + [(rn, None, None) for rn in roi_lst0] # finally add the rest rois in deep plane

    return triplets


def get_plane_ns(nwb_f):
    keys = [k[-6:] for k in nwb_f['processing'].keys() if 'rois_and_traces_' in k]
    return keys


def get_roi_ns(nwb_f, plane_n):
    roi_lst = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane/roi_list'.format(plane_n)][()]
    roi_lst = [r.decode('utf-8') for r in roi_lst]
    roi_ns = [r for r in roi_lst if r[0:4] == 'roi_']
    return roi_ns


def get_sampling_rate(nwb_f, ts_name):
    grp = nwb_f['acquisition/timeseries/{}'.format(ts_name)]

    if 'starting_time' in grp.keys():
        return grp['starting_time'].attrs['rate']
    else:
        ts = grp['timestamps'][()]
        return 1. / np.mean(np.diff(ts))


def get_strf_grp_key(nwb_f):
    analysis_grp = nwb_f['analysis']
    strf_key = [k for k in analysis_grp.keys() if k[0:4] == 'strf' and 'SparseNoise' in k]
    if len(strf_key) == 0:
        return None
    elif len(strf_key) == 1:
        return strf_key[0]
    else:
        raise LookupError('more than one drifting grating response table found.')


def get_strf(nwb_f, plane_n, roi_ind, trace_type):
    strf_key = get_strf_grp_key(nwb_f=nwb_f)

    if strf_key is not None:
        strf_grp = nwb_f['analysis/{}/{}'.format(strf_key, plane_n)]
        strf = sca.get_strf_from_nwb(h5_grp=strf_grp, roi_ind=roi_ind, trace_type=trace_type)
        return strf
        # try:
        #     strf_grp = nwb_f['analysis/{}/{}'.format(strf_key, plane_n)]
        #     strf = sca.get_strf_from_nwb(h5_grp=strf_grp, roi_ind=roi_ind, trace_type=trace_type)
        #     return strf
        # except Exception:
        #     return None
    else:
        return None


def get_dgcrm_grp_key(nwb_f):
    analysis_grp = nwb_f['analysis']
    dgcrt_key = [k for k in analysis_grp.keys() if k[0:14] == 'response_table' and 'DriftingGrating' in k]
    if len(dgcrt_key) == 0:
        return None
    elif len(dgcrt_key) == 1:
        return dgcrt_key[0]
    else:
        raise LookupError('more than one drifting grating response table found.')


def get_dgcrm(nwb_f, plane_n, roi_ind, trace_type):

    dgcrm_key = get_dgcrm_grp_key(nwb_f=nwb_f)

    if dgcrm_key is not None:
        dgcrm_grp = nwb_f['analysis/{}/{}'.format(dgcrm_key, plane_n)]
        dgcrm = sca.get_dgc_response_matrix_from_nwb(h5_grp=dgcrm_grp,
                                                     roi_ind=roi_ind,
                                                     trace_type=trace_type)
        return dgcrm
        # try:
        #     dgcrm_grp = nwb_f['analysis/{}/{}'.format(dgcrm_key, plane_n)]
        #     dgcrm = sca.get_dgc_response_matrix_from_nwb(h5_grp=dgcrm_grp,
        #                                                  roi_ind=roi_ind,
        #                                                  trace_type=trace_type)
        #     return dgcrm
        # except Exception as e:
        #     # print(e)
        #     return None
    else:
        return None


def get_rf_properties(srf,
                      polarity,
                      sigma=None, # ANALYSIS_PARAMS['gaussian_filter_sigma_rf'],
                      interpolate_rate=None, #ANALYSIS_PARAMS['interpolate_rate_rf'],
                      z_thr_abs=ANALYSIS_PARAMS['rf_z_thr_abs'],
                      z_thr_rel=ANALYSIS_PARAMS['rf_z_thr_rel']):
    """
    return receptive field properties from a SpatialReceptiveField

    :param srf: SingleCellAnalysis.SpatialReceptiveField object
    :param polarity: str, 'positive' or 'negative', the direction to apply threshold
    :param probe_size: list of two floats, height and width of square size
    :param simgma: float, 2d gaussian filter size, in pixel
    :param interpolate_rate: int, interpolation upsample rate
    :param peak_z_thr:
    :return rf_z: peak absolute zscore after filtering and interpolation
    :return srf_new: corticalmapping.SingleCellAnalysis.SpatialReceptiveField object, filtered, interpolated,
                     thresholded receptive field
    """

    srf_new = srf.copy()

    if sigma is not None:
        srf_new = srf_new.gaussian_filter(sigma=sigma)

    if interpolate_rate is not None:
        srf_new = srf_new.interpolate(ratio=interpolate_rate)

    if polarity == 'positive':
        rf_z = np.max(srf_new.weights)
    elif polarity == 'negative':
        srf_new.weights = -srf_new.weights
        rf_z = np.max(srf_new.weights)
    else:
        raise LookupError('Do not understand "polarity" ({}), should be "positive" or "negative".'.format(polarity))

    if rf_z > (z_thr_abs / z_thr_rel):
        srf_new = srf_new.threshold(thr=(rf_z * z_thr_rel))
    else:
        srf_new = srf_new.threshold(thr=z_thr_abs)
    # rf_center = srf_new.get_weighted_rf_center()
    # rf_area = srf_new.get_binary_rf_area()
    # rf_mask = srf_new.get_weighted_mask()
    return  rf_z, srf_new


def get_roi(nwb_f, plane_n, roi_n):
    """

    :param nwb_f: h5py File object of the nwb file
    :param plane_n:
    :param roi_n:
    :return: core.ImageAnalysis.WeightedROI object of the specified roi
    """

    try:
        pixel_size = nwb_f['acquisition/timeseries/2p_movie_{}/pixel_size'.format(plane_n)][()]
        pixel_size_unit = nwb_f['acquisition/timeseries/2p_movie_{}/pixel_size_unit'.format(plane_n)][()]
    except Exception as e:
        pixel_size = None
        pixel_size_unit = None

    roi_grp = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane/{}'.format(plane_n, roi_n)]
    mask = roi_grp['img_mask'][()]
    return ia.WeightedROI(mask=mask, pixelSize=pixel_size, pixelSizeUnit=pixel_size_unit)


def get_traces(nwb_f, plane_n, trace_type=ANALYSIS_PARAMS['trace_type']):

    traces = nwb_f['processing/rois_and_traces_{}/Fluorescence/{}/data'.format(plane_n, trace_type)][()]
    trace_ts = nwb_f['processing/rois_and_traces_{}/Fluorescence/{}/timestamps'.format(plane_n, trace_type)][()]
    return traces, trace_ts


def get_single_trace(nwb_f, plane_n, roi_n, trace_type=ANALYSIS_PARAMS['trace_type']):
    roi_i = int(roi_n[-4:])
    trace = nwb_f['processing/rois_and_traces_{}/Fluorescence/{}/data'.format(plane_n, trace_type)][roi_i, :]
    trace_ts = nwb_f['processing/rois_and_traces_{}/Fluorescence/{}/timestamps'.format(plane_n, trace_type)][()]
    return trace, trace_ts


def render_rb(rf_on, rf_off, vmax=PLOTTING_PARAMS['rf_zscore_vmax']):

    rf_on = (rf_on / vmax)
    rf_on[rf_on < 0] = 0
    rf_on[rf_on > 1] = 1
    rf_on = np.array(rf_on * 255, dtype=np.uint8)

    rf_off = (rf_off / vmax)
    rf_off[rf_off < 0] = 0
    rf_off[rf_off > 1] = 1
    rf_off = np.array(rf_off * 255, dtype=np.uint8)

    g_channel = np.zeros(rf_on.shape, dtype=np.uint8)
    rf_rgb = np.array([rf_on, g_channel, rf_off]).transpose([1, 2, 0])
    return rf_rgb


def get_UC_ts_mask(nwb_f, plane_n='plane0', len_thr=100):
    """
    return a 1d boolean array, same size as imaging timestamps of the traces in plane_n.
    These index masks represent the time period of all UniformContrast stimuli.

    :param len_thr: uint, the threshold of the number of detected time points. If there are
                    time points less than this number, has_uc will be False

    :return mask: 1d boolean array.
    :return has_uc: bool, False: has no UniformContrast stimulus
                          True: has UniformContrast stimulus
    """

    ts = nwb_f['processing/rois_and_traces_{}/Fluorescence/f_center_raw/timestamps'.format(plane_n)][()]
    mask = np.zeros(ts.shape, dtype=np.bool)

    stim_ns = [n for n in nwb_f['stimulus/presentation'].keys() if 'UniformContrast' in n]

    if len(stim_ns) == 0:
        return mask, False

    else:
        for stim_n in stim_ns:

            stim_dur = nwb_f['stimulus/presentation/{}/duration'.format(stim_n)][()]

            pd_grp = nwb_f['analysis/photodiode_onsets/{}'.format(stim_n)]
            pd_key = pd_grp.keys()[0]
            stim_onset = pd_grp[pd_key]['pd_onset_ts_sec'][0]

            curr_inds = np.logical_and(ts >= stim_onset, ts <= (stim_onset + stim_dur))
            mask = np.logical_or(mask, curr_inds)

        if np.sum(mask) < len_thr:
            return mask, False
        else:
            return mask, True


def get_DGC_spont_ts_mask(nwb_f, plane_n='plane0', len_thr=100):
    """
    return a 1d boolean array, same size as imaging timestamps of the traces in plane_n.
    These index masks represent the time period of blank sweep and second half of intersweep
    intervals. This representing the "spontaneous" period during DriftingGratingCircle stimuli

    :param len_thr: uint, the threshold of the number of detected time points. If there are
                    time points less than this number, has_dgc will be False

    :return mask: 1d boolean array
    :return has_dgc: bool, False: has no DriftingGratingCircle stimulus
                           True: has DriftingGratingCircle stimulus
    """

    ts = nwb_f['processing/rois_and_traces_{}/Fluorescence/f_center_raw/timestamps'.format(plane_n)][()]
    mask = np.zeros(ts.shape, dtype=np.bool)

    stim_ns = [n for n in nwb_f['stimulus/presentation'].keys() if 'DriftingGratingCircle' in n]

    if len(stim_ns) == 0:
        return mask, False

    else:
        for stim_n in stim_ns:
            midgap_dur = nwb_f['stimulus/presentation/{}/midgap_dur'.format(stim_n)][()]
            block_dur = nwb_f['stimulus/presentation/{}/block_dur'.format(stim_n)][()]

            pd_grp = nwb_f['analysis/photodiode_onsets/{}'.format(stim_n)]
            pd_keys = pd_grp.keys()

            for pd_key in pd_keys:

                stim_onsets = pd_grp[pd_key]['pd_onset_ts_sec']

                for stim_onset in stim_onsets:

                    if pd_key[-36:] == 'sf0.00_tf00.0_dire000_con0.00_rad000':  # blank sweeps

                        curr_inds = np.logical_and(ts >= (stim_onset - 0.5 * midgap_dur),
                                                   ts <= (stim_onset + block_dur + midgap_dur))
                        mask = np.logical_or(mask, curr_inds)

                    else: # other sweeps
                        curr_inds = np.logical_and(ts >= (stim_onset - 0.5 * midgap_dur), ts <= stim_onset)
                        mask = np.logical_or(mask, curr_inds)

        if np.sum(mask) < len_thr:
            return mask, False
        else:
            return mask, True


def get_LSN_ts_mask(nwb_f, plane_n='plane0', len_thr=100):
    """
    return a 1d boolean array, same size as imaging timestamps of the traces in plane_n.
    These index masks represent the time period of all LocallySparseNoise stimuli.

    :param len_thr: uint, the threshold of the number of detected time points. If there are
                    time points less than this number, has_lsn will be False

    :return mask: 1d boolean array.
    :return has_lsn: bool, False: has no LocallySparseNoise stimulus
                          True: has LocallySparseNoise stimulus
    """

    ts = nwb_f['processing/rois_and_traces_{}/Fluorescence/f_center_raw/timestamps'.format(plane_n)][()]
    mask = np.zeros(ts.shape, dtype=np.bool)

    stim_ns = [n for n in nwb_f['stimulus/presentation'].keys() if 'LocallySparseNoise' in n]

    if len(stim_ns) == 0:
        return mask, False

    else:
        for stim_n in stim_ns:

            probe_frame_num = nwb_f['stimulus/presentation/{}/probe_frame_num'.format(stim_n)][()]
            probe_dur = probe_frame_num / 60.

            pd_grp = nwb_f['analysis/photodiode_onsets/{}'.format(stim_n)]
            pd_keys = pd_grp.keys()

            stim_onset = None
            stim_offset = None

            for pd_key in pd_keys:
                curr_onsets = pd_grp[pd_key]['pd_onset_ts_sec'][()]

                if stim_onset is None:
                    stim_onset = np.min(curr_onsets)
                else:
                    stim_onset = min([stim_onset, np.min(curr_onsets)])

                if stim_offset is None:
                    stim_offset = np.max(curr_onsets)
                else:
                    stim_offset = max([stim_offset, np.max(curr_onsets)])

            stim_offset = stim_offset + probe_dur

            curr_inds = np.logical_and(ts >= stim_onset, ts <= stim_offset)

            mask = np.logical_or(mask, curr_inds)

        if np.sum(mask) < len_thr:
            return mask, False
        else:
            return mask, True


def plot_roi_retinotopy(coords_roi, coords_rf, ax_alt, ax_azi, alt_range=None, azi_range=None, cmap='viridis',
                        canvas_shape=(512, 512), nan_color='#cccccc', **kwargs):
    """
    plot color coded retinotopy on roi locations
    :param coords_roi: 2d array with shape (n, 2), row and col of roi location
    :param coords_rf: 2d array with same shape of coords_roi, alt and azi locations for each roi
    :param ax_alt: plotting axis for altitude
    :param ax_azi: plotting axis for azimuth
    :param alt_range:
        if None, the range to decide color is [minimum of altitudes of all rois, maximum of altitude of all rois]
        if float, the range to decide color is [median altitude - alt_range, median altitude + alt_range]
        if list of two floats, the range to decide color is [alt_range[0], alt_range[1]]
    :param azi_range: same as alt_range but for azimuth
    :param cmap: matplotlib color map
    :param canvas_shape: plotting shape (height, width)
    :param nan_color: color string, for nan data point, if None, do not plot nan data points
    :param kwargs: inputs to plotting functions
    :return:
    """

    if len(coords_roi.shape) != 2:
        raise ValueError('input coords_roi should be 2d array.')

    if coords_roi.shape[1] != 2:
        raise ValueError('input coords_roi should have 2 columns.')

    if coords_roi.shape != coords_rf.shape:
        raise ValueError('coords_roi and coords_rf should have same shape.')

    if alt_range is None:
        alt_ratio = ia.array_nor(coords_rf[:, 0])
    elif isinstance(alt_range, Number):
        if alt_range > 0:
            alt_median = np.nanmedian(coords_rf[:, 0])
            alt_min = alt_median - float(alt_range)
            alt_max = alt_median + float(alt_range)
            alt_ratio = (coords_rf[:, 0] - alt_min) / (alt_max - alt_min)
        else:
            raise ValueError('if "alt_range" is a number, it should be larger than 0.')
    elif len(alt_range) == 2:
        if alt_range[0] < alt_range[1]:
            alt_ratio = (coords_rf[:, 0] - alt_range[0]) / (alt_range[1] - alt_range[0])
        else:
            raise ValueError('if "alt_range" is a list or a tuple or a array, the first element should be '
                             'smaller than the second element.')
    else:
        raise ValueError('Do not understand input "alt_range", should be None or a single positive number or a '
                         'list or a tuple or a array with two elements with the first element smaller than the '
                         'second.')

    if azi_range is None:
        azi_ratio = ia.array_nor(coords_rf[:, 1])
    elif isinstance(azi_range, Number):
        if azi_range > 0:
            azi_median = np.nanmedian(coords_rf[:, 1])
            azi_min = azi_median - float(azi_range)
            azi_max = azi_median + float(azi_range)
            azi_ratio = (coords_rf[:, 1] - azi_min) / (azi_max - azi_min)
        else:
            raise ValueError('if "azi_range" is a number, it should be larger than 0.')
    elif len(azi_range) == 2:
        if azi_range[0] < azi_range[1]:
            azi_ratio = (coords_rf[:, 1] - azi_range[0]) / (azi_range[1] - azi_range[0])
        else:
            raise ValueError('if "azi_range" is a list or a tuple or a array, the first element should be '
                             'smaller than the second element.')
    else:
        raise ValueError('Do not understand input "azi_range", should be None or a single positive number or a '
                         'list or a tuple or a array with two elements with the first element smaller than the '
                         'second.')

    xs = coords_roi[:, 1]
    ys = coords_roi[:, 0]

    ax_alt.set_xlim([0, canvas_shape[1]])
    ax_alt.set_ylim([0, canvas_shape[0]])
    ax_alt.set_aspect('equal')
    ax_alt.invert_yaxis()
    ax_alt.set_xticks([])
    ax_alt.set_yticks([])

    ax_azi.set_xlim([0, canvas_shape[1]])
    ax_azi.set_ylim([0, canvas_shape[0]])
    ax_azi.set_aspect('equal')
    ax_azi.invert_yaxis()
    ax_azi.set_xticks([])
    ax_azi.set_yticks([])

    for roi_i in range(coords_roi.shape[0]):

        curr_alt_ratio = alt_ratio[roi_i]
        if np.isnan(curr_alt_ratio):
            if nan_color is not None:
                ax_alt.scatter([xs[roi_i]], [ys[roi_i]], marker='o', color=nan_color, **kwargs)
        else:
            alt_c = pt.cmap_2_rgb(curr_alt_ratio, cmap_string=cmap)
            ax_alt.scatter([xs[roi_i]], [ys[roi_i]], marker='o', color=alt_c, **kwargs)

        curr_azi_ratio = azi_ratio[roi_i]
        if np.isnan(curr_azi_ratio):
            if nan_color is not None:
                ax_azi.scatter([xs[roi_i]], [ys[roi_i]], marker='o', color=nan_color, **kwargs)
        else:
            azi_c = pt.cmap_2_rgb(azi_ratio[roi_i], cmap_string=cmap)
            ax_azi.scatter([xs[roi_i]], [ys[roi_i]], marker='o', color=azi_c, **kwargs)


def get_pupil_area(nwb_f, module_name, ell_thr=0.5, median_win=3.):

    pupil_shape = nwb_f['processing/{}/PupilTracking/eyetracking/pupil_shape'.format(module_name)][()]
    pupil_ts = nwb_f['processing/{}/PupilTracking/eyetracking/timestamps'.format(module_name)][()]

    fs = 1. / np.mean(np.diff(pupil_ts))
    # print(fs)

    pupil_area = da.get_pupil_area(pupil_shapes=pupil_shape, fs=fs, ell_thr=ell_thr, median_win=median_win)
    return pupil_area, pupil_ts


def get_running_speed(nwb_f, disk_radius=8., fs_final=30., speed_thr_pos=100., speed_thr_neg=-20.,
                      gauss_sig=1.):

    ref = nwb_f['acquisition/timeseries/analog_running_ref/data'][()]
    sig = nwb_f['acquisition/timeseries/analog_running_sig/data'][()]
    starting_time = nwb_f['acquisition/timeseries/analog_running_ref/starting_time'][()]
    ts_rate = nwb_f['acquisition/timeseries/analog_running_ref/starting_time'].attrs['rate']
    num_sample = nwb_f['acquisition/timeseries/analog_running_ref/num_samples'][()]

    ts = starting_time + np.arange(num_sample) / ts_rate

    speed, speed_ts = da.get_running_speed(sig=sig, ts=ts, ref=ref, disk_radius=disk_radius, fs_final=fs_final,
                                           speed_thr_pos=speed_thr_pos, speed_thr_neg=speed_thr_neg,
                                           gauss_sig=gauss_sig)

    return speed, speed_ts


def plot_roi_contour_on_background(nwb_f, plane_n, plot_ax, **kwargs):
    """
    :param nwb_f:
    :param plane_n:
    :param plot_ax:
    :param kwargs: input variable to corticalmapping.core.PlottingTools.plot_mask_borders
    :return:
    """

    seg_grp = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane'.format(plane_n)]

    if 'max_projection' in seg_grp['reference_images']:
        bg = seg_grp['reference_images/max_projection/data'][()]
        bg = ia.array_nor(bg)
        plot_ax.imshow(bg, vmin=0, vmax=0.8, cmap='gray', interpolation='nearest')
    elif 'max_projection' in seg_grp['reference_images']:
        bg = seg_grp['reference_images/mean_projection/data'][()]
        bg = ia.array_nor(bg)
        plot_ax.imshow(bg, vmin=0, vmax=0.8, cmap='gray', interpolation='nearest')
    else:
        print('cannot find reference image, set background to black')
        # plot_ax.set_facecolor('#000000') # for matplotlib >= v2.0
        plot_ax.set_axis_bgcolor('#000000') # for matplotlib < v2.0

    roi_ns = [r for r in seg_grp['roi_list'] if r[0:4] == 'roi_']
    for roi_n in roi_ns:
        roi_mask = seg_grp[roi_n]['img_mask'][()]
        pt.plot_mask_borders(mask=roi_mask, plotAxis=plot_ax, **kwargs)


def get_everything_from_roi(nwb_f, plane_n, roi_n, params=ANALYSIS_PARAMS, verbose=False):
    """

    :param nwbf: h5py.File object
    :param plane_n:
    :param roi_n:
    :return:
    """

    roi_ind = int(roi_n[-4:])

    roi_properties = {'date': nwb_f['identifier'][()][0:6],
                      'mouse_id': nwb_f['identifier'][()][7:14],
                      'plane_n': plane_n,
                      'roi_n': roi_n,
                      'depth': get_depth(nwb_f=nwb_f, plane_n=plane_n)}

    # get roi properties
    roi = get_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
    pixel_size = nwb_f['acquisition/timeseries/2p_movie_{}/pixel_size'.format(plane_n)][()] * 1000000.
    roi_area = roi.get_binary_area() * pixel_size[0] * pixel_size[1]
    roi_center_row, roi_center_col = roi.get_weighted_center()
    roi_properties.update({'roi_area': roi_area,
                           'roi_center_row': roi_center_row,
                           'roi_center_col': roi_center_col})

    # get skewness
    trace, trace_ts = get_single_trace(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n,
                                       trace_type=params['trace_type'])
    skew_raw, skew_fil = sca.get_skewness(trace=trace, ts=trace_ts,
                                          filter_length=params['filter_length_skew_sec'])
    roi_properties.update({'skew_raw': skew_raw,
                           'skew_fil': skew_fil})

    if np.min(trace) < params['trace_abs_minimum']:
        add_to_trace = -np.min(trace) + params['trace_abs_minimum']
    else:
        add_to_trace = 0.

    strf = get_strf(nwb_f=nwb_f, plane_n=plane_n, roi_ind=roi_ind, trace_type='sta_' + params['trace_type'])
    if strf is not None:

        # get strf properties
        strf_dff = strf.get_local_dff_strf(is_collaps_before_normalize=True, add_to_trace=add_to_trace)

        # positive spatial receptive fields
        srf_pos_on, srf_pos_off = strf_dff.get_zscore_receptive_field(timeWindow=params['response_window_positive_rf'])

        # # get filter sigma in pixels
        # mean_probe_size = (np.abs(np.mean(np.diff(srf_pos_on.altPos))) +
        #                   np.abs(np.mean(np.diff(srf_pos_on.aziPos)))) / 2.
        # print(mean_probe_size)
        # sigma = params['gaussian_filter_sigma_rf'] / mean_probe_size
        # print(sigma)

        # ON positive spatial receptive field
        rf_pos_on_z, rf_pos_on_new = get_rf_properties(srf= srf_pos_on,
                                                       polarity='positive',
                                                       sigma=params['gaussian_filter_sigma_rf'],
                                                       interpolate_rate=params['interpolate_rate_rf'],
                                                       z_thr_abs=params['rf_z_thr_abs'],
                                                       z_thr_rel=params['rf_z_thr_rel'])
        rf_pos_on_area = rf_pos_on_new.get_binary_rf_area()
        rf_pos_on_center = rf_pos_on_new.get_weighted_rf_center()

        roi_properties.update({'rf_pos_on_peak_z': rf_pos_on_z,
                               'rf_pos_on_area': rf_pos_on_area,
                               'rf_pos_on_center_alt': rf_pos_on_center[0],
                               'rf_pos_on_center_azi': rf_pos_on_center[1]})

        # OFF positive spatial receptive field
        rf_pos_off_z, rf_pos_off_new = get_rf_properties(srf=srf_pos_off,
                                                         polarity='positive',
                                                         sigma=params['gaussian_filter_sigma_rf'],
                                                         interpolate_rate=params['interpolate_rate_rf'],
                                                         z_thr_abs=params['rf_z_thr_abs'],
                                                         z_thr_rel=params['rf_z_thr_rel'])

        rf_pos_off_area = rf_pos_off_new.get_binary_rf_area()
        rf_pos_off_center = rf_pos_off_new.get_weighted_rf_center()

        roi_properties.update({'rf_pos_off_peak_z': rf_pos_off_z,
                               'rf_pos_off_area': rf_pos_off_area,
                               'rf_pos_off_center_alt': rf_pos_off_center[0],
                               'rf_pos_off_center_azi': rf_pos_off_center[1]})

        # on off overlapping
        rf_pos_on_mask = rf_pos_on_new.get_weighted_mask()
        rf_pos_off_mask = rf_pos_off_new.get_weighted_mask()
        rf_pos_lsi = sca.get_local_similarity_index(rf_pos_on_mask, rf_pos_off_mask)

        rf_pos_onoff_new = sca.SpatialReceptiveField(mask=np.max([rf_pos_on_mask, rf_pos_off_mask], axis=0),
                                                     altPos=rf_pos_on_new.altPos,
                                                     aziPos=rf_pos_on_new.aziPos,
                                                     sign='ON_OFF',
                                                     thr=params['rf_z_thr_abs'])
        if len(rf_pos_onoff_new.weights) == 0:
            rf_pos_onoff_z = np.nan
        else:
            rf_pos_onoff_z = np.max(rf_pos_onoff_new.weights)
        rf_pos_onoff_area = rf_pos_onoff_new.get_binary_rf_area()
        rf_pos_onoff_center = rf_pos_onoff_new.get_weighted_rf_center()
        roi_properties.update({'rf_pos_lsi': rf_pos_lsi,
                               'rf_pos_onoff_peak_z':rf_pos_onoff_z,
                               'rf_pos_onoff_area': rf_pos_onoff_area,
                               'rf_pos_onoff_center_alt': rf_pos_onoff_center[0],
                               'rf_pos_onoff_center_azi': rf_pos_onoff_center[1]})


        # negative spatial receptive fields
        srf_neg_on, srf_neg_off = strf_dff.get_zscore_receptive_field(timeWindow=params['response_window_negative_rf'])

        # ON negative spatial receptive field
        rf_neg_on_z, rf_neg_on_new = get_rf_properties(srf=srf_neg_on,
                                                       polarity='negative',
                                                       sigma=params['gaussian_filter_sigma_rf'],
                                                       interpolate_rate=params['interpolate_rate_rf'],
                                                       z_thr_abs=params['rf_z_thr_abs'],
                                                       z_thr_rel=params['rf_z_thr_rel'])
        rf_neg_on_area = rf_neg_on_new.get_binary_rf_area()
        rf_neg_on_center = rf_neg_on_new.get_weighted_rf_center()
        roi_properties.update({'rf_neg_on_peak_z': rf_neg_on_z,
                               'rf_neg_on_area': rf_neg_on_area,
                               'rf_neg_on_center_alt': rf_neg_on_center[0],
                               'rf_neg_on_center_azi': rf_neg_on_center[1]})

        # OFF negative spatial receptive field
        rf_neg_off_z, rf_neg_off_new = get_rf_properties(srf=srf_neg_off,
                                                         polarity='negative',
                                                         sigma=params['gaussian_filter_sigma_rf'],
                                                         interpolate_rate=params['interpolate_rate_rf'],
                                                         z_thr_abs=params['rf_z_thr_abs'],
                                                         z_thr_rel=params['rf_z_thr_rel'])
        rf_neg_off_area = rf_neg_off_new.get_binary_rf_area()
        rf_neg_off_center = rf_neg_off_new.get_weighted_rf_center()
        roi_properties.update({'rf_neg_off_peak_z': rf_neg_off_z,
                               'rf_neg_off_area': rf_neg_off_area,
                               'rf_neg_off_center_alt': rf_neg_off_center[0],
                               'rf_neg_off_center_azi': rf_neg_off_center[1]})

        # on off overlapping
        rf_neg_on_mask = rf_neg_on_new.get_weighted_mask()
        rf_neg_off_mask = rf_neg_off_new.get_weighted_mask()
        rf_neg_lsi = sca.get_local_similarity_index(rf_neg_on_mask, rf_neg_off_mask)

        rf_neg_onoff_new = sca.SpatialReceptiveField(mask=np.max([rf_neg_on_mask, rf_neg_off_mask], axis=0),
                                                     altPos=rf_neg_on_new.altPos,
                                                     aziPos=rf_neg_on_new.aziPos,
                                                     sign='ON_OFF',
                                                     thr=params['rf_z_thr_abs'])
        if len(rf_neg_onoff_new.weights) == 0:
            rf_neg_onoff_z = np.nan
        else:
            rf_neg_onoff_z = np.max(rf_neg_onoff_new.weights)
        rf_neg_onoff_area = rf_neg_onoff_new.get_binary_rf_area()
        rf_neg_onoff_center = rf_neg_onoff_new.get_weighted_rf_center()
        roi_properties.update({'rf_neg_onoff_peak_z': rf_neg_onoff_z,
                               'rf_neg_onoff_area': rf_neg_onoff_area,
                               'rf_neg_onoff_center_alt': rf_neg_onoff_center[0],
                               'rf_neg_onoff_center_azi': rf_neg_onoff_center[1],
                               'rf_neg_lsi': rf_neg_lsi})
    else:
        srf_pos_on = None
        srf_pos_off = None
        srf_neg_on = None
        srf_neg_off = None

        roi_properties.update({'rf_pos_on_peak_z': np.nan,
                               'rf_pos_on_area': np.nan,
                               'rf_pos_on_center_alt': np.nan,
                               'rf_pos_on_center_azi': np.nan,
                               'rf_pos_off_peak_z': np.nan,
                               'rf_pos_off_area': np.nan,
                               'rf_pos_off_center_alt': np.nan,
                               'rf_pos_off_center_azi': np.nan,
                               'rf_pos_onoff_peak_z': np.nan,
                               'rf_pos_onoff_area': np.nan,
                               'rf_pos_onoff_center_alt': np.nan,
                               'rf_pos_onoff_center_azi': np.nan,
                               'rf_pos_lsi': np.nan,
                               'rf_neg_on_peak_z': np.nan,
                               'rf_neg_on_area': np.nan,
                               'rf_neg_on_center_alt': np.nan,
                               'rf_neg_on_center_azi': np.nan,
                               'rf_neg_off_peak_z': np.nan,
                               'rf_neg_off_area': np.nan,
                               'rf_neg_off_center_alt': np.nan,
                               'rf_neg_off_center_azi': np.nan,
                               'rf_neg_onoff_peak_z': np.nan,
                               'rf_neg_onoff_area': np.nan,
                               'rf_neg_onoff_center_alt': np.nan,
                               'rf_neg_onoff_center_azi': np.nan,
                               'rf_neg_lsi': np.nan,
                               })


    # analyze response to drifring grating
    dgcrm = get_dgcrm(nwb_f=nwb_f, plane_n=plane_n, roi_ind=roi_ind, trace_type='sta_' + params['trace_type'])
    if dgcrm is not None:
        dgcrm_grp_key = get_dgcrm_grp_key(nwb_f=nwb_f)
        dgc_block_dur = nwb_f['stimulus/presentation/{}/block_dur'.format(dgcrm_grp_key[15:])][()]
        # print('block duration: {}'.format(block_dur))

        # get df statistics ============================================================================================
        _ = dgcrm.get_df_response_table(baseline_win=params['baseline_window_dgc'],
                                        response_win=params['response_window_dgc'])
        dgcrt_df, dgc_p_anova_df, dgc_pos_p_ttest_df, dgc_neg_p_ttest_df = _
        roi_properties.update({'dgc_pos_peak_df': dgcrt_df.peak_response_pos,
                               'dgc_neg_peak_df': dgcrt_df.peak_response_neg,
                               'dgc_pos_p_ttest_df': dgc_pos_p_ttest_df,
                               'dgc_neg_p_ttest_df': dgc_neg_p_ttest_df,
                               'dgc_p_anova_df': dgc_p_anova_df})

        # get dff statics ==============================================================================================
        _ = dgcrm.get_dff_response_table(baseline_win=params['baseline_window_dgc'],
                                         response_win=params['response_window_dgc'],
                                         bias=add_to_trace)
        dgcrt_dff, dgc_p_anova_dff, dgc_pos_p_ttest_dff, dgc_neg_p_ttest_dff = _
        roi_properties.update({'dgc_pos_peak_dff': dgcrt_dff.peak_response_pos,
                               'dgc_neg_peak_dff': dgcrt_dff.peak_response_neg,
                               'dgc_pos_p_ttest_dff': dgc_pos_p_ttest_dff,
                               'dgc_neg_p_ttest_dff': dgc_neg_p_ttest_dff,
                               'dgc_p_anova_dff': dgc_p_anova_dff})

        # get zscore statistics ========================================================================================
        _ = dgcrm.get_zscore_response_table(baseline_win=params['baseline_window_dgc'],
                                            response_win=params['response_window_dgc'])
        dgcrt_z, dgc_p_anova_z, dgc_pos_p_ttest_z, dgc_neg_p_ttest_z = _
        roi_properties.update({'dgc_pos_peak_z': dgcrt_z.peak_response_pos,
                               'dgc_neg_peak_z': dgcrt_z.peak_response_neg,
                               'dgc_pos_p_ttest_z': dgc_pos_p_ttest_z,
                               'dgc_neg_p_ttest_z': dgc_neg_p_ttest_z,
                               'dgc_p_anova_z': dgc_p_anova_z})

        # get dgc response matrices ====================================================================================
        dgcrm_df = dgcrm.get_df_response_matrix(baseline_win=params['baseline_window_dgc'])
        dgcrm_dff = dgcrm.get_dff_response_matrix(baseline_win=params['baseline_window_dgc'],
                                                  bias=add_to_trace)
        dgcrm_z = dgcrm.get_zscore_response_matrix(baseline_win=params['baseline_window_dgc'])


        # direction/orientation tuning of df responses in positive direction ===========================================
        dire_tuning_df_pos = dgcrt_df.get_dire_tuning(response_dir='pos',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_df_pos_raw, dsi_df_pos_raw, gosi_df_pos_raw, gdsi_df_pos_raw, \
        osi_df_pos_ele, dsi_df_pos_ele, gosi_df_pos_ele, gdsi_df_pos_ele, \
        osi_df_pos_rec, dsi_df_pos_rec, gosi_df_pos_rec, gdsi_df_pos_rec, \
        peak_dire_raw_df_pos, vs_dire_raw_df_pos, vs_dire_ele_df_pos, vs_dire_rec_df_pos\
            = dgcrt_df.get_dire_tuning_properties(dire_tuning_df_pos,
                                                  response_dir='pos',
                                                  elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_osi_raw_df': osi_df_pos_raw,
                               'dgc_pos_dsi_raw_df': dsi_df_pos_raw,
                               'dgc_pos_gosi_raw_df': gosi_df_pos_raw,
                               'dgc_pos_gdsi_raw_df': gdsi_df_pos_raw,
                               'dgc_pos_osi_ele_df': osi_df_pos_ele,
                               'dgc_pos_dsi_ele_df': dsi_df_pos_ele,
                               'dgc_pos_gosi_ele_df': gosi_df_pos_ele,
                               'dgc_pos_gdsi_ele_df': gdsi_df_pos_ele,
                               'dgc_pos_osi_rec_df': osi_df_pos_rec,
                               'dgc_pos_dsi_rec_df': dsi_df_pos_rec,
                               'dgc_pos_gosi_rec_df': gosi_df_pos_rec,
                               'dgc_pos_gdsi_rec_df': gdsi_df_pos_rec,
                               'dgc_pos_peak_dire_raw_df': peak_dire_raw_df_pos,
                               'dgc_pos_vs_dire_raw_df': vs_dire_raw_df_pos,
                               'dgc_pos_vs_dire_ele_df': vs_dire_ele_df_pos,
                               'dgc_pos_vs_dire_rec_df': vs_dire_rec_df_pos})


        # direction/orientation tuning of df responses in negative direction ===========================================
        dire_tuning_df_neg = dgcrt_df.get_dire_tuning(response_dir='neg',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_df_neg_raw, dsi_df_neg_raw, gosi_df_neg_raw, gdsi_df_neg_raw, \
        osi_df_neg_ele, dsi_df_neg_ele, gosi_df_neg_ele, gdsi_df_neg_ele, \
        osi_df_neg_rec, dsi_df_neg_rec, gosi_df_neg_rec, gdsi_df_neg_rec, \
        peak_dire_raw_df_neg, vs_dire_raw_df_neg, vs_dire_ele_df_neg, vs_dire_rec_df_neg \
            = dgcrt_df.get_dire_tuning_properties(dire_tuning_df_neg,
                                                  response_dir='neg',
                                                  elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_osi_raw_df': osi_df_neg_raw,
                               'dgc_neg_dsi_raw_df': dsi_df_neg_raw,
                               'dgc_neg_gosi_raw_df': gosi_df_neg_raw,
                               'dgc_neg_gdsi_raw_df': gdsi_df_neg_raw,
                               'dgc_neg_osi_ele_df': osi_df_neg_ele,
                               'dgc_neg_dsi_ele_df': dsi_df_neg_ele,
                               'dgc_neg_gosi_ele_df': gosi_df_neg_ele,
                               'dgc_neg_gdsi_ele_df': gdsi_df_neg_ele,
                               'dgc_neg_osi_rec_df': osi_df_neg_rec,
                               'dgc_neg_dsi_rec_df': dsi_df_neg_rec,
                               'dgc_neg_gosi_rec_df': gosi_df_neg_rec,
                               'dgc_neg_gdsi_rec_df': gdsi_df_neg_rec,
                               'dgc_neg_peak_dire_raw_df': peak_dire_raw_df_neg,
                               'dgc_neg_vs_dire_raw_df': vs_dire_raw_df_neg,
                               'dgc_neg_vs_dire_ele_df': vs_dire_ele_df_neg,
                               'dgc_neg_vs_dire_rec_df': vs_dire_rec_df_neg})


        # sf tuning of df responses in positive direction ==============================================================
        sf_tuning_df_pos = dgcrt_df.get_sf_tuning(response_dir='pos', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_df_pos, weighted_sf_raw_df_pos, weighted_sf_log_raw_df_pos, \
                            weighted_sf_ele_df_pos, weighted_sf_log_ele_df_pos, \
                            weighted_sf_rec_df_pos, weighted_sf_log_rec_df_pos= \
            dgcrt_df.get_sf_tuning_properties(sf_tuning_df_pos, response_dir='pos',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_peak_sf_raw_df': peak_sf_raw_df_pos,
                               'dgc_pos_weighted_sf_raw_df': weighted_sf_raw_df_pos,
                               'dgc_pos_weighted_sf_log_raw_df': weighted_sf_log_raw_df_pos,
                               'dgc_pos_weighted_sf_ele_df': weighted_sf_ele_df_pos,
                               'dgc_pos_weighted_sf_log_ele_df': weighted_sf_log_ele_df_pos,
                               'dgc_pos_weighted_sf_rec_df': weighted_sf_rec_df_pos,
                               'dgc_pos_weighted_sf_log_rec_df': weighted_sf_log_rec_df_pos})


        # sf tuning of df responses in negative direction ==============================================================
        sf_tuning_df_neg = dgcrt_df.get_sf_tuning(response_dir='neg', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_df_neg, weighted_sf_raw_df_neg, weighted_sf_log_raw_df_neg, \
        weighted_sf_ele_df_neg, weighted_sf_log_ele_df_neg, \
        weighted_sf_rec_df_neg, weighted_sf_log_rec_df_neg = \
            dgcrt_df.get_sf_tuning_properties(sf_tuning_df_neg, response_dir='neg',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_peak_sf_raw_df': peak_sf_raw_df_neg,
                               'dgc_neg_weighted_sf_raw_df': weighted_sf_raw_df_neg,
                               'dgc_neg_weighted_sf_log_raw_df': weighted_sf_log_raw_df_neg,
                               'dgc_neg_weighted_sf_ele_df': weighted_sf_ele_df_neg,
                               'dgc_neg_weighted_sf_log_ele_df': weighted_sf_log_ele_df_neg,
                               'dgc_neg_weighted_sf_rec_df': weighted_sf_rec_df_neg,
                               'dgc_neg_weighted_sf_log_rec_df': weighted_sf_log_rec_df_neg})


        # tf tuning of df responses in positive direction ==============================================================
        tf_tuning_df_pos = dgcrt_df.get_tf_tuning(response_dir='pos', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_df_pos, weighted_tf_raw_df_pos, weighted_tf_log_raw_df_pos, \
        weighted_tf_ele_df_pos, weighted_tf_log_ele_df_pos, \
        weighted_tf_rec_df_pos, weighted_tf_log_rec_df_pos = \
            dgcrt_df.get_tf_tuning_properties(tf_tuning_df_pos, response_dir='pos',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_peak_tf_raw_df': peak_tf_raw_df_pos,
                               'dgc_pos_weighted_tf_raw_df': weighted_tf_raw_df_pos,
                               'dgc_pos_weighted_tf_log_raw_df': weighted_tf_log_raw_df_pos,
                               'dgc_pos_weighted_tf_ele_df': weighted_tf_ele_df_pos,
                               'dgc_pos_weighted_tf_log_ele_df': weighted_tf_log_ele_df_pos,
                               'dgc_pos_weighted_tf_rec_df': weighted_tf_rec_df_pos,
                               'dgc_pos_weighted_tf_log_rec_df': weighted_tf_log_rec_df_pos})

        # tf tuning of df responses in negative direction ==============================================================
        tf_tuning_df_neg = dgcrt_df.get_tf_tuning(response_dir='neg', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_df_neg, weighted_tf_raw_df_neg, weighted_tf_log_raw_df_neg, \
        weighted_tf_ele_df_neg, weighted_tf_log_ele_df_neg, \
        weighted_tf_rec_df_neg, weighted_tf_log_rec_df_neg = \
            dgcrt_df.get_tf_tuning_properties(tf_tuning_df_neg, response_dir='neg',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_peak_tf_raw_df': peak_tf_raw_df_neg,
                               'dgc_neg_weighted_tf_raw_df': weighted_tf_raw_df_neg,
                               'dgc_neg_weighted_tf_log_raw_df': weighted_tf_log_raw_df_neg,
                               'dgc_neg_weighted_tf_ele_df': weighted_tf_ele_df_neg,
                               'dgc_neg_weighted_tf_log_ele_df': weighted_tf_log_ele_df_neg,
                               'dgc_neg_weighted_tf_rec_df': weighted_tf_rec_df_neg,
                               'dgc_neg_weighted_tf_log_rec_df': weighted_tf_log_rec_df_neg})

        # direction/orientation tuning of dff responses in positive direction ===========================================
        dire_tuning_dff_pos = dgcrt_dff.get_dire_tuning(response_dir='pos',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_dff_pos_raw, dsi_dff_pos_raw, gosi_dff_pos_raw, gdsi_dff_pos_raw, \
        osi_dff_pos_ele, dsi_dff_pos_ele, gosi_dff_pos_ele, gdsi_dff_pos_ele, \
        osi_dff_pos_rec, dsi_dff_pos_rec, gosi_dff_pos_rec, gdsi_dff_pos_rec, \
        peak_dire_raw_dff_pos, vs_dire_raw_dff_pos, vs_dire_ele_dff_pos, vs_dire_rec_dff_pos \
            = dgcrt_dff.get_dire_tuning_properties(dire_tuning_dff_pos,
                                                  response_dir='pos',
                                                  elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_osi_raw_dff': osi_dff_pos_raw,
                               'dgc_pos_dsi_raw_dff': dsi_dff_pos_raw,
                               'dgc_pos_gosi_raw_dff': gosi_dff_pos_raw,
                               'dgc_pos_gdsi_raw_dff': gdsi_dff_pos_raw,
                               'dgc_pos_osi_ele_dff': osi_dff_pos_ele,
                               'dgc_pos_dsi_ele_dff': dsi_dff_pos_ele,
                               'dgc_pos_gosi_ele_dff': gosi_dff_pos_ele,
                               'dgc_pos_gdsi_ele_dff': gdsi_dff_pos_ele,
                               'dgc_pos_osi_rec_dff': osi_dff_pos_rec,
                               'dgc_pos_dsi_rec_dff': dsi_dff_pos_rec,
                               'dgc_pos_gosi_rec_dff': gosi_dff_pos_rec,
                               'dgc_pos_gdsi_rec_dff': gdsi_dff_pos_rec,
                               'dgc_pos_peak_dire_raw_dff': peak_dire_raw_dff_pos,
                               'dgc_pos_vs_dire_raw_dff': vs_dire_raw_dff_pos,
                               'dgc_pos_vs_dire_ele_dff': vs_dire_ele_dff_pos,
                               'dgc_pos_vs_dire_rec_dff': vs_dire_rec_dff_pos})

        # direction/orientation tuning of dff responses in negative direction ===========================================
        dire_tuning_dff_neg = dgcrt_dff.get_dire_tuning(response_dir='neg',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_dff_neg_raw, dsi_dff_neg_raw, gosi_dff_neg_raw, gdsi_dff_neg_raw, \
        osi_dff_neg_ele, dsi_dff_neg_ele, gosi_dff_neg_ele, gdsi_dff_neg_ele, \
        osi_dff_neg_rec, dsi_dff_neg_rec, gosi_dff_neg_rec, gdsi_dff_neg_rec, \
        peak_dire_raw_dff_neg, vs_dire_raw_dff_neg, vs_dire_ele_dff_neg, vs_dire_rec_dff_neg \
            = dgcrt_dff.get_dire_tuning_properties(dire_tuning_dff_neg,
                                                  response_dir='neg',
                                                  elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_osi_raw_dff': osi_dff_neg_raw,
                               'dgc_neg_dsi_raw_dff': dsi_dff_neg_raw,
                               'dgc_neg_gosi_raw_dff': gosi_dff_neg_raw,
                               'dgc_neg_gdsi_raw_dff': gdsi_dff_neg_raw,
                               'dgc_neg_osi_ele_dff': osi_dff_neg_ele,
                               'dgc_neg_dsi_ele_dff': dsi_dff_neg_ele,
                               'dgc_neg_gosi_ele_dff': gosi_dff_neg_ele,
                               'dgc_neg_gdsi_ele_dff': gdsi_dff_neg_ele,
                               'dgc_neg_osi_rec_dff': osi_dff_neg_rec,
                               'dgc_neg_dsi_rec_dff': dsi_dff_neg_rec,
                               'dgc_neg_gosi_rec_dff': gosi_dff_neg_rec,
                               'dgc_neg_gdsi_rec_dff': gdsi_dff_neg_rec,
                               'dgc_neg_peak_dire_raw_dff': peak_dire_raw_dff_neg,
                               'dgc_neg_vs_dire_raw_dff': vs_dire_raw_dff_neg,
                               'dgc_neg_vs_dire_ele_dff': vs_dire_ele_dff_neg,
                               'dgc_neg_vs_dire_rec_dff': vs_dire_rec_dff_neg})

        # sf tuning of dff responses in positive direction ==============================================================
        sf_tuning_dff_pos = dgcrt_dff.get_sf_tuning(response_dir='pos', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_dff_pos, weighted_sf_raw_dff_pos, weighted_sf_log_raw_dff_pos, \
        weighted_sf_ele_dff_pos, weighted_sf_log_ele_dff_pos, \
        weighted_sf_rec_dff_pos, weighted_sf_log_rec_dff_pos = \
            dgcrt_dff.get_sf_tuning_properties(sf_tuning_dff_pos, response_dir='pos',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_peak_sf_raw_dff': peak_sf_raw_dff_pos,
                               'dgc_pos_weighted_sf_raw_dff': weighted_sf_raw_dff_pos,
                               'dgc_pos_weighted_sf_log_raw_dff': weighted_sf_log_raw_dff_pos,
                               'dgc_pos_weighted_sf_ele_dff': weighted_sf_ele_dff_pos,
                               'dgc_pos_weighted_sf_log_ele_dff': weighted_sf_log_ele_dff_pos,
                               'dgc_pos_weighted_sf_rec_dff': weighted_sf_rec_dff_pos,
                               'dgc_pos_weighted_sf_log_rec_dff': weighted_sf_log_rec_dff_pos})

        # sf tuning of dff responses in negative direction ==============================================================
        sf_tuning_dff_neg = dgcrt_dff.get_sf_tuning(response_dir='neg', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_dff_neg, weighted_sf_raw_dff_neg, weighted_sf_log_raw_dff_neg, \
        weighted_sf_ele_dff_neg, weighted_sf_log_ele_dff_neg, \
        weighted_sf_rec_dff_neg, weighted_sf_log_rec_dff_neg = \
            dgcrt_dff.get_sf_tuning_properties(sf_tuning_dff_neg, response_dir='neg',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_peak_sf_raw_dff': peak_sf_raw_dff_neg,
                               'dgc_neg_weighted_sf_raw_dff': weighted_sf_raw_dff_neg,
                               'dgc_neg_weighted_sf_log_raw_dff': weighted_sf_log_raw_dff_neg,
                               'dgc_neg_weighted_sf_ele_dff': weighted_sf_ele_dff_neg,
                               'dgc_neg_weighted_sf_log_ele_dff': weighted_sf_log_ele_dff_neg,
                               'dgc_neg_weighted_sf_rec_dff': weighted_sf_rec_dff_neg,
                               'dgc_neg_weighted_sf_log_rec_dff': weighted_sf_log_rec_dff_neg})

        # tf tuning of dff responses in positive direction ==============================================================
        tf_tuning_dff_pos = dgcrt_dff.get_tf_tuning(response_dir='pos', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_dff_pos, weighted_tf_raw_dff_pos, weighted_tf_log_raw_dff_pos, \
        weighted_tf_ele_dff_pos, weighted_tf_log_ele_dff_pos, \
        weighted_tf_rec_dff_pos, weighted_tf_log_rec_dff_pos = \
            dgcrt_dff.get_tf_tuning_properties(tf_tuning_dff_pos, response_dir='pos',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_peak_tf_raw_dff': peak_tf_raw_dff_pos,
                               'dgc_pos_weighted_tf_raw_dff': weighted_tf_raw_dff_pos,
                               'dgc_pos_weighted_tf_log_raw_dff': weighted_tf_log_raw_dff_pos,
                               'dgc_pos_weighted_tf_ele_dff': weighted_tf_ele_dff_pos,
                               'dgc_pos_weighted_tf_log_ele_dff': weighted_tf_log_ele_dff_pos,
                               'dgc_pos_weighted_tf_rec_dff': weighted_tf_rec_dff_pos,
                               'dgc_pos_weighted_tf_log_rec_dff': weighted_tf_log_rec_dff_pos})

        # tf tuning of dff responses in negative direction ==============================================================
        tf_tuning_dff_neg = dgcrt_dff.get_tf_tuning(response_dir='neg', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_dff_neg, weighted_tf_raw_dff_neg, weighted_tf_log_raw_dff_neg, \
        weighted_tf_ele_dff_neg, weighted_tf_log_ele_dff_neg, \
        weighted_tf_rec_dff_neg, weighted_tf_log_rec_dff_neg = \
            dgcrt_dff.get_tf_tuning_properties(tf_tuning_dff_neg, response_dir='neg',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_peak_tf_raw_dff': peak_tf_raw_dff_neg,
                               'dgc_neg_weighted_tf_raw_dff': weighted_tf_raw_dff_neg,
                               'dgc_neg_weighted_tf_log_raw_dff': weighted_tf_log_raw_dff_neg,
                               'dgc_neg_weighted_tf_ele_dff': weighted_tf_ele_dff_neg,
                               'dgc_neg_weighted_tf_log_ele_dff': weighted_tf_log_ele_dff_neg,
                               'dgc_neg_weighted_tf_rec_dff': weighted_tf_rec_dff_neg,
                               'dgc_neg_weighted_tf_log_rec_dff': weighted_tf_log_rec_dff_neg})


        # direction/orientation tuning of zscore responses in positive direction ===========================================
        dire_tuning_z_pos = dgcrt_z.get_dire_tuning(response_dir='pos',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_z_pos_raw, dsi_z_pos_raw, gosi_z_pos_raw, gdsi_z_pos_raw, \
        osi_z_pos_ele, dsi_z_pos_ele, gosi_z_pos_ele, gdsi_z_pos_ele, \
        osi_z_pos_rec, dsi_z_pos_rec, gosi_z_pos_rec, gdsi_z_pos_rec, \
        peak_dire_raw_z_pos, vs_dire_raw_z_pos, vs_dire_ele_z_pos, vs_dire_rec_z_pos \
            = dgcrt_z.get_dire_tuning_properties(dire_tuning_z_pos,
                                                  response_dir='pos',
                                                  elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_osi_raw_z': osi_z_pos_raw,
                               'dgc_pos_dsi_raw_z': dsi_z_pos_raw,
                               'dgc_pos_gosi_raw_z': gosi_z_pos_raw,
                               'dgc_pos_gdsi_raw_z': gdsi_z_pos_raw,
                               'dgc_pos_osi_ele_z': osi_z_pos_ele,
                               'dgc_pos_dsi_ele_z': dsi_z_pos_ele,
                               'dgc_pos_gosi_ele_z': gosi_z_pos_ele,
                               'dgc_pos_gdsi_ele_z': gdsi_z_pos_ele,
                               'dgc_pos_osi_rec_z': osi_z_pos_rec,
                               'dgc_pos_dsi_rec_z': dsi_z_pos_rec,
                               'dgc_pos_gosi_rec_z': gosi_z_pos_rec,
                               'dgc_pos_gdsi_rec_z': gdsi_z_pos_rec,
                               'dgc_pos_peak_dire_raw_z': peak_dire_raw_z_pos,
                               'dgc_pos_vs_dire_raw_z': vs_dire_raw_z_pos,
                               'dgc_pos_vs_dire_ele_z': vs_dire_ele_z_pos,
                               'dgc_pos_vs_dire_rec_z': vs_dire_rec_z_pos})

        # direction/orientation tuning of zscore responses in negative direction ===========================================
        dire_tuning_z_neg = dgcrt_z.get_dire_tuning(response_dir='neg',
                                                      is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_tf=params['is_collapse_tf'])
        osi_z_neg_raw, dsi_z_neg_raw, gosi_z_neg_raw, gdsi_z_neg_raw, \
        osi_z_neg_ele, dsi_z_neg_ele, gosi_z_neg_ele, gdsi_z_neg_ele, \
        osi_z_neg_rec, dsi_z_neg_rec, gosi_z_neg_rec, gdsi_z_neg_rec, \
        peak_dire_raw_z_neg, vs_dire_raw_z_neg, vs_dire_ele_z_neg, vs_dire_rec_z_neg \
            = dgcrt_z.get_dire_tuning_properties(dire_tuning_z_neg,
                                                  response_dir='neg',
                                                  elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_osi_raw_z': osi_z_neg_raw,
                               'dgc_neg_dsi_raw_z': dsi_z_neg_raw,
                               'dgc_neg_gosi_raw_z': gosi_z_neg_raw,
                               'dgc_neg_gdsi_raw_z': gdsi_z_neg_raw,
                               'dgc_neg_osi_ele_z': osi_z_neg_ele,
                               'dgc_neg_dsi_ele_z': dsi_z_neg_ele,
                               'dgc_neg_gosi_ele_z': gosi_z_neg_ele,
                               'dgc_neg_gdsi_ele_z': gdsi_z_neg_ele,
                               'dgc_neg_osi_rec_z': osi_z_neg_rec,
                               'dgc_neg_dsi_rec_z': dsi_z_neg_rec,
                               'dgc_neg_gosi_rec_z': gosi_z_neg_rec,
                               'dgc_neg_gdsi_rec_z': gdsi_z_neg_rec,
                               'dgc_neg_peak_dire_raw_z': peak_dire_raw_z_neg,
                               'dgc_neg_vs_dire_raw_z': vs_dire_raw_z_neg,
                               'dgc_neg_vs_dire_ele_z': vs_dire_ele_z_neg,
                               'dgc_neg_vs_dire_rec_z': vs_dire_rec_z_neg})

        # sf tuning of zscore responses in positive direction ==============================================================
        sf_tuning_z_pos = dgcrt_z.get_sf_tuning(response_dir='pos', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_z_pos, weighted_sf_raw_z_pos, weighted_sf_log_raw_z_pos, \
        weighted_sf_ele_z_pos, weighted_sf_log_ele_z_pos, \
        weighted_sf_rec_z_pos, weighted_sf_log_rec_z_pos = \
            dgcrt_z.get_sf_tuning_properties(sf_tuning_z_pos, response_dir='pos',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_peak_sf_raw_z': peak_sf_raw_z_pos,
                               'dgc_pos_weighted_sf_raw_z': weighted_sf_raw_z_pos,
                               'dgc_pos_weighted_sf_log_raw_z': weighted_sf_log_raw_z_pos,
                               'dgc_pos_weighted_sf_ele_z': weighted_sf_ele_z_pos,
                               'dgc_pos_weighted_sf_log_ele_z': weighted_sf_log_ele_z_pos,
                               'dgc_pos_weighted_sf_rec_z': weighted_sf_rec_z_pos,
                               'dgc_pos_weighted_sf_log_rec_z': weighted_sf_log_rec_z_pos})

        # sf tuning of zscore responses in negative direction ==============================================================
        sf_tuning_z_neg = dgcrt_z.get_sf_tuning(response_dir='neg', is_collapse_tf=params['is_collapse_tf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_sf_raw_z_neg, weighted_sf_raw_z_neg, weighted_sf_log_raw_z_neg, \
        weighted_sf_ele_z_neg, weighted_sf_log_ele_z_neg, \
        weighted_sf_rec_z_neg, weighted_sf_log_rec_z_neg = \
            dgcrt_z.get_sf_tuning_properties(sf_tuning_z_neg, response_dir='neg',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_peak_sf_raw_z': peak_sf_raw_z_neg,
                               'dgc_neg_weighted_sf_raw_z': weighted_sf_raw_z_neg,
                               'dgc_neg_weighted_sf_log_raw_z': weighted_sf_log_raw_z_neg,
                               'dgc_neg_weighted_sf_ele_z': weighted_sf_ele_z_neg,
                               'dgc_neg_weighted_sf_log_ele_z': weighted_sf_log_ele_z_neg,
                               'dgc_neg_weighted_sf_rec_z': weighted_sf_rec_z_neg,
                               'dgc_neg_weighted_sf_log_rec_z': weighted_sf_log_rec_z_neg})

        # tf tuning of zcore responses in positive direction ==============================================================
        tf_tuning_z_pos = dgcrt_z.get_tf_tuning(response_dir='pos', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_z_pos, weighted_tf_raw_z_pos, weighted_tf_log_raw_z_pos, \
        weighted_tf_ele_z_pos, weighted_tf_log_ele_z_pos, \
        weighted_tf_rec_z_pos, weighted_tf_log_rec_z_pos = \
            dgcrt_z.get_tf_tuning_properties(tf_tuning_z_pos, response_dir='pos',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_pos_peak_tf_raw_z': peak_tf_raw_z_pos,
                               'dgc_pos_weighted_tf_raw_z': weighted_tf_raw_z_pos,
                               'dgc_pos_weighted_tf_log_raw_z': weighted_tf_log_raw_z_pos,
                               'dgc_pos_weighted_tf_ele_z': weighted_tf_ele_z_pos,
                               'dgc_pos_weighted_tf_log_ele_z': weighted_tf_log_ele_z_pos,
                               'dgc_pos_weighted_tf_rec_z': weighted_tf_rec_z_pos,
                               'dgc_pos_weighted_tf_log_rec_z': weighted_tf_log_rec_z_pos})

        # tf tuning of zscore responses in negative direction ==============================================================
        tf_tuning_z_neg = dgcrt_z.get_tf_tuning(response_dir='neg', is_collapse_sf=params['is_collapse_sf'],
                                                  is_collapse_dire=params['is_collapse_dire'])
        peak_tf_raw_z_neg, weighted_tf_raw_z_neg, weighted_tf_log_raw_z_neg, \
        weighted_tf_ele_z_neg, weighted_tf_log_ele_z_neg, \
        weighted_tf_rec_z_neg, weighted_tf_log_rec_z_neg = \
            dgcrt_z.get_tf_tuning_properties(tf_tuning_z_neg, response_dir='neg',
                                              elevation_bias=params['dgc_elevation_bias'])
        roi_properties.update({'dgc_neg_peak_tf_raw_z': peak_tf_raw_z_neg,
                               'dgc_neg_weighted_tf_raw_z': weighted_tf_raw_z_neg,
                               'dgc_neg_weighted_tf_log_raw_z': weighted_tf_log_raw_z_neg,
                               'dgc_neg_weighted_tf_ele_z': weighted_tf_ele_z_neg,
                               'dgc_neg_weighted_tf_log_ele_z': weighted_tf_log_ele_z_neg,
                               'dgc_neg_weighted_tf_rec_z': weighted_tf_rec_z_neg,
                               'dgc_neg_weighted_tf_log_rec_z': weighted_tf_log_rec_z_neg})

    else:
        dgcrm_df = None
        dgcrm_dff = None
        dgcrm_z = None
        dgcrt_df = None
        dgcrt_dff = None
        dgcrt_z = None
        dgc_block_dur = None

        roi_properties.update({'dgc_pos_peak_df': np.nan,
                               'dgc_neg_peak_df': np.nan,
                               'dgc_pos_p_ttest_df': np.nan,
                               'dgc_neg_p_ttest_df': np.nan,
                               'dgc_p_anova_df': np.nan,
                               'dgc_pos_peak_dff': np.nan,
                               'dgc_neg_peak_dff': np.nan,
                               'dgc_pos_p_ttest_dff': np.nan,
                               'dgc_neg_p_ttest_dff': np.nan,
                               'dgc_p_anova_dff': np.nan,
                               'dgc_pos_peak_z': np.nan,
                               'dgc_neg_peak_z': np.nan,
                               'dgc_pos_p_ttest_z': np.nan,
                               'dgc_neg_p_ttest_z': np.nan,
                               'dgc_p_anova_z': np.nan,

                               'dgc_pos_osi_raw_df': np.nan,
                               'dgc_pos_dsi_raw_df': np.nan,
                               'dgc_pos_gosi_raw_df': np.nan,
                               'dgc_pos_gdsi_raw_df': np.nan,
                               'dgc_pos_osi_ele_df': np.nan,
                               'dgc_pos_dsi_ele_df': np.nan,
                               'dgc_pos_gosi_ele_df': np.nan,
                               'dgc_pos_gdsi_ele_df': np.nan,
                               'dgc_pos_osi_rec_df': np.nan,
                               'dgc_pos_dsi_rec_df': np.nan,
                               'dgc_pos_gosi_rec_df': np.nan,
                               'dgc_pos_gdsi_rec_df': np.nan,
                               'dgc_pos_peak_dire_raw_df': np.nan,
                               'dgc_pos_vs_dire_raw_df': np.nan,
                               'dgc_pos_vs_dire_ele_df': np.nan,
                               'dgc_pos_vs_dire_rec_df': np.nan,
                               'dgc_neg_osi_raw_df': np.nan,
                               'dgc_neg_dsi_raw_df': np.nan,
                               'dgc_neg_gosi_raw_df': np.nan,
                               'dgc_neg_gdsi_raw_df': np.nan,
                               'dgc_neg_osi_ele_df': np.nan,
                               'dgc_neg_dsi_ele_df': np.nan,
                               'dgc_neg_gosi_ele_df': np.nan,
                               'dgc_neg_gdsi_ele_df': np.nan,
                               'dgc_neg_osi_rec_df': np.nan,
                               'dgc_neg_dsi_rec_df': np.nan,
                               'dgc_neg_gosi_rec_df': np.nan,
                               'dgc_neg_gdsi_rec_df': np.nan,
                               'dgc_neg_peak_dire_raw_df': np.nan,
                               'dgc_neg_vs_dire_raw_df': np.nan,
                               'dgc_neg_vs_dire_ele_df': np.nan,
                               'dgc_neg_vs_dire_rec_df': np.nan,
                               'dgc_pos_peak_sf_raw_df': np.nan,
                               'dgc_pos_weighted_sf_raw_df': np.nan,
                               'dgc_pos_weighted_sf_log_raw_df': np.nan,
                               'dgc_pos_weighted_sf_ele_df': np.nan,
                               'dgc_pos_weighted_sf_log_ele_df': np.nan,
                               'dgc_pos_weighted_sf_rec_df': np.nan,
                               'dgc_pos_weighted_sf_log_rec_df': np.nan,
                               'dgc_neg_peak_sf_raw_df': np.nan,
                               'dgc_neg_weighted_sf_raw_df': np.nan,
                               'dgc_neg_weighted_sf_log_raw_df': np.nan,
                               'dgc_neg_weighted_sf_ele_df': np.nan,
                               'dgc_neg_weighted_sf_log_ele_df': np.nan,
                               'dgc_neg_weighted_sf_rec_df': np.nan,
                               'dgc_neg_weighted_sf_log_rec_df': np.nan,
                               'dgc_pos_peak_tf_raw_df': np.nan,
                               'dgc_pos_weighted_tf_raw_df': np.nan,
                               'dgc_pos_weighted_tf_log_raw_df': np.nan,
                               'dgc_pos_weighted_tf_ele_df': np.nan,
                               'dgc_pos_weighted_tf_log_ele_df': np.nan,
                               'dgc_pos_weighted_tf_rec_df': np.nan,
                               'dgc_pos_weighted_tf_log_rec_df': np.nan,
                               'dgc_neg_peak_tf_raw_df': np.nan,
                               'dgc_neg_weighted_tf_raw_df': np.nan,
                               'dgc_neg_weighted_tf_log_raw_df': np.nan,
                               'dgc_neg_weighted_tf_ele_df': np.nan,
                               'dgc_neg_weighted_tf_log_ele_df': np.nan,
                               'dgc_neg_weighted_tf_rec_df': np.nan,
                               'dgc_neg_weighted_tf_log_rec_df': np.nan,

                               'dgc_pos_osi_raw_dff': np.nan,
                               'dgc_pos_dsi_raw_dff': np.nan,
                               'dgc_pos_gosi_raw_dff': np.nan,
                               'dgc_pos_gdsi_raw_dff': np.nan,
                               'dgc_pos_osi_ele_dff': np.nan,
                               'dgc_pos_dsi_ele_dff': np.nan,
                               'dgc_pos_gosi_ele_dff': np.nan,
                               'dgc_pos_gdsi_ele_dff': np.nan,
                               'dgc_pos_osi_rec_dff': np.nan,
                               'dgc_pos_dsi_rec_dff': np.nan,
                               'dgc_pos_gosi_rec_dff': np.nan,
                               'dgc_pos_gdsi_rec_dff': np.nan,
                               'dgc_pos_peak_dire_raw_dff': np.nan,
                               'dgc_pos_vs_dire_raw_dff': np.nan,
                               'dgc_pos_vs_dire_ele_dff': np.nan,
                               'dgc_pos_vs_dire_rec_dff': np.nan,
                               'dgc_neg_osi_raw_dff': np.nan,
                               'dgc_neg_dsi_raw_dff': np.nan,
                               'dgc_neg_gosi_raw_dff': np.nan,
                               'dgc_neg_gdsi_raw_dff': np.nan,
                               'dgc_neg_osi_ele_dff': np.nan,
                               'dgc_neg_dsi_ele_dff': np.nan,
                               'dgc_neg_gosi_ele_dff': np.nan,
                               'dgc_neg_gdsi_ele_dff': np.nan,
                               'dgc_neg_osi_rec_dff': np.nan,
                               'dgc_neg_dsi_rec_dff': np.nan,
                               'dgc_neg_gosi_rec_dff': np.nan,
                               'dgc_neg_gdsi_rec_dff': np.nan,
                               'dgc_neg_peak_dire_raw_dff': np.nan,
                               'dgc_neg_vs_dire_raw_dff': np.nan,
                               'dgc_neg_vs_dire_ele_dff': np.nan,
                               'dgc_neg_vs_dire_rec_dff': np.nan,
                               'dgc_pos_peak_sf_raw_dff': np.nan,
                               'dgc_pos_weighted_sf_raw_dff': np.nan,
                               'dgc_pos_weighted_sf_log_raw_dff': np.nan,
                               'dgc_pos_weighted_sf_ele_dff': np.nan,
                               'dgc_pos_weighted_sf_log_ele_dff': np.nan,
                               'dgc_pos_weighted_sf_rec_dff': np.nan,
                               'dgc_pos_weighted_sf_log_rec_dff': np.nan,
                               'dgc_neg_peak_sf_raw_dff': np.nan,
                               'dgc_neg_weighted_sf_raw_dff': np.nan,
                               'dgc_neg_weighted_sf_log_raw_dff': np.nan,
                               'dgc_neg_weighted_sf_ele_dff': np.nan,
                               'dgc_neg_weighted_sf_log_ele_dff': np.nan,
                               'dgc_neg_weighted_sf_rec_dff': np.nan,
                               'dgc_neg_weighted_sf_log_rec_dff': np.nan,
                               'dgc_pos_peak_tf_raw_dff': np.nan,
                               'dgc_pos_weighted_tf_raw_dff': np.nan,
                               'dgc_pos_weighted_tf_log_raw_dff': np.nan,
                               'dgc_pos_weighted_tf_ele_dff': np.nan,
                               'dgc_pos_weighted_tf_log_ele_dff': np.nan,
                               'dgc_pos_weighted_tf_rec_dff': np.nan,
                               'dgc_pos_weighted_tf_log_rec_dff': np.nan,
                               'dgc_neg_peak_tf_raw_dff': np.nan,
                               'dgc_neg_weighted_tf_raw_dff': np.nan,
                               'dgc_neg_weighted_tf_log_raw_dff': np.nan,
                               'dgc_neg_weighted_tf_ele_dff': np.nan,
                               'dgc_neg_weighted_tf_log_ele_dff': np.nan,
                               'dgc_neg_weighted_tf_rec_dff': np.nan,
                               'dgc_neg_weighted_tf_log_rec_dff': np.nan,

                               'dgc_pos_osi_raw_z': np.nan,
                               'dgc_pos_dsi_raw_z': np.nan,
                               'dgc_pos_gosi_raw_z': np.nan,
                               'dgc_pos_gdsi_raw_z': np.nan,
                               'dgc_pos_osi_ele_z': np.nan,
                               'dgc_pos_dsi_ele_z': np.nan,
                               'dgc_pos_gosi_ele_z': np.nan,
                               'dgc_pos_gdsi_ele_z': np.nan,
                               'dgc_pos_osi_rec_z': np.nan,
                               'dgc_pos_dsi_rec_z': np.nan,
                               'dgc_pos_gosi_rec_z': np.nan,
                               'dgc_pos_gdsi_rec_z': np.nan,
                               'dgc_pos_peak_dire_raw_z': np.nan,
                               'dgc_pos_vs_dire_raw_z': np.nan,
                               'dgc_pos_vs_dire_ele_z': np.nan,
                               'dgc_pos_vs_dire_rec_z': np.nan,
                               'dgc_neg_osi_raw_z': np.nan,
                               'dgc_neg_dsi_raw_z': np.nan,
                               'dgc_neg_gosi_raw_z': np.nan,
                               'dgc_neg_gdsi_raw_z': np.nan,
                               'dgc_neg_osi_ele_z': np.nan,
                               'dgc_neg_dsi_ele_z': np.nan,
                               'dgc_neg_gosi_ele_z': np.nan,
                               'dgc_neg_gdsi_ele_z': np.nan,
                               'dgc_neg_osi_rec_z': np.nan,
                               'dgc_neg_dsi_rec_z': np.nan,
                               'dgc_neg_gosi_rec_z': np.nan,
                               'dgc_neg_gdsi_rec_z': np.nan,
                               'dgc_neg_peak_dire_raw_z': np.nan,
                               'dgc_neg_vs_dire_raw_z': np.nan,
                               'dgc_neg_vs_dire_ele_z': np.nan,
                               'dgc_neg_vs_dire_rec_z': np.nan,
                               'dgc_pos_peak_sf_raw_z': np.nan,
                               'dgc_pos_weighted_sf_raw_z': np.nan,
                               'dgc_pos_weighted_sf_log_raw_z': np.nan,
                               'dgc_pos_weighted_sf_ele_z': np.nan,
                               'dgc_pos_weighted_sf_log_ele_z': np.nan,
                               'dgc_pos_weighted_sf_rec_z': np.nan,
                               'dgc_pos_weighted_sf_log_rec_z': np.nan,
                               'dgc_neg_peak_sf_raw_z': np.nan,
                               'dgc_neg_weighted_sf_raw_z': np.nan,
                               'dgc_neg_weighted_sf_log_raw_z': np.nan,
                               'dgc_neg_weighted_sf_ele_z': np.nan,
                               'dgc_neg_weighted_sf_log_ele_z': np.nan,
                               'dgc_neg_weighted_sf_rec_z': np.nan,
                               'dgc_neg_weighted_sf_log_rec_z': np.nan,
                               'dgc_pos_peak_tf_raw_z': np.nan,
                               'dgc_pos_weighted_tf_raw_z': np.nan,
                               'dgc_pos_weighted_tf_log_raw_z': np.nan,
                               'dgc_pos_weighted_tf_ele_z': np.nan,
                               'dgc_pos_weighted_tf_log_ele_z': np.nan,
                               'dgc_pos_weighted_tf_rec_z': np.nan,
                               'dgc_pos_weighted_tf_log_rec_z': np.nan,
                               'dgc_neg_peak_tf_raw_z': np.nan,
                               'dgc_neg_weighted_tf_raw_z': np.nan,
                               'dgc_neg_weighted_tf_log_raw_z': np.nan,
                               'dgc_neg_weighted_tf_ele_z': np.nan,
                               'dgc_neg_weighted_tf_log_ele_z': np.nan,
                               'dgc_neg_weighted_tf_rec_z': np.nan,
                               'dgc_neg_weighted_tf_log_rec_z': np.nan,
                               })

    if verbose:
        # max_len_key = max([len(k) for k in roi_properties.keys()])
        # print max_len_key
        print('\n'.join(['{:>31}:  {}'.format(k, v) for k, v in roi_properties.items()]))

    return roi_properties, roi, trace, srf_pos_on, srf_pos_off, srf_neg_on, srf_neg_off, dgcrm_df, dgcrm_dff, \
           dgcrm_z, dgcrt_df, dgcrt_dff, dgcrt_z, dgc_block_dur


def get_axon_ind_from_clu_f(clu_f, axon_n):
    """
    based on the axon name return the index of that axon in the clustering result file for extracting
    traces, strf and dgcrm

    because the clustering result file only contains traces, strfs, dgcrms for axons with more than
    one rois. the function will return None for axons not in the clustering result file or axons
    with only one roi.

    :param clu_f:
    :param axon_n:
    :return:
    """

    if axon_n in clu_f['axons'].keys():
        roi_lst = clu_f['axons/{}'.format(axon_n)][()]
        if len(roi_lst) == 1:
            print('\tThere is only one roi in the axon ({}). Returning None.'.format(axon_n))
            return None
        else:
            # print('\tThere are {} rois in the axon ({}).'.format(len(roi_lst), axon_n))
            axon_ind = list(clu_f['rois_and_traces/axon_list']).index(axon_n)
            return axon_ind
    else:
        print('\taxon ({}) not in the clu_f file. Returning None.'.format(axon_n))
        return None


def get_axon_roi_from_clu_f(clu_f, axon_n):

    axon_ind = get_axon_ind_from_clu_f(clu_f=clu_f, axon_n=axon_n)

    if axon_ind is None:
        return None
    else:
        mask = clu_f['rois_and_traces/masks_center'][axon_ind, :, :]
        return ia.WeightedROI(mask)


def get_axon_trace_from_clu_f(clu_f, axon_n, trace_type):

    axon_ind = get_axon_ind_from_clu_f(clu_f=clu_f, axon_n=axon_n)

    if axon_ind is None:
        return None
    else:
        return clu_f['rois_and_traces/{}'.format(trace_type)][axon_ind, :]


def get_axon_strf_from_clu_f(clu_f, plane_n, axon_n, trace_type,
                             location_unit='degree'):

    strf_grp = [k for k in clu_f.keys() if len(k) > 5 and k[0:5] == 'strf_']
    if len(strf_grp) == 0:
        print('\tDid not find strf in "clu_f". Returning None.')
        return
    elif len(strf_grp) > 1:
        raise ValueError('More than one strf groups found in "clu_f".')

    strf_grp = clu_f['{}/{}'.format(strf_grp[0], plane_n)]

    axon_ind = get_axon_ind_from_clu_f(clu_f=clu_f, axon_n=axon_n)

    if axon_ind is None:
        return None
    else:
        strf = sca.get_strf_from_nwb(h5_grp=strf_grp, roi_ind=axon_ind, trace_type=trace_type,
                                     location_unit=location_unit)
        strf.name = axon_n
        return strf


def get_axon_dgcrm_from_clu_f(clu_f, plane_n, axon_n, trace_type):

    dgcrm_grp = [k for k in clu_f.keys() if len(k) > 15 and k[0:15] == 'response_table_']
    if len(dgcrm_grp) == 0:
        print('\tDid not find dgcrm in "clu_f". Returning None.')
        return
    elif len(dgcrm_grp) > 1:
        raise ValueError('More than one dgcrm groups found in "clu_f".')

    dgcrm_grp = clu_f['{}/{}'.format(dgcrm_grp[0], plane_n)]

    axon_ind = get_axon_ind_from_clu_f(clu_f=clu_f, axon_n=axon_n)

    if axon_ind is None:
        return None
    else:
        dgcrm = sca.get_dgc_response_matrix_from_nwb(h5_grp=dgcrm_grp,
                                                     roi_ind=axon_ind,
                                                     trace_type=trace_type)
        return dgcrm


def get_axon_morphology(clu_f, nwb_f, plane_n, axon_n):

    axon_morph = {}

    mc_grp = nwb_f['processing/motion_correction/MotionCorrection/{}/corrected'.format(plane_n)]
    pixel_size = mc_grp['pixel_size'][()]
    # print(pixel_size)
    pixel_size_mean = np.mean(pixel_size)

    bout_ns = clu_f['axons/{}'.format(axon_n)][()]
    # print(bout_ns)
    bout_num = len(bout_ns)
    axon_morph['bouton_num'] = bout_num

    if bout_num == 1:
        axon_roi = get_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=bout_ns[0])
    else:
        axon_roi = get_axon_roi_from_clu_f(clu_f=clu_f, axon_n=axon_n)
        axon_roi = ia.WeightedROI(axon_roi.get_weighted_mask(), pixelSize=pixel_size,
                                  pixelSizeUnit=mc_grp['pixel_size_unit'][()])

    # plt.imshow(axon_roi.get_binary_mask(), interpolation='nearest')
    # plt.show()

    axon_morph['axon_row_range'] = (np.max(axon_roi.pixels[0]) -
                                    np.min(axon_roi.pixels[0])) * pixel_size[0] * 1e6
    axon_morph['axon_col_range'] = (np.max(axon_roi.pixels[1]) -
                                    np.min(axon_roi.pixels[1])) * pixel_size[1] * 1e6

    axon_morph['axon_area'] = axon_roi.get_pixel_area() * 1e12

    axon_qhull = spatial.ConvexHull(np.array(axon_roi.pixels).transpose())
    # print(axon_qhull.volume)
    axon_morph['axon_qhull_area'] = axon_qhull.volume * axon_roi.pixelSizeX * axon_roi.pixelSizeY * 1e12

    bout_rois = []
    for bout_n in bout_ns:
        bout_rois.append(get_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=bout_n))

    bout_areas = [r.get_pixel_area() for r in bout_rois]
    bout_area_mean = np.mean(bout_areas)
    axon_morph['bouton_area_mean'] = bout_area_mean * 1e12

    if bout_num == 1:
        bout_area_std = np.nan
    else:
        bout_area_std = np.std(bout_areas)
    axon_morph['bouton_area_std'] = bout_area_std * 1e12

    bout_coords = np.array([r.get_center() for r in bout_rois]) # [[y0, x0], [y1, x1], ... , [yn, xn]]
    if bout_num == 1:
        axon_morph['bouton_row_std'] = np.nan
        axon_morph['bouton_col_std'] = np.nan
        axon_morph['bouton_dis_mean'] = np.nan
        axon_morph['bouton_dis_std'] = np.nan
        axon_morph['bouton_dis_median'] = np.nan
        axon_morph['bouton_dis_max'] = np.nan
    else:
        axon_morph['bouton_row_std'] = np.std(bout_coords[:, 0]) * pixel_size_mean * 1e6
        axon_morph['bouton_col_std'] = np.std(bout_coords[:, 1]) * pixel_size_mean * 1e6

        bout_dis = spatial.distance.pdist(bout_coords) * pixel_size_mean
        axon_morph['bouton_dis_mean'] = np.mean(bout_dis) * 1e6
        axon_morph['bouton_dis_median'] = np.median(bout_dis) * 1e6
        axon_morph['bouton_dis_max'] = np.max(bout_dis) * 1e6

        if bout_num == 2:
            axon_morph['bouton_dis_std'] = np.nan
        else:
            axon_morph['bouton_dis_std'] = np.std(bout_dis) * 1e6

    return axon_morph


def get_axon_roi(clu_f, nwb_f, plane_n, axon_n):
    """
    get axon roi as corticalmapping.core.ImageAnalysis.WeightedROI object

    :param clu_f: hdf5.File
    :param nwb_f: hdf5.File
    :param plane_n: str
    :param axon_n: str
    :return axon_roi: corticalmapping.core.ImageAnalysis.WeightedROI object
    """

    pixel_size_s = nwb_f['acquisition/timeseries/2p_movie_{}/pixel_size'.format(plane_n)][()]
    pixel_size_u_s = nwb_f['acquisition/timeseries/2p_movie_{}/pixel_size_unit'.format(plane_n)][()]

    if axon_n in clu_f['rois_and_traces/axon_list'][()]:
        axon_roi = get_axon_roi_from_clu_f(clu_f=clu_f, axon_n=axon_n)
        axon_roi.pixelSizeX = pixel_size_s[1]
        axon_roi.pixelSizeY = pixel_size_s[0]
        axon_roi.pixelSizeUnit = pixel_size_u_s
    else:
        roi_n = clu_f['axons/{}'.format(axon_n)][()]
        if len(roi_n) == 0 :
            raise ValueError('Did not find bouton rois for this axon: {} / {}'.format(clu_f.filename, axon_n))
        elif len(roi_n) > 1:
            raise ValueError('More than one bouton rois found in this axon: {} /{}'.format(clu_f.filename, axon_n))
        else:
            roi_n = roi_n[0]

        axon_roi = get_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)

    return axon_roi


def get_everything_from_axon(nwb_f, clu_f, plane_n, axon_n, params=ANALYSIS_PARAMS, verbose=False):
    """

    :param nwb_f:
    :param clu_f:
    :param roidf:
    :param plane_n:
    :param axon_n:
    :param params:
    :return:
    """

    date = nwb_f['identifier'][()][0:6]
    if clu_f['meta/date'][()] != date:
        raise ValueError('the date ({}) specified in nwb_f does not match the date ({}) specified in '
                         '"clu_f".'.format(date, clu_f['meta/date'][()]))

    mid = nwb_f['identifier'][()][7:14]
    if clu_f['meta/mouse_id'][()] != mid:
        raise ValueError('the mouse_id ({}) specified in nwb_f does not match the mouse_id ({}) '
                         'specified in clu_f.'.format(plane_n, clu_f['meta/mouse_id'][()]))

    if clu_f['meta/plane_n'][()] != plane_n:
        raise ValueError('the input "plane_n" ({}) does not match the plane_n ({}) specified in '
                         '"clu_f".'.format(plane_n, clu_f['meta/plane_n'][()]))

    roi_lst = clu_f['axons/{}'.format(axon_n)][()]
    if len(roi_lst) == 1:
        roi_n = roi_lst[0]
        print('\tThere is only one roi ({}) in the axon ({}).'.format(roi_n, axon_n))
        return get_everything_from_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n,
                                       params=params)
    else:
        print('\tThere are {} rois in the axon ({}).'.format(len(roi_lst), axon_n))

        axon_properties = {'date': date,
                           'mouse_id': mid,
                           'plane_n': plane_n,
                           'roi_n': axon_n,
                           'depth': nwb_f['processing/rois_and_traces_{}/imaging_depth_micron'.format(plane_n)][()]}

        # get mask properties
        axon_roi = get_axon_roi_from_clu_f(clu_f=clu_f, axon_n=axon_n)
        pixel_size = nwb_f['acquisition/timeseries/2p_movie_{}/pixel_size'.format(plane_n)][()] * 1000000.
        roi_area = axon_roi.get_binary_area() * pixel_size[0] * pixel_size[1]
        roi_center_row, roi_center_col = axon_roi.get_weighted_center()
        axon_properties.update({'roi_area': roi_area,
                               'roi_center_row': roi_center_row,
                               'roi_center_col': roi_center_col})

        #get skewness
        tt = params['trace_type'].replace('f', 'traces')
        trace = get_axon_trace_from_clu_f(clu_f=clu_f, axon_n=axon_n, trace_type=tt)
        trace_ts = nwb_f['processing/rois_and_traces_{}/' \
                         'Fluorescence/{}/timestamps'.format(plane_n, params['trace_type'])][()]
        skew_raw, skew_fil = sca.get_skewness(trace=trace, ts=trace_ts,
                                              filter_length=params['filter_length_skew_sec'])
        axon_properties.update({'skew_raw': skew_raw,
                               'skew_fil': skew_fil})

        if np.min(trace) < params['trace_abs_minimum']:
            add_to_trace = -np.min(trace) + params['trace_abs_minimum']
        else:
            add_to_trace = 0.


        # get strf properties
        tt = params['trace_type'].replace('f', 'sta_traces')
        strf = get_axon_strf_from_clu_f(clu_f=clu_f, plane_n=plane_n, axon_n=axon_n,
                                        trace_type=tt, location_unit='degree')

        if strf is not None:

            # get strf properties
            strf_dff = strf.get_local_dff_strf(is_collaps_before_normalize=True, add_to_trace=add_to_trace)

            # positive spatial receptive fields
            srf_pos_on, srf_pos_off = strf_dff.get_zscore_receptive_field(timeWindow=params['response_window_positive_rf'])

            # # get filter sigma in pixels
            # mean_probe_size = (np.abs(np.mean(np.diff(srf_pos_on.altPos))) +
            #                   np.abs(np.mean(np.diff(srf_pos_on.aziPos)))) / 2.
            # print(mean_probe_size)
            # sigma = params['gaussian_filter_sigma_rf'] / mean_probe_size
            # print(sigma)

            # ON positive spatial receptive field
            rf_pos_on_z, rf_pos_on_new = get_rf_properties(srf=srf_pos_on,
                                                           polarity='positive',
                                                           sigma=params['gaussian_filter_sigma_rf'],
                                                           interpolate_rate=params['interpolate_rate_rf'],
                                                           z_thr_abs=params['rf_z_thr_abs'],
                                                           z_thr_rel=params['rf_z_thr_rel'])
            rf_pos_on_area = rf_pos_on_new.get_binary_rf_area()
            rf_pos_on_center = rf_pos_on_new.get_weighted_rf_center()

            axon_properties.update({'rf_pos_on_peak_z': rf_pos_on_z,
                                   'rf_pos_on_area': rf_pos_on_area,
                                   'rf_pos_on_center_alt': rf_pos_on_center[0],
                                   'rf_pos_on_center_azi': rf_pos_on_center[1]})

            # OFF positive spatial receptive field
            rf_pos_off_z, rf_pos_off_new = get_rf_properties(srf=srf_pos_off,
                                                             polarity='positive',
                                                             sigma=params['gaussian_filter_sigma_rf'],
                                                             interpolate_rate=params['interpolate_rate_rf'],
                                                             z_thr_abs=params['rf_z_thr_abs'],
                                                             z_thr_rel=params['rf_z_thr_rel'])

            rf_pos_off_area = rf_pos_off_new.get_binary_rf_area()
            rf_pos_off_center = rf_pos_off_new.get_weighted_rf_center()

            axon_properties.update({'rf_pos_off_peak_z': rf_pos_off_z,
                                   'rf_pos_off_area': rf_pos_off_area,
                                   'rf_pos_off_center_alt': rf_pos_off_center[0],
                                   'rf_pos_off_center_azi': rf_pos_off_center[1]})

            # on off overlapping
            rf_pos_on_mask = rf_pos_on_new.get_weighted_mask()
            rf_pos_off_mask = rf_pos_off_new.get_weighted_mask()
            rf_pos_lsi = sca.get_local_similarity_index(rf_pos_on_mask, rf_pos_off_mask)

            rf_pos_onoff_new = sca.SpatialReceptiveField(mask=np.max([rf_pos_on_mask, rf_pos_off_mask], axis=0),
                                                         altPos=rf_pos_on_new.altPos,
                                                         aziPos=rf_pos_on_new.aziPos,
                                                         sign='ON_OFF',
                                                         thr=params['rf_z_thr_abs'])
            if len(rf_pos_onoff_new.weights) == 0:
                rf_pos_onoff_z = np.nan
            else:
                rf_pos_onoff_z = np.max(rf_pos_onoff_new.weights)
            rf_pos_onoff_area = rf_pos_onoff_new.get_binary_rf_area()
            rf_pos_onoff_center = rf_pos_onoff_new.get_weighted_rf_center()
            axon_properties.update({'rf_pos_lsi': rf_pos_lsi,
                                   'rf_pos_onoff_peak_z': rf_pos_onoff_z,
                                   'rf_pos_onoff_area': rf_pos_onoff_area,
                                   'rf_pos_onoff_center_alt': rf_pos_onoff_center[0],
                                   'rf_pos_onoff_center_azi': rf_pos_onoff_center[1]})

            # negative spatial receptive fields
            srf_neg_on, srf_neg_off = strf_dff.get_zscore_receptive_field(timeWindow=params['response_window_negative_rf'])

            # ON negative spatial receptive field
            rf_neg_on_z, rf_neg_on_new = get_rf_properties(srf=srf_neg_on,
                                                           polarity='negative',
                                                           sigma=params['gaussian_filter_sigma_rf'],
                                                           interpolate_rate=params['interpolate_rate_rf'],
                                                           z_thr_abs=params['rf_z_thr_abs'],
                                                           z_thr_rel=params['rf_z_thr_rel'])
            rf_neg_on_area = rf_neg_on_new.get_binary_rf_area()
            rf_neg_on_center = rf_neg_on_new.get_weighted_rf_center()
            axon_properties.update({'rf_neg_on_peak_z': rf_neg_on_z,
                                   'rf_neg_on_area': rf_neg_on_area,
                                   'rf_neg_on_center_alt': rf_neg_on_center[0],
                                   'rf_neg_on_center_azi': rf_neg_on_center[1]})

            # OFF negative spatial receptive field
            rf_neg_off_z, rf_neg_off_new = get_rf_properties(srf=srf_neg_off,
                                                             polarity='negative',
                                                             sigma=params['gaussian_filter_sigma_rf'],
                                                             interpolate_rate=params['interpolate_rate_rf'],
                                                             z_thr_abs=params['rf_z_thr_abs'],
                                                             z_thr_rel=params['rf_z_thr_rel'])
            rf_neg_off_area = rf_neg_off_new.get_binary_rf_area()
            rf_neg_off_center = rf_neg_off_new.get_weighted_rf_center()
            axon_properties.update({'rf_neg_off_peak_z': rf_neg_off_z,
                                   'rf_neg_off_area': rf_neg_off_area,
                                   'rf_neg_off_center_alt': rf_neg_off_center[0],
                                   'rf_neg_off_center_azi': rf_neg_off_center[1]})

            # on off overlapping
            rf_neg_on_mask = rf_neg_on_new.get_weighted_mask()
            rf_neg_off_mask = rf_neg_off_new.get_weighted_mask()
            rf_neg_lsi = sca.get_local_similarity_index(rf_neg_on_mask, rf_neg_off_mask)

            rf_neg_onoff_new = sca.SpatialReceptiveField(mask=np.max([rf_neg_on_mask, rf_neg_off_mask], axis=0),
                                                         altPos=rf_neg_on_new.altPos,
                                                         aziPos=rf_neg_on_new.aziPos,
                                                         sign='ON_OFF',
                                                         thr=params['rf_z_thr_abs'])
            if len(rf_neg_onoff_new.weights) == 0:
                rf_neg_onoff_z = np.nan
            else:
                rf_neg_onoff_z = np.max(rf_neg_onoff_new.weights)
            rf_neg_onoff_area = rf_neg_onoff_new.get_binary_rf_area()
            rf_neg_onoff_center = rf_neg_onoff_new.get_weighted_rf_center()
            axon_properties.update({'rf_neg_onoff_peak_z': rf_neg_onoff_z,
                                   'rf_neg_onoff_area': rf_neg_onoff_area,
                                   'rf_neg_onoff_center_alt': rf_neg_onoff_center[0],
                                   'rf_neg_onoff_center_azi': rf_neg_onoff_center[1],
                                   'rf_neg_lsi': rf_neg_lsi})
        else:
            srf_pos_on = None
            srf_pos_off = None
            srf_neg_on = None
            srf_neg_off = None

            axon_properties.update({'rf_pos_on_peak_z': np.nan,
                                   'rf_pos_on_area': np.nan,
                                   'rf_pos_on_center_alt': np.nan,
                                   'rf_pos_on_center_azi': np.nan,
                                   'rf_pos_off_peak_z': np.nan,
                                   'rf_pos_off_area': np.nan,
                                   'rf_pos_off_center_alt': np.nan,
                                   'rf_pos_off_center_azi': np.nan,
                                   'rf_pos_onoff_peak_z': np.nan,
                                   'rf_pos_onoff_area': np.nan,
                                   'rf_pos_onoff_center_alt': np.nan,
                                   'rf_pos_onoff_center_azi': np.nan,
                                   'rf_pos_lsi': np.nan,
                                   'rf_neg_on_peak_z': np.nan,
                                   'rf_neg_on_area': np.nan,
                                   'rf_neg_on_center_alt': np.nan,
                                   'rf_neg_on_center_azi': np.nan,
                                   'rf_neg_off_peak_z': np.nan,
                                   'rf_neg_off_area': np.nan,
                                   'rf_neg_off_center_alt': np.nan,
                                   'rf_neg_off_center_azi': np.nan,
                                   'rf_neg_onoff_peak_z': np.nan,
                                   'rf_neg_onoff_area': np.nan,
                                   'rf_neg_onoff_center_alt': np.nan,
                                   'rf_neg_onoff_center_azi': np.nan,
                                   'rf_neg_lsi': np.nan,
                                   })

        # analyze response to drifring grating
        tt = params['trace_type'].replace('f', 'sta_traces')
        dgcrm = get_axon_dgcrm_from_clu_f(clu_f=clu_f, plane_n=plane_n, axon_n=axon_n,
                                          trace_type=tt)

        if dgcrm is not None:
            dgcrm_grp_key = get_dgcrm_grp_key(nwb_f=nwb_f)
            dgc_block_dur = nwb_f['stimulus/presentation/{}/block_dur'.format(dgcrm_grp_key[15:])][()]
            # print('block duration: {}'.format(block_dur))

            # get df statistics ============================================================================================
            _ = dgcrm.get_df_response_table(baseline_win=params['baseline_window_dgc'],
                                            response_win=params['response_window_dgc'])
            dgcrt_df, dgc_p_anova_df, dgc_pos_p_ttest_df, dgc_neg_p_ttest_df = _
            axon_properties.update({'dgc_pos_peak_df': dgcrt_df.peak_response_pos,
                                   'dgc_neg_peak_df': dgcrt_df.peak_response_neg,
                                   'dgc_pos_p_ttest_df': dgc_pos_p_ttest_df,
                                   'dgc_neg_p_ttest_df': dgc_neg_p_ttest_df,
                                   'dgc_p_anova_df': dgc_p_anova_df})

            # get dff statics ==============================================================================================
            _ = dgcrm.get_dff_response_table(baseline_win=params['baseline_window_dgc'],
                                             response_win=params['response_window_dgc'],
                                             bias=add_to_trace)
            dgcrt_dff, dgc_p_anova_dff, dgc_pos_p_ttest_dff, dgc_neg_p_ttest_dff = _
            axon_properties.update({'dgc_pos_peak_dff': dgcrt_dff.peak_response_pos,
                                   'dgc_neg_peak_dff': dgcrt_dff.peak_response_neg,
                                   'dgc_pos_p_ttest_dff': dgc_pos_p_ttest_dff,
                                   'dgc_neg_p_ttest_dff': dgc_neg_p_ttest_dff,
                                   'dgc_p_anova_dff': dgc_p_anova_dff})

            # get zscore statistics ========================================================================================
            _ = dgcrm.get_zscore_response_table(baseline_win=params['baseline_window_dgc'],
                                                response_win=params['response_window_dgc'])
            dgcrt_z, dgc_p_anova_z, dgc_pos_p_ttest_z, dgc_neg_p_ttest_z = _
            axon_properties.update({'dgc_pos_peak_z': dgcrt_z.peak_response_pos,
                                   'dgc_neg_peak_z': dgcrt_z.peak_response_neg,
                                   'dgc_pos_p_ttest_z': dgc_pos_p_ttest_z,
                                   'dgc_neg_p_ttest_z': dgc_neg_p_ttest_z,
                                   'dgc_p_anova_z': dgc_p_anova_z})

            # get dgc response matrices ====================================================================================
            dgcrm_df = dgcrm.get_df_response_matrix(baseline_win=params['baseline_window_dgc'])
            dgcrm_dff = dgcrm.get_dff_response_matrix(baseline_win=params['baseline_window_dgc'],
                                                      bias=add_to_trace)
            dgcrm_z = dgcrm.get_zscore_response_matrix(baseline_win=params['baseline_window_dgc'])

            # direction/orientation tuning of df responses in positive direction ===========================================
            dire_tuning_df_pos = dgcrt_df.get_dire_tuning(response_dir='pos',
                                                          is_collapse_sf=params['is_collapse_sf'],
                                                          is_collapse_tf=params['is_collapse_tf'])
            osi_df_pos_raw, dsi_df_pos_raw, gosi_df_pos_raw, gdsi_df_pos_raw, \
            osi_df_pos_ele, dsi_df_pos_ele, gosi_df_pos_ele, gdsi_df_pos_ele, \
            osi_df_pos_rec, dsi_df_pos_rec, gosi_df_pos_rec, gdsi_df_pos_rec, \
            peak_dire_raw_df_pos, vs_dire_raw_df_pos, vs_dire_ele_df_pos, vs_dire_rec_df_pos \
                = dgcrt_df.get_dire_tuning_properties(dire_tuning_df_pos,
                                                      response_dir='pos',
                                                      elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_pos_osi_raw_df': osi_df_pos_raw,
                                   'dgc_pos_dsi_raw_df': dsi_df_pos_raw,
                                   'dgc_pos_gosi_raw_df': gosi_df_pos_raw,
                                   'dgc_pos_gdsi_raw_df': gdsi_df_pos_raw,
                                   'dgc_pos_osi_ele_df': osi_df_pos_ele,
                                   'dgc_pos_dsi_ele_df': dsi_df_pos_ele,
                                   'dgc_pos_gosi_ele_df': gosi_df_pos_ele,
                                   'dgc_pos_gdsi_ele_df': gdsi_df_pos_ele,
                                   'dgc_pos_osi_rec_df': osi_df_pos_rec,
                                   'dgc_pos_dsi_rec_df': dsi_df_pos_rec,
                                   'dgc_pos_gosi_rec_df': gosi_df_pos_rec,
                                   'dgc_pos_gdsi_rec_df': gdsi_df_pos_rec,
                                   'dgc_pos_peak_dire_raw_df': peak_dire_raw_df_pos,
                                   'dgc_pos_vs_dire_raw_df': vs_dire_raw_df_pos,
                                   'dgc_pos_vs_dire_ele_df': vs_dire_ele_df_pos,
                                   'dgc_pos_vs_dire_rec_df': vs_dire_rec_df_pos})

            # direction/orientation tuning of df responses in negative direction ===========================================
            dire_tuning_df_neg = dgcrt_df.get_dire_tuning(response_dir='neg',
                                                          is_collapse_sf=params['is_collapse_sf'],
                                                          is_collapse_tf=params['is_collapse_tf'])
            osi_df_neg_raw, dsi_df_neg_raw, gosi_df_neg_raw, gdsi_df_neg_raw, \
            osi_df_neg_ele, dsi_df_neg_ele, gosi_df_neg_ele, gdsi_df_neg_ele, \
            osi_df_neg_rec, dsi_df_neg_rec, gosi_df_neg_rec, gdsi_df_neg_rec, \
            peak_dire_raw_df_neg, vs_dire_raw_df_neg, vs_dire_ele_df_neg, vs_dire_rec_df_neg \
                = dgcrt_df.get_dire_tuning_properties(dire_tuning_df_neg,
                                                      response_dir='neg',
                                                      elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_neg_osi_raw_df': osi_df_neg_raw,
                                   'dgc_neg_dsi_raw_df': dsi_df_neg_raw,
                                   'dgc_neg_gosi_raw_df': gosi_df_neg_raw,
                                   'dgc_neg_gdsi_raw_df': gdsi_df_neg_raw,
                                   'dgc_neg_osi_ele_df': osi_df_neg_ele,
                                   'dgc_neg_dsi_ele_df': dsi_df_neg_ele,
                                   'dgc_neg_gosi_ele_df': gosi_df_neg_ele,
                                   'dgc_neg_gdsi_ele_df': gdsi_df_neg_ele,
                                   'dgc_neg_osi_rec_df': osi_df_neg_rec,
                                   'dgc_neg_dsi_rec_df': dsi_df_neg_rec,
                                   'dgc_neg_gosi_rec_df': gosi_df_neg_rec,
                                   'dgc_neg_gdsi_rec_df': gdsi_df_neg_rec,
                                   'dgc_neg_peak_dire_raw_df': peak_dire_raw_df_neg,
                                   'dgc_neg_vs_dire_raw_df': vs_dire_raw_df_neg,
                                   'dgc_neg_vs_dire_ele_df': vs_dire_ele_df_neg,
                                   'dgc_neg_vs_dire_rec_df': vs_dire_rec_df_neg})

            # sf tuning of df responses in positive direction ==============================================================
            sf_tuning_df_pos = dgcrt_df.get_sf_tuning(response_dir='pos', is_collapse_tf=params['is_collapse_tf'],
                                                      is_collapse_dire=params['is_collapse_dire'])
            peak_sf_raw_df_pos, weighted_sf_raw_df_pos, weighted_sf_log_raw_df_pos, \
            weighted_sf_ele_df_pos, weighted_sf_log_ele_df_pos, \
            weighted_sf_rec_df_pos, weighted_sf_log_rec_df_pos = \
                dgcrt_df.get_sf_tuning_properties(sf_tuning_df_pos, response_dir='pos',
                                                  elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_pos_peak_sf_raw_df': peak_sf_raw_df_pos,
                                   'dgc_pos_weighted_sf_raw_df': weighted_sf_raw_df_pos,
                                   'dgc_pos_weighted_sf_log_raw_df': weighted_sf_log_raw_df_pos,
                                   'dgc_pos_weighted_sf_ele_df': weighted_sf_ele_df_pos,
                                   'dgc_pos_weighted_sf_log_ele_df': weighted_sf_log_ele_df_pos,
                                   'dgc_pos_weighted_sf_rec_df': weighted_sf_rec_df_pos,
                                   'dgc_pos_weighted_sf_log_rec_df': weighted_sf_log_rec_df_pos})

            # sf tuning of df responses in negative direction ==============================================================
            sf_tuning_df_neg = dgcrt_df.get_sf_tuning(response_dir='neg', is_collapse_tf=params['is_collapse_tf'],
                                                      is_collapse_dire=params['is_collapse_dire'])
            peak_sf_raw_df_neg, weighted_sf_raw_df_neg, weighted_sf_log_raw_df_neg, \
            weighted_sf_ele_df_neg, weighted_sf_log_ele_df_neg, \
            weighted_sf_rec_df_neg, weighted_sf_log_rec_df_neg = \
                dgcrt_df.get_sf_tuning_properties(sf_tuning_df_neg, response_dir='neg',
                                                  elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_neg_peak_sf_raw_df': peak_sf_raw_df_neg,
                                   'dgc_neg_weighted_sf_raw_df': weighted_sf_raw_df_neg,
                                   'dgc_neg_weighted_sf_log_raw_df': weighted_sf_log_raw_df_neg,
                                   'dgc_neg_weighted_sf_ele_df': weighted_sf_ele_df_neg,
                                   'dgc_neg_weighted_sf_log_ele_df': weighted_sf_log_ele_df_neg,
                                   'dgc_neg_weighted_sf_rec_df': weighted_sf_rec_df_neg,
                                   'dgc_neg_weighted_sf_log_rec_df': weighted_sf_log_rec_df_neg})

            # tf tuning of df responses in positive direction ==============================================================
            tf_tuning_df_pos = dgcrt_df.get_tf_tuning(response_dir='pos', is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_dire=params['is_collapse_dire'])
            peak_tf_raw_df_pos, weighted_tf_raw_df_pos, weighted_tf_log_raw_df_pos, \
            weighted_tf_ele_df_pos, weighted_tf_log_ele_df_pos, \
            weighted_tf_rec_df_pos, weighted_tf_log_rec_df_pos = \
                dgcrt_df.get_tf_tuning_properties(tf_tuning_df_pos, response_dir='pos',
                                                  elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_pos_peak_tf_raw_df': peak_tf_raw_df_pos,
                                   'dgc_pos_weighted_tf_raw_df': weighted_tf_raw_df_pos,
                                   'dgc_pos_weighted_tf_log_raw_df': weighted_tf_log_raw_df_pos,
                                   'dgc_pos_weighted_tf_ele_df': weighted_tf_ele_df_pos,
                                   'dgc_pos_weighted_tf_log_ele_df': weighted_tf_log_ele_df_pos,
                                   'dgc_pos_weighted_tf_rec_df': weighted_tf_rec_df_pos,
                                   'dgc_pos_weighted_tf_log_rec_df': weighted_tf_log_rec_df_pos})

            # tf tuning of df responses in negative direction ==============================================================
            tf_tuning_df_neg = dgcrt_df.get_tf_tuning(response_dir='neg', is_collapse_sf=params['is_collapse_sf'],
                                                      is_collapse_dire=params['is_collapse_dire'])
            peak_tf_raw_df_neg, weighted_tf_raw_df_neg, weighted_tf_log_raw_df_neg, \
            weighted_tf_ele_df_neg, weighted_tf_log_ele_df_neg, \
            weighted_tf_rec_df_neg, weighted_tf_log_rec_df_neg = \
                dgcrt_df.get_tf_tuning_properties(tf_tuning_df_neg, response_dir='neg',
                                                  elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_neg_peak_tf_raw_df': peak_tf_raw_df_neg,
                                   'dgc_neg_weighted_tf_raw_df': weighted_tf_raw_df_neg,
                                   'dgc_neg_weighted_tf_log_raw_df': weighted_tf_log_raw_df_neg,
                                   'dgc_neg_weighted_tf_ele_df': weighted_tf_ele_df_neg,
                                   'dgc_neg_weighted_tf_log_ele_df': weighted_tf_log_ele_df_neg,
                                   'dgc_neg_weighted_tf_rec_df': weighted_tf_rec_df_neg,
                                   'dgc_neg_weighted_tf_log_rec_df': weighted_tf_log_rec_df_neg})

            # direction/orientation tuning of dff responses in positive direction ===========================================
            dire_tuning_dff_pos = dgcrt_dff.get_dire_tuning(response_dir='pos',
                                                            is_collapse_sf=params['is_collapse_sf'],
                                                            is_collapse_tf=params['is_collapse_tf'])
            osi_dff_pos_raw, dsi_dff_pos_raw, gosi_dff_pos_raw, gdsi_dff_pos_raw, \
            osi_dff_pos_ele, dsi_dff_pos_ele, gosi_dff_pos_ele, gdsi_dff_pos_ele, \
            osi_dff_pos_rec, dsi_dff_pos_rec, gosi_dff_pos_rec, gdsi_dff_pos_rec, \
            peak_dire_raw_dff_pos, vs_dire_raw_dff_pos, vs_dire_ele_dff_pos, vs_dire_rec_dff_pos \
                = dgcrt_dff.get_dire_tuning_properties(dire_tuning_dff_pos,
                                                       response_dir='pos',
                                                       elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_pos_osi_raw_dff': osi_dff_pos_raw,
                                   'dgc_pos_dsi_raw_dff': dsi_dff_pos_raw,
                                   'dgc_pos_gosi_raw_dff': gosi_dff_pos_raw,
                                   'dgc_pos_gdsi_raw_dff': gdsi_dff_pos_raw,
                                   'dgc_pos_osi_ele_dff': osi_dff_pos_ele,
                                   'dgc_pos_dsi_ele_dff': dsi_dff_pos_ele,
                                   'dgc_pos_gosi_ele_dff': gosi_dff_pos_ele,
                                   'dgc_pos_gdsi_ele_dff': gdsi_dff_pos_ele,
                                   'dgc_pos_osi_rec_dff': osi_dff_pos_rec,
                                   'dgc_pos_dsi_rec_dff': dsi_dff_pos_rec,
                                   'dgc_pos_gosi_rec_dff': gosi_dff_pos_rec,
                                   'dgc_pos_gdsi_rec_dff': gdsi_dff_pos_rec,
                                   'dgc_pos_peak_dire_raw_dff': peak_dire_raw_dff_pos,
                                   'dgc_pos_vs_dire_raw_dff': vs_dire_raw_dff_pos,
                                   'dgc_pos_vs_dire_ele_dff': vs_dire_ele_dff_pos,
                                   'dgc_pos_vs_dire_rec_dff': vs_dire_rec_dff_pos})

            # direction/orientation tuning of dff responses in negative direction ===========================================
            dire_tuning_dff_neg = dgcrt_dff.get_dire_tuning(response_dir='neg',
                                                            is_collapse_sf=params['is_collapse_sf'],
                                                            is_collapse_tf=params['is_collapse_tf'])
            osi_dff_neg_raw, dsi_dff_neg_raw, gosi_dff_neg_raw, gdsi_dff_neg_raw, \
            osi_dff_neg_ele, dsi_dff_neg_ele, gosi_dff_neg_ele, gdsi_dff_neg_ele, \
            osi_dff_neg_rec, dsi_dff_neg_rec, gosi_dff_neg_rec, gdsi_dff_neg_rec, \
            peak_dire_raw_dff_neg, vs_dire_raw_dff_neg, vs_dire_ele_dff_neg, vs_dire_rec_dff_neg \
                = dgcrt_dff.get_dire_tuning_properties(dire_tuning_dff_neg,
                                                       response_dir='neg',
                                                       elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_neg_osi_raw_dff': osi_dff_neg_raw,
                                   'dgc_neg_dsi_raw_dff': dsi_dff_neg_raw,
                                   'dgc_neg_gosi_raw_dff': gosi_dff_neg_raw,
                                   'dgc_neg_gdsi_raw_dff': gdsi_dff_neg_raw,
                                   'dgc_neg_osi_ele_dff': osi_dff_neg_ele,
                                   'dgc_neg_dsi_ele_dff': dsi_dff_neg_ele,
                                   'dgc_neg_gosi_ele_dff': gosi_dff_neg_ele,
                                   'dgc_neg_gdsi_ele_dff': gdsi_dff_neg_ele,
                                   'dgc_neg_osi_rec_dff': osi_dff_neg_rec,
                                   'dgc_neg_dsi_rec_dff': dsi_dff_neg_rec,
                                   'dgc_neg_gosi_rec_dff': gosi_dff_neg_rec,
                                   'dgc_neg_gdsi_rec_dff': gdsi_dff_neg_rec,
                                   'dgc_neg_peak_dire_raw_dff': peak_dire_raw_dff_neg,
                                   'dgc_neg_vs_dire_raw_dff': vs_dire_raw_dff_neg,
                                   'dgc_neg_vs_dire_ele_dff': vs_dire_ele_dff_neg,
                                   'dgc_neg_vs_dire_rec_dff': vs_dire_rec_dff_neg})

            # sf tuning of dff responses in positive direction ==============================================================
            sf_tuning_dff_pos = dgcrt_dff.get_sf_tuning(response_dir='pos', is_collapse_tf=params['is_collapse_tf'],
                                                        is_collapse_dire=params['is_collapse_dire'])
            peak_sf_raw_dff_pos, weighted_sf_raw_dff_pos, weighted_sf_log_raw_dff_pos, \
            weighted_sf_ele_dff_pos, weighted_sf_log_ele_dff_pos, \
            weighted_sf_rec_dff_pos, weighted_sf_log_rec_dff_pos = \
                dgcrt_dff.get_sf_tuning_properties(sf_tuning_dff_pos, response_dir='pos',
                                                   elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_pos_peak_sf_raw_dff': peak_sf_raw_dff_pos,
                                   'dgc_pos_weighted_sf_raw_dff': weighted_sf_raw_dff_pos,
                                   'dgc_pos_weighted_sf_log_raw_dff': weighted_sf_log_raw_dff_pos,
                                   'dgc_pos_weighted_sf_ele_dff': weighted_sf_ele_dff_pos,
                                   'dgc_pos_weighted_sf_log_ele_dff': weighted_sf_log_ele_dff_pos,
                                   'dgc_pos_weighted_sf_rec_dff': weighted_sf_rec_dff_pos,
                                   'dgc_pos_weighted_sf_log_rec_dff': weighted_sf_log_rec_dff_pos})

            # sf tuning of dff responses in negative direction ==============================================================
            sf_tuning_dff_neg = dgcrt_dff.get_sf_tuning(response_dir='neg', is_collapse_tf=params['is_collapse_tf'],
                                                        is_collapse_dire=params['is_collapse_dire'])
            peak_sf_raw_dff_neg, weighted_sf_raw_dff_neg, weighted_sf_log_raw_dff_neg, \
            weighted_sf_ele_dff_neg, weighted_sf_log_ele_dff_neg, \
            weighted_sf_rec_dff_neg, weighted_sf_log_rec_dff_neg = \
                dgcrt_dff.get_sf_tuning_properties(sf_tuning_dff_neg, response_dir='neg',
                                                   elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_neg_peak_sf_raw_dff': peak_sf_raw_dff_neg,
                                   'dgc_neg_weighted_sf_raw_dff': weighted_sf_raw_dff_neg,
                                   'dgc_neg_weighted_sf_log_raw_dff': weighted_sf_log_raw_dff_neg,
                                   'dgc_neg_weighted_sf_ele_dff': weighted_sf_ele_dff_neg,
                                   'dgc_neg_weighted_sf_log_ele_dff': weighted_sf_log_ele_dff_neg,
                                   'dgc_neg_weighted_sf_rec_dff': weighted_sf_rec_dff_neg,
                                   'dgc_neg_weighted_sf_log_rec_dff': weighted_sf_log_rec_dff_neg})

            # tf tuning of dff responses in positive direction ==============================================================
            tf_tuning_dff_pos = dgcrt_dff.get_tf_tuning(response_dir='pos', is_collapse_sf=params['is_collapse_sf'],
                                                        is_collapse_dire=params['is_collapse_dire'])
            peak_tf_raw_dff_pos, weighted_tf_raw_dff_pos, weighted_tf_log_raw_dff_pos, \
            weighted_tf_ele_dff_pos, weighted_tf_log_ele_dff_pos, \
            weighted_tf_rec_dff_pos, weighted_tf_log_rec_dff_pos = \
                dgcrt_dff.get_tf_tuning_properties(tf_tuning_dff_pos, response_dir='pos',
                                                   elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_pos_peak_tf_raw_dff': peak_tf_raw_dff_pos,
                                   'dgc_pos_weighted_tf_raw_dff': weighted_tf_raw_dff_pos,
                                   'dgc_pos_weighted_tf_log_raw_dff': weighted_tf_log_raw_dff_pos,
                                   'dgc_pos_weighted_tf_ele_dff': weighted_tf_ele_dff_pos,
                                   'dgc_pos_weighted_tf_log_ele_dff': weighted_tf_log_ele_dff_pos,
                                   'dgc_pos_weighted_tf_rec_dff': weighted_tf_rec_dff_pos,
                                   'dgc_pos_weighted_tf_log_rec_dff': weighted_tf_log_rec_dff_pos})

            # tf tuning of dff responses in negative direction ==============================================================
            tf_tuning_dff_neg = dgcrt_dff.get_tf_tuning(response_dir='neg', is_collapse_sf=params['is_collapse_sf'],
                                                        is_collapse_dire=params['is_collapse_dire'])
            peak_tf_raw_dff_neg, weighted_tf_raw_dff_neg, weighted_tf_log_raw_dff_neg, \
            weighted_tf_ele_dff_neg, weighted_tf_log_ele_dff_neg, \
            weighted_tf_rec_dff_neg, weighted_tf_log_rec_dff_neg = \
                dgcrt_dff.get_tf_tuning_properties(tf_tuning_dff_neg, response_dir='neg',
                                                   elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_neg_peak_tf_raw_dff': peak_tf_raw_dff_neg,
                                   'dgc_neg_weighted_tf_raw_dff': weighted_tf_raw_dff_neg,
                                   'dgc_neg_weighted_tf_log_raw_dff': weighted_tf_log_raw_dff_neg,
                                   'dgc_neg_weighted_tf_ele_dff': weighted_tf_ele_dff_neg,
                                   'dgc_neg_weighted_tf_log_ele_dff': weighted_tf_log_ele_dff_neg,
                                   'dgc_neg_weighted_tf_rec_dff': weighted_tf_rec_dff_neg,
                                   'dgc_neg_weighted_tf_log_rec_dff': weighted_tf_log_rec_dff_neg})

            # direction/orientation tuning of zscore responses in positive direction ===========================================
            dire_tuning_z_pos = dgcrt_z.get_dire_tuning(response_dir='pos',
                                                        is_collapse_sf=params['is_collapse_sf'],
                                                        is_collapse_tf=params['is_collapse_tf'])
            osi_z_pos_raw, dsi_z_pos_raw, gosi_z_pos_raw, gdsi_z_pos_raw, \
            osi_z_pos_ele, dsi_z_pos_ele, gosi_z_pos_ele, gdsi_z_pos_ele, \
            osi_z_pos_rec, dsi_z_pos_rec, gosi_z_pos_rec, gdsi_z_pos_rec, \
            peak_dire_raw_z_pos, vs_dire_raw_z_pos, vs_dire_ele_z_pos, vs_dire_rec_z_pos \
                = dgcrt_z.get_dire_tuning_properties(dire_tuning_z_pos,
                                                     response_dir='pos',
                                                     elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_pos_osi_raw_z': osi_z_pos_raw,
                                   'dgc_pos_dsi_raw_z': dsi_z_pos_raw,
                                   'dgc_pos_gosi_raw_z': gosi_z_pos_raw,
                                   'dgc_pos_gdsi_raw_z': gdsi_z_pos_raw,
                                   'dgc_pos_osi_ele_z': osi_z_pos_ele,
                                   'dgc_pos_dsi_ele_z': dsi_z_pos_ele,
                                   'dgc_pos_gosi_ele_z': gosi_z_pos_ele,
                                   'dgc_pos_gdsi_ele_z': gdsi_z_pos_ele,
                                   'dgc_pos_osi_rec_z': osi_z_pos_rec,
                                   'dgc_pos_dsi_rec_z': dsi_z_pos_rec,
                                   'dgc_pos_gosi_rec_z': gosi_z_pos_rec,
                                   'dgc_pos_gdsi_rec_z': gdsi_z_pos_rec,
                                   'dgc_pos_peak_dire_raw_z': peak_dire_raw_z_pos,
                                   'dgc_pos_vs_dire_raw_z': vs_dire_raw_z_pos,
                                   'dgc_pos_vs_dire_ele_z': vs_dire_ele_z_pos,
                                   'dgc_pos_vs_dire_rec_z': vs_dire_rec_z_pos})

            # direction/orientation tuning of zscore responses in negative direction ===========================================
            dire_tuning_z_neg = dgcrt_z.get_dire_tuning(response_dir='neg',
                                                        is_collapse_sf=params['is_collapse_sf'],
                                                        is_collapse_tf=params['is_collapse_tf'])
            osi_z_neg_raw, dsi_z_neg_raw, gosi_z_neg_raw, gdsi_z_neg_raw, \
            osi_z_neg_ele, dsi_z_neg_ele, gosi_z_neg_ele, gdsi_z_neg_ele, \
            osi_z_neg_rec, dsi_z_neg_rec, gosi_z_neg_rec, gdsi_z_neg_rec, \
            peak_dire_raw_z_neg, vs_dire_raw_z_neg, vs_dire_ele_z_neg, vs_dire_rec_z_neg \
                = dgcrt_z.get_dire_tuning_properties(dire_tuning_z_neg,
                                                     response_dir='neg',
                                                     elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_neg_osi_raw_z': osi_z_neg_raw,
                                   'dgc_neg_dsi_raw_z': dsi_z_neg_raw,
                                   'dgc_neg_gosi_raw_z': gosi_z_neg_raw,
                                   'dgc_neg_gdsi_raw_z': gdsi_z_neg_raw,
                                   'dgc_neg_osi_ele_z': osi_z_neg_ele,
                                   'dgc_neg_dsi_ele_z': dsi_z_neg_ele,
                                   'dgc_neg_gosi_ele_z': gosi_z_neg_ele,
                                   'dgc_neg_gdsi_ele_z': gdsi_z_neg_ele,
                                   'dgc_neg_osi_rec_z': osi_z_neg_rec,
                                   'dgc_neg_dsi_rec_z': dsi_z_neg_rec,
                                   'dgc_neg_gosi_rec_z': gosi_z_neg_rec,
                                   'dgc_neg_gdsi_rec_z': gdsi_z_neg_rec,
                                   'dgc_neg_peak_dire_raw_z': peak_dire_raw_z_neg,
                                   'dgc_neg_vs_dire_raw_z': vs_dire_raw_z_neg,
                                   'dgc_neg_vs_dire_ele_z': vs_dire_ele_z_neg,
                                   'dgc_neg_vs_dire_rec_z': vs_dire_rec_z_neg})

            # sf tuning of zscore responses in positive direction ==============================================================
            sf_tuning_z_pos = dgcrt_z.get_sf_tuning(response_dir='pos', is_collapse_tf=params['is_collapse_tf'],
                                                    is_collapse_dire=params['is_collapse_dire'])
            peak_sf_raw_z_pos, weighted_sf_raw_z_pos, weighted_sf_log_raw_z_pos, \
            weighted_sf_ele_z_pos, weighted_sf_log_ele_z_pos, \
            weighted_sf_rec_z_pos, weighted_sf_log_rec_z_pos = \
                dgcrt_z.get_sf_tuning_properties(sf_tuning_z_pos, response_dir='pos',
                                                 elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_pos_peak_sf_raw_z': peak_sf_raw_z_pos,
                                   'dgc_pos_weighted_sf_raw_z': weighted_sf_raw_z_pos,
                                   'dgc_pos_weighted_sf_log_raw_z': weighted_sf_log_raw_z_pos,
                                   'dgc_pos_weighted_sf_ele_z': weighted_sf_ele_z_pos,
                                   'dgc_pos_weighted_sf_log_ele_z': weighted_sf_log_ele_z_pos,
                                   'dgc_pos_weighted_sf_rec_z': weighted_sf_rec_z_pos,
                                   'dgc_pos_weighted_sf_log_rec_z': weighted_sf_log_rec_z_pos})

            # sf tuning of zscore responses in negative direction ==============================================================
            sf_tuning_z_neg = dgcrt_z.get_sf_tuning(response_dir='neg', is_collapse_tf=params['is_collapse_tf'],
                                                    is_collapse_dire=params['is_collapse_dire'])
            peak_sf_raw_z_neg, weighted_sf_raw_z_neg, weighted_sf_log_raw_z_neg, \
            weighted_sf_ele_z_neg, weighted_sf_log_ele_z_neg, \
            weighted_sf_rec_z_neg, weighted_sf_log_rec_z_neg = \
                dgcrt_z.get_sf_tuning_properties(sf_tuning_z_neg, response_dir='neg',
                                                 elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_neg_peak_sf_raw_z': peak_sf_raw_z_neg,
                                   'dgc_neg_weighted_sf_raw_z': weighted_sf_raw_z_neg,
                                   'dgc_neg_weighted_sf_log_raw_z': weighted_sf_log_raw_z_neg,
                                   'dgc_neg_weighted_sf_ele_z': weighted_sf_ele_z_neg,
                                   'dgc_neg_weighted_sf_log_ele_z': weighted_sf_log_ele_z_neg,
                                   'dgc_neg_weighted_sf_rec_z': weighted_sf_rec_z_neg,
                                   'dgc_neg_weighted_sf_log_rec_z': weighted_sf_log_rec_z_neg})

            # tf tuning of zcore responses in positive direction ==============================================================
            tf_tuning_z_pos = dgcrt_z.get_tf_tuning(response_dir='pos', is_collapse_sf=params['is_collapse_sf'],
                                                    is_collapse_dire=params['is_collapse_dire'])
            peak_tf_raw_z_pos, weighted_tf_raw_z_pos, weighted_tf_log_raw_z_pos, \
            weighted_tf_ele_z_pos, weighted_tf_log_ele_z_pos, \
            weighted_tf_rec_z_pos, weighted_tf_log_rec_z_pos = \
                dgcrt_z.get_tf_tuning_properties(tf_tuning_z_pos, response_dir='pos',
                                                 elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_pos_peak_tf_raw_z': peak_tf_raw_z_pos,
                                   'dgc_pos_weighted_tf_raw_z': weighted_tf_raw_z_pos,
                                   'dgc_pos_weighted_tf_log_raw_z': weighted_tf_log_raw_z_pos,
                                   'dgc_pos_weighted_tf_ele_z': weighted_tf_ele_z_pos,
                                   'dgc_pos_weighted_tf_log_ele_z': weighted_tf_log_ele_z_pos,
                                   'dgc_pos_weighted_tf_rec_z': weighted_tf_rec_z_pos,
                                   'dgc_pos_weighted_tf_log_rec_z': weighted_tf_log_rec_z_pos})

            # tf tuning of zscore responses in negative direction ==============================================================
            tf_tuning_z_neg = dgcrt_z.get_tf_tuning(response_dir='neg', is_collapse_sf=params['is_collapse_sf'],
                                                    is_collapse_dire=params['is_collapse_dire'])
            peak_tf_raw_z_neg, weighted_tf_raw_z_neg, weighted_tf_log_raw_z_neg, \
            weighted_tf_ele_z_neg, weighted_tf_log_ele_z_neg, \
            weighted_tf_rec_z_neg, weighted_tf_log_rec_z_neg = \
                dgcrt_z.get_tf_tuning_properties(tf_tuning_z_neg, response_dir='neg',
                                                 elevation_bias=params['dgc_elevation_bias'])
            axon_properties.update({'dgc_neg_peak_tf_raw_z': peak_tf_raw_z_neg,
                                   'dgc_neg_weighted_tf_raw_z': weighted_tf_raw_z_neg,
                                   'dgc_neg_weighted_tf_log_raw_z': weighted_tf_log_raw_z_neg,
                                   'dgc_neg_weighted_tf_ele_z': weighted_tf_ele_z_neg,
                                   'dgc_neg_weighted_tf_log_ele_z': weighted_tf_log_ele_z_neg,
                                   'dgc_neg_weighted_tf_rec_z': weighted_tf_rec_z_neg,
                                   'dgc_neg_weighted_tf_log_rec_z': weighted_tf_log_rec_z_neg})

        else:
            dgcrm_df = None
            dgcrm_dff = None
            dgcrm_z = None
            dgcrt_df = None
            dgcrt_dff = None
            dgcrt_z = None
            dgc_block_dur = None

            axon_properties.update({'dgc_pos_peak_df': np.nan,
                                   'dgc_neg_peak_df': np.nan,
                                   'dgc_pos_p_ttest_df': np.nan,
                                   'dgc_neg_p_ttest_df': np.nan,
                                   'dgc_p_anova_df': np.nan,
                                   'dgc_pos_peak_dff': np.nan,
                                   'dgc_neg_peak_dff': np.nan,
                                   'dgc_pos_p_ttest_dff': np.nan,
                                   'dgc_neg_p_ttest_dff': np.nan,
                                   'dgc_p_anova_dff': np.nan,
                                   'dgc_pos_peak_z': np.nan,
                                   'dgc_neg_peak_z': np.nan,
                                   'dgc_pos_p_ttest_z': np.nan,
                                   'dgc_neg_p_ttest_z': np.nan,
                                   'dgc_p_anova_z': np.nan,

                                   'dgc_pos_osi_raw_df': np.nan,
                                   'dgc_pos_dsi_raw_df': np.nan,
                                   'dgc_pos_gosi_raw_df': np.nan,
                                   'dgc_pos_gdsi_raw_df': np.nan,
                                   'dgc_pos_osi_ele_df': np.nan,
                                   'dgc_pos_dsi_ele_df': np.nan,
                                   'dgc_pos_gosi_ele_df': np.nan,
                                   'dgc_pos_gdsi_ele_df': np.nan,
                                   'dgc_pos_osi_rec_df': np.nan,
                                   'dgc_pos_dsi_rec_df': np.nan,
                                   'dgc_pos_gosi_rec_df': np.nan,
                                   'dgc_pos_gdsi_rec_df': np.nan,
                                   'dgc_pos_peak_dire_raw_df': np.nan,
                                   'dgc_pos_vs_dire_raw_df': np.nan,
                                   'dgc_pos_vs_dire_ele_df': np.nan,
                                   'dgc_pos_vs_dire_rec_df': np.nan,
                                   'dgc_neg_osi_raw_df': np.nan,
                                   'dgc_neg_dsi_raw_df': np.nan,
                                   'dgc_neg_gosi_raw_df': np.nan,
                                   'dgc_neg_gdsi_raw_df': np.nan,
                                   'dgc_neg_osi_ele_df': np.nan,
                                   'dgc_neg_dsi_ele_df': np.nan,
                                   'dgc_neg_gosi_ele_df': np.nan,
                                   'dgc_neg_gdsi_ele_df': np.nan,
                                   'dgc_neg_osi_rec_df': np.nan,
                                   'dgc_neg_dsi_rec_df': np.nan,
                                   'dgc_neg_gosi_rec_df': np.nan,
                                   'dgc_neg_gdsi_rec_df': np.nan,
                                   'dgc_neg_peak_dire_raw_df': np.nan,
                                   'dgc_neg_vs_dire_raw_df': np.nan,
                                   'dgc_neg_vs_dire_ele_df': np.nan,
                                   'dgc_neg_vs_dire_rec_df': np.nan,
                                   'dgc_pos_peak_sf_raw_df': np.nan,
                                   'dgc_pos_weighted_sf_raw_df': np.nan,
                                   'dgc_pos_weighted_sf_log_raw_df': np.nan,
                                   'dgc_pos_weighted_sf_ele_df': np.nan,
                                   'dgc_pos_weighted_sf_log_ele_df': np.nan,
                                   'dgc_pos_weighted_sf_rec_df': np.nan,
                                   'dgc_pos_weighted_sf_log_rec_df': np.nan,
                                   'dgc_neg_peak_sf_raw_df': np.nan,
                                   'dgc_neg_weighted_sf_raw_df': np.nan,
                                   'dgc_neg_weighted_sf_log_raw_df': np.nan,
                                   'dgc_neg_weighted_sf_ele_df': np.nan,
                                   'dgc_neg_weighted_sf_log_ele_df': np.nan,
                                   'dgc_neg_weighted_sf_rec_df': np.nan,
                                   'dgc_neg_weighted_sf_log_rec_df': np.nan,
                                   'dgc_pos_peak_tf_raw_df': np.nan,
                                   'dgc_pos_weighted_tf_raw_df': np.nan,
                                   'dgc_pos_weighted_tf_log_raw_df': np.nan,
                                   'dgc_pos_weighted_tf_ele_df': np.nan,
                                   'dgc_pos_weighted_tf_log_ele_df': np.nan,
                                   'dgc_pos_weighted_tf_rec_df': np.nan,
                                   'dgc_pos_weighted_tf_log_rec_df': np.nan,
                                   'dgc_neg_peak_tf_raw_df': np.nan,
                                   'dgc_neg_weighted_tf_raw_df': np.nan,
                                   'dgc_neg_weighted_tf_log_raw_df': np.nan,
                                   'dgc_neg_weighted_tf_ele_df': np.nan,
                                   'dgc_neg_weighted_tf_log_ele_df': np.nan,
                                   'dgc_neg_weighted_tf_rec_df': np.nan,
                                   'dgc_neg_weighted_tf_log_rec_df': np.nan,

                                   'dgc_pos_osi_raw_dff': np.nan,
                                   'dgc_pos_dsi_raw_dff': np.nan,
                                   'dgc_pos_gosi_raw_dff': np.nan,
                                   'dgc_pos_gdsi_raw_dff': np.nan,
                                   'dgc_pos_osi_ele_dff': np.nan,
                                   'dgc_pos_dsi_ele_dff': np.nan,
                                   'dgc_pos_gosi_ele_dff': np.nan,
                                   'dgc_pos_gdsi_ele_dff': np.nan,
                                   'dgc_pos_osi_rec_dff': np.nan,
                                   'dgc_pos_dsi_rec_dff': np.nan,
                                   'dgc_pos_gosi_rec_dff': np.nan,
                                   'dgc_pos_gdsi_rec_dff': np.nan,
                                   'dgc_pos_peak_dire_raw_dff': np.nan,
                                   'dgc_pos_vs_dire_raw_dff': np.nan,
                                   'dgc_pos_vs_dire_ele_dff': np.nan,
                                   'dgc_pos_vs_dire_rec_dff': np.nan,
                                   'dgc_neg_osi_raw_dff': np.nan,
                                   'dgc_neg_dsi_raw_dff': np.nan,
                                   'dgc_neg_gosi_raw_dff': np.nan,
                                   'dgc_neg_gdsi_raw_dff': np.nan,
                                   'dgc_neg_osi_ele_dff': np.nan,
                                   'dgc_neg_dsi_ele_dff': np.nan,
                                   'dgc_neg_gosi_ele_dff': np.nan,
                                   'dgc_neg_gdsi_ele_dff': np.nan,
                                   'dgc_neg_osi_rec_dff': np.nan,
                                   'dgc_neg_dsi_rec_dff': np.nan,
                                   'dgc_neg_gosi_rec_dff': np.nan,
                                   'dgc_neg_gdsi_rec_dff': np.nan,
                                   'dgc_neg_peak_dire_raw_dff': np.nan,
                                   'dgc_neg_vs_dire_raw_dff': np.nan,
                                   'dgc_neg_vs_dire_ele_dff': np.nan,
                                   'dgc_neg_vs_dire_rec_dff': np.nan,
                                   'dgc_pos_peak_sf_raw_dff': np.nan,
                                   'dgc_pos_weighted_sf_raw_dff': np.nan,
                                   'dgc_pos_weighted_sf_log_raw_dff': np.nan,
                                   'dgc_pos_weighted_sf_ele_dff': np.nan,
                                   'dgc_pos_weighted_sf_log_ele_dff': np.nan,
                                   'dgc_pos_weighted_sf_rec_dff': np.nan,
                                   'dgc_pos_weighted_sf_log_rec_dff': np.nan,
                                   'dgc_neg_peak_sf_raw_dff': np.nan,
                                   'dgc_neg_weighted_sf_raw_dff': np.nan,
                                   'dgc_neg_weighted_sf_log_raw_dff': np.nan,
                                   'dgc_neg_weighted_sf_ele_dff': np.nan,
                                   'dgc_neg_weighted_sf_log_ele_dff': np.nan,
                                   'dgc_neg_weighted_sf_rec_dff': np.nan,
                                   'dgc_neg_weighted_sf_log_rec_dff': np.nan,
                                   'dgc_pos_peak_tf_raw_dff': np.nan,
                                   'dgc_pos_weighted_tf_raw_dff': np.nan,
                                   'dgc_pos_weighted_tf_log_raw_dff': np.nan,
                                   'dgc_pos_weighted_tf_ele_dff': np.nan,
                                   'dgc_pos_weighted_tf_log_ele_dff': np.nan,
                                   'dgc_pos_weighted_tf_rec_dff': np.nan,
                                   'dgc_pos_weighted_tf_log_rec_dff': np.nan,
                                   'dgc_neg_peak_tf_raw_dff': np.nan,
                                   'dgc_neg_weighted_tf_raw_dff': np.nan,
                                   'dgc_neg_weighted_tf_log_raw_dff': np.nan,
                                   'dgc_neg_weighted_tf_ele_dff': np.nan,
                                   'dgc_neg_weighted_tf_log_ele_dff': np.nan,
                                   'dgc_neg_weighted_tf_rec_dff': np.nan,
                                   'dgc_neg_weighted_tf_log_rec_dff': np.nan,

                                   'dgc_pos_osi_raw_z': np.nan,
                                   'dgc_pos_dsi_raw_z': np.nan,
                                   'dgc_pos_gosi_raw_z': np.nan,
                                   'dgc_pos_gdsi_raw_z': np.nan,
                                   'dgc_pos_osi_ele_z': np.nan,
                                   'dgc_pos_dsi_ele_z': np.nan,
                                   'dgc_pos_gosi_ele_z': np.nan,
                                   'dgc_pos_gdsi_ele_z': np.nan,
                                   'dgc_pos_osi_rec_z': np.nan,
                                   'dgc_pos_dsi_rec_z': np.nan,
                                   'dgc_pos_gosi_rec_z': np.nan,
                                   'dgc_pos_gdsi_rec_z': np.nan,
                                   'dgc_pos_peak_dire_raw_z': np.nan,
                                   'dgc_pos_vs_dire_raw_z': np.nan,
                                   'dgc_pos_vs_dire_ele_z': np.nan,
                                   'dgc_pos_vs_dire_rec_z': np.nan,
                                   'dgc_neg_osi_raw_z': np.nan,
                                   'dgc_neg_dsi_raw_z': np.nan,
                                   'dgc_neg_gosi_raw_z': np.nan,
                                   'dgc_neg_gdsi_raw_z': np.nan,
                                   'dgc_neg_osi_ele_z': np.nan,
                                   'dgc_neg_dsi_ele_z': np.nan,
                                   'dgc_neg_gosi_ele_z': np.nan,
                                   'dgc_neg_gdsi_ele_z': np.nan,
                                   'dgc_neg_osi_rec_z': np.nan,
                                   'dgc_neg_dsi_rec_z': np.nan,
                                   'dgc_neg_gosi_rec_z': np.nan,
                                   'dgc_neg_gdsi_rec_z': np.nan,
                                   'dgc_neg_peak_dire_raw_z': np.nan,
                                   'dgc_neg_vs_dire_raw_z': np.nan,
                                   'dgc_neg_vs_dire_ele_z': np.nan,
                                   'dgc_neg_vs_dire_rec_z': np.nan,
                                   'dgc_pos_peak_sf_raw_z': np.nan,
                                   'dgc_pos_weighted_sf_raw_z': np.nan,
                                   'dgc_pos_weighted_sf_log_raw_z': np.nan,
                                   'dgc_pos_weighted_sf_ele_z': np.nan,
                                   'dgc_pos_weighted_sf_log_ele_z': np.nan,
                                   'dgc_pos_weighted_sf_rec_z': np.nan,
                                   'dgc_pos_weighted_sf_log_rec_z': np.nan,
                                   'dgc_neg_peak_sf_raw_z': np.nan,
                                   'dgc_neg_weighted_sf_raw_z': np.nan,
                                   'dgc_neg_weighted_sf_log_raw_z': np.nan,
                                   'dgc_neg_weighted_sf_ele_z': np.nan,
                                   'dgc_neg_weighted_sf_log_ele_z': np.nan,
                                   'dgc_neg_weighted_sf_rec_z': np.nan,
                                   'dgc_neg_weighted_sf_log_rec_z': np.nan,
                                   'dgc_pos_peak_tf_raw_z': np.nan,
                                   'dgc_pos_weighted_tf_raw_z': np.nan,
                                   'dgc_pos_weighted_tf_log_raw_z': np.nan,
                                   'dgc_pos_weighted_tf_ele_z': np.nan,
                                   'dgc_pos_weighted_tf_log_ele_z': np.nan,
                                   'dgc_pos_weighted_tf_rec_z': np.nan,
                                   'dgc_pos_weighted_tf_log_rec_z': np.nan,
                                   'dgc_neg_peak_tf_raw_z': np.nan,
                                   'dgc_neg_weighted_tf_raw_z': np.nan,
                                   'dgc_neg_weighted_tf_log_raw_z': np.nan,
                                   'dgc_neg_weighted_tf_ele_z': np.nan,
                                   'dgc_neg_weighted_tf_log_ele_z': np.nan,
                                   'dgc_neg_weighted_tf_rec_z': np.nan,
                                   'dgc_neg_weighted_tf_log_rec_z': np.nan,
                                   })

    if verbose:
        # max_len_key = max([len(k) for k in axon_properties.keys()])
        # print max_len_key
        print('\n'.join(['{:>31}:  {}'.format(k, v) for k, v in axon_properties.items()]))

    return axon_properties, axon_roi, trace, srf_pos_on, srf_pos_off, srf_neg_on, srf_neg_off, dgcrm_df, dgcrm_dff, \
           dgcrm_z, dgcrt_df, dgcrt_dff, dgcrt_z, dgc_block_dur


def roi_page_report(nwb_f, plane_n, roi_n, params=ANALYSIS_PARAMS, plot_params=PLOTTING_PARAMS):
    """
    generate a page of description of an roi

    :param nwb_f: h5py.File object
    :param plane_n:
    :param roi_n:
    :param params:
    :return:
    """

    roi_ind = int(roi_n[-4:])

    roi_properties, roi, trace, srf_pos_on, srf_pos_off, srf_neg_on, srf_neg_off, dgcrm_df, dgcrm_dff, \
    dgcrm_z, dgcrt_df, dgcrt_dff, dgcrt_z, dgc_block_dur = get_everything_from_roi(nwb_f=nwb_f,
                                                                                   plane_n=plane_n,
                                                                                   roi_n=roi_n,
                                                                                   params=params)

    segmentation_grp = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane'.format(plane_n)]
    rf_img_grp = segmentation_grp['reference_images']
    if 'mean_projection' in rf_img_grp.keys():
        rf_img = rf_img_grp['mean_projection/data'][()]
    else:
        rf_img = rf_img_grp['max_projection/data'][()]

    f = plt.figure(figsize=plot_params['fig_size'], facecolor=plot_params['fig_facecolor'])

    # plot roi mask
    f.subplots_adjust(0, 0, 1, 1)
    ax_roi_img = f.add_axes(plot_params['ax_roi_img_coord'])
    ax_roi_img.imshow(ia.array_nor(rf_img), cmap='gray', vmin=plot_params['rf_img_vmin'],
                      vmax=plot_params['rf_img_vmax'], interpolation='nearest')
    pt.plot_mask_borders(mask=roi.get_binary_mask(), plotAxis=ax_roi_img, color=plot_params['roi_border_color'],
                         borderWidth=plot_params['roi_border_width'])
    ax_roi_img.set_axis_off()

    # plot traces
    trace_chunk_length = trace.shape[0] // plot_params['traces_panels']
    trace_max = np.max(trace)
    trace_min = np.min(trace)

    trace_axis_height = (plot_params['field_traces_coord'][3] - (0.01 * (plot_params['traces_panels'] - 1))) \
                        / plot_params['traces_panels']
    for trace_i in range(plot_params['traces_panels']):
        curr_trace_axis = f.add_axes([
            plot_params['field_traces_coord'][0],
            plot_params['field_traces_coord'][1] + trace_i * (0.01 + trace_axis_height),
            plot_params['field_traces_coord'][2],
            trace_axis_height
        ])
        curr_trace_chunk = trace[trace_i * trace_chunk_length: (trace_i + 1) * trace_chunk_length]
        curr_trace_axis.plot(curr_trace_chunk, color=plot_params['traces_color'],
                             lw=plot_params['traces_line_width'])
        curr_trace_axis.set_xlim([0, trace_chunk_length])
        curr_trace_axis.set_ylim([trace_min, trace_max])
        curr_trace_axis.set_axis_off()

    # plot receptive field
    if srf_pos_on is not None:
        ax_rf_pos = f.add_axes(plot_params['ax_rf_pos_coord'])
        zscore_pos = render_rb(rf_on=srf_pos_on.get_weighted_mask(),
                               rf_off=srf_pos_off.get_weighted_mask(), vmax=plot_params['rf_zscore_vmax'])
        ax_rf_pos.imshow(zscore_pos, interpolation='nearest')
        ax_rf_pos.set_axis_off()

        # plotting negative ON and OFF receptive fields
        ax_rf_neg = f.add_axes(plot_params['ax_rf_neg_coord'])
        zscore_neg = render_rb(rf_on=-srf_neg_on.get_weighted_mask(),
                               rf_off=-srf_neg_off.get_weighted_mask(), vmax=plot_params['rf_zscore_vmax'])
        ax_rf_neg.imshow(zscore_neg, interpolation='nearest')
        ax_rf_neg.set_axis_off()

    # select dgc response matrix and response table for plotting
    if plot_params['response_type_for_plot'] == 'df':
        dgcrm_plot = dgcrm_df
        dgcrt_plot = dgcrt_df
    elif plot_params['response_type_for_plot'] == 'dff':
        dgcrm_plot = dgcrm_dff
        dgcrt_plot = dgcrt_dff
    elif plot_params['response_type_for_plot'] == 'zscore':
        dgcrm_plot = dgcrm_z
        dgcrt_plot = dgcrt_z
    else:
        raise LookupError("Do not understand 'response_type_for_plot': {}. Should be "
                          "'df', 'dff' or 'zscore'.".format(params['response_type_for_plot']))

    if dgcrm_plot is not None:

        # plot peak condition traces
        ax_peak_traces_pos = f.add_axes(plot_params['ax_peak_traces_pos_coord'])
        ax_peak_traces_neg = f.add_axes(plot_params['ax_peak_traces_neg_coord'])

        ymin_pos, ymax_pos = dgcrm_plot.plot_traces(condi_ind=dgcrt_plot.peak_condi_ind_pos,
                                                    axis=ax_peak_traces_pos,
                                                    blank_ind=dgcrt_plot.blank_condi_ind,
                                                    block_dur=dgc_block_dur,
                                                    response_window=params['response_window_dgc'],
                                                    baseline_window=params['baseline_window_dgc'],
                                                    trace_color=plot_params['peak_traces_pos_color'],
                                                    block_face_color=plot_params['block_face_color'],
                                                    response_window_color=plot_params['response_window_color'],
                                                    baseline_window_color=plot_params['baseline_window_color'],
                                                    blank_trace_color=plot_params['blank_traces_color'],
                                                    lw_single=plot_params['single_traces_lw'],
                                                    lw_mean=plot_params['mean_traces_lw'])

        ymin_neg, ymax_neg = dgcrm_plot.plot_traces(condi_ind=dgcrt_plot.peak_condi_ind_neg,
                                                    axis=ax_peak_traces_neg,
                                                    blank_ind=dgcrt_plot.blank_condi_ind,
                                                    block_dur=dgc_block_dur,
                                                    response_window=params['response_window_dgc'],
                                                    baseline_window=params['baseline_window_dgc'],
                                                    trace_color=plot_params['peak_traces_neg_color'],
                                                    block_face_color=plot_params['block_face_color'],
                                                    response_window_color=plot_params['response_window_color'],
                                                    baseline_window_color=plot_params['baseline_window_color'],
                                                    blank_trace_color=plot_params['blank_traces_color'],
                                                    lw_single=plot_params['single_traces_lw'],
                                                    lw_mean=plot_params['mean_traces_lw'])

        ax_peak_traces_pos.set_ylim(min([ymin_pos, ymin_neg]), max([ymax_pos, ymax_neg]))
        ax_peak_traces_neg.set_ylim(min([ymin_pos, ymin_neg]), max([ymax_pos, ymax_neg]))
        ax_peak_traces_pos.set_xticks([])
        ax_peak_traces_pos.set_yticks([])
        ax_peak_traces_neg.set_xticks([])
        ax_peak_traces_neg.set_yticks([])

        # plot sf-tf matrix
        ax_sftf_pos = f.add_axes(plot_params['ax_sftf_pos_coord'])
        ax_sftf_neg = f.add_axes(plot_params['ax_sftf_neg_coord'])

        dgcrt_plot.plot_sf_tf_matrix(response_dir='pos',
                                     axis=ax_sftf_pos,
                                     cmap=plot_params['sftf_cmap'],
                                     vmax=plot_params['sftf_vmax'],
                                     vmin=plot_params['sftf_vmin'])
        dgcrt_plot.plot_sf_tf_matrix(response_dir='neg',
                                     axis=ax_sftf_neg,
                                     cmap=plot_params['sftf_cmap'],
                                     vmax=plot_params['sftf_vmax'],
                                     vmin=plot_params['sftf_vmin'])

        # plot direction tuning curve
        ax_dire_pos = f.add_axes(plot_params['ax_dire_pos_coord'], projection='polar')
        ax_dire_neg = f.add_axes(plot_params['ax_dire_neg_coord'], projection='polar')

        r_max_pos = dgcrt_plot.plot_dire_tuning(response_dir='pos', axis=ax_dire_pos,
                                                is_collapse_sf=params['is_collapse_sf'],
                                                is_collapse_tf=params['is_collapse_tf'],
                                                trace_color=plot_params['dire_color_pos'],
                                                lw=plot_params['dire_line_width'],
                                                postprocess=plot_params['dgc_postprocess'])

        r_max_neg = dgcrt_plot.plot_dire_tuning(response_dir='neg', axis=ax_dire_neg,
                                                is_collapse_sf=params['is_collapse_sf'],
                                                is_collapse_tf=params['is_collapse_tf'],
                                                trace_color=plot_params['dire_color_neg'],
                                                lw=plot_params['dire_line_width'],
                                                postprocess=plot_params['dgc_postprocess'])

        rmax = max([r_max_pos, r_max_neg])

        ax_dire_pos.set_rlim([0, rmax])
        ax_dire_pos.set_rticks([rmax])
        ax_dire_neg.set_rlim([0, rmax])
        ax_dire_neg.set_rticks([rmax])

    # print text
    ax_text = f.add_axes(plot_params['ax_text_coord'])
    ax_text.set_xticks([])
    ax_text.set_yticks([])

    file_n = os.path.splitext(os.path.split(nwb_f.filename)[1])[0]

    txt = '{}\n'.format(file_n)
    txt += '\n'
    txt += 'plane name:          {}\n'.format(plane_n)
    txt += 'roi name:            {}\n'.format(roi_n)
    txt += 'depth (um):          {}\n'.format(roi_properties['depth'])
    txt += 'roi area (um^2):     {:.2f}\n'.format(roi_properties['roi_area'])
    # txt += '\n'
    txt += 'trace type:{:>19}\n'.format(params['trace_type'])
    txt += 'response type:{:>14}\n'.format(plot_params['response_type_for_plot'])
    txt += 'dgc postprocess:{:>13}\n'.format(plot_params['dgc_postprocess'])
    txt += '\n'
    txt += 'skewness raw:        {:.2f}\n'.format(roi_properties['skew_raw'])
    txt += 'skewness fil:        {:.2f}\n'.format(roi_properties['skew_fil'])
    # txt += '\n'

    rf_pos_peak_z = max([roi_properties['rf_pos_on_peak_z'],
                         roi_properties['rf_pos_off_peak_z']])
    rf_neg_peak_z = max([roi_properties['rf_neg_on_peak_z'],
                         roi_properties['rf_neg_off_peak_z']])

    if plot_params['response_type_for_plot'] == 'df':
        surfix1 = 'df'
    elif plot_params['response_type_for_plot'] == 'dff':
        surfix1 = 'dff'
    elif plot_params['response_type_for_plot'] == 'zscore':
        surfix1 = 'z'
    else:
        raise LookupError("Do not ',understand 'response_type_for_plot': {}. Should be "
                          "'df 'dff' or 'zscore'.".format(plot_params['response_type_for_plot']))

    if plot_params['dgc_postprocess'] == 'raw':
        surfix2 = 'raw'
    elif plot_params['dgc_postprocess'] == 'elevate':
        surfix2 = 'ele'
    elif plot_params['dgc_postprocess'] == 'rectify':
        surfix2 = 'rec'
    else:
        raise LookupError("Do not ',understand 'response_type_for_plot': {}. Should be "
                          "'raw', 'elevate' or 'rectify'.".format(plot_params['dgc_postprocess']))

    txt += 'dgc_p_anova:         {:.2f}\n'.format(roi_properties['dgc_p_anova_{}'.format(surfix1)])
    txt += '\n'
    txt += 'positive response:\n'
    txt += 'rf_peak_z:           {:.2f}\n'.format(rf_pos_peak_z)
    txt += 'rf_lsi:              {:.2f}\n'.format(roi_properties['rf_pos_lsi'])
    txt += 'dgc_p_ttest:         {:.2f}\n'.format(roi_properties['dgc_pos_p_ttest_{}'.format(surfix1)])
    txt += 'dgc_peak_resp:       {:.2f}\n'.format(roi_properties['dgc_pos_peak_{}'.format(surfix1)])
    txt += 'dgc_OSI:             {:.2f}\n'.format(roi_properties['dgc_pos_osi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_gOSI:            {:.2f}\n'.format(roi_properties['dgc_pos_gosi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_DSI:             {:.2f}\n'.format(roi_properties['dgc_pos_dsi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_gDSI:            {:.2f}\n'.format(roi_properties['dgc_pos_gdsi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_vs_dire:         {:.2f}\n'.format(roi_properties['dgc_pos_vs_dire_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_sf:     {:.2f}\n'.format(roi_properties['dgc_pos_weighted_sf'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_sf_log: {:.2f}\n'.format(roi_properties['dgc_pos_weighted_sf_log'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_tf:     {:.2f}\n'.format(roi_properties['dgc_pos_weighted_tf'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_tf_log: {:.2f}\n'.format(roi_properties['dgc_pos_weighted_tf_log'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += '\nnegative response:\n'
    txt += 'rf_peak_z:           {:.2f}\n'.format(rf_neg_peak_z)
    txt += 'rf_lsi:              {:.2f}\n'.format(roi_properties['rf_neg_lsi'])
    txt += 'dgc_p_ttest:         {:.2f}\n'.format(roi_properties['dgc_neg_p_ttest_{}'.format(surfix1)])
    txt += 'dgc_peak_resp:       {:.2f}\n'.format(roi_properties['dgc_neg_peak_{}'.format(surfix1)])
    txt += 'dgc_OSI:             {:.2f}\n'.format(roi_properties['dgc_neg_osi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_gOSI:            {:.2f}\n'.format(roi_properties['dgc_neg_gosi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_DSI:             {:.2f}\n'.format(roi_properties['dgc_neg_dsi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_gDSI:            {:.2f}\n'.format(roi_properties['dgc_neg_gdsi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_vs_dire:         {:.2f}\n'.format(roi_properties['dgc_neg_vs_dire_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_sf:     {:.2f}\n'.format(roi_properties['dgc_neg_weighted_sf'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_sf_log: {:.2f}\n'.format(roi_properties['dgc_neg_weighted_sf_log'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_tf:     {:.2f}\n'.format(roi_properties['dgc_neg_weighted_tf'
                                                                     '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_tf_log: {:.2f}\n'.format(roi_properties['dgc_neg_weighted_tf_log'
                                                                     '_{}_{}'.format(surfix2, surfix1)])

    ax_text.text(0.01, 0.99, txt, horizontalalignment='left', verticalalignment='top', family='monospace')

    # plt.show()
    return f


def axon_page_report(nwb_f, clu_f, plane_n, axon_n, params=ANALYSIS_PARAMS, plot_params=PLOTTING_PARAMS):
    """
    generate a page of description of an roi

    :param nwb_f: h5py.File object
    :param plane_n:
    :param roi_n:
    :param params:
    :return:
    """

    roi_properties, roi, trace, srf_pos_on, srf_pos_off, srf_neg_on, srf_neg_off, dgcrm_df, dgcrm_dff, \
    dgcrm_z, dgcrt_df, dgcrt_dff, dgcrt_z, dgc_block_dur = get_everything_from_axon(nwb_f=nwb_f,
                                                                                    clu_f=clu_f,
                                                                                    plane_n=plane_n,
                                                                                    axon_n=axon_n,
                                                                                    params=params)

    segmentation_grp = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane'.format(plane_n)]
    rf_img_grp = segmentation_grp['reference_images']
    if 'mean_projection' in rf_img_grp.keys():
        rf_img = rf_img_grp['mean_projection/data'][()]
    else:
        rf_img = rf_img_grp['max_projection/data'][()]

    f = plt.figure(figsize=plot_params['fig_size'], facecolor=plot_params['fig_facecolor'])

    # plot roi mask
    f.subplots_adjust(0, 0, 1, 1)
    ax_roi_img = f.add_axes(plot_params['ax_roi_img_coord'])
    ax_roi_img.imshow(ia.array_nor(rf_img), cmap='gray', vmin=plot_params['rf_img_vmin'],
                      vmax=plot_params['rf_img_vmax'], interpolation='nearest')
    pt.plot_mask_borders(mask=roi.get_binary_mask(), plotAxis=ax_roi_img, color=plot_params['roi_border_color'],
                         borderWidth=plot_params['roi_border_width'])
    ax_roi_img.set_axis_off()

    # plot traces
    trace_chunk_length = trace.shape[0] // plot_params['traces_panels']
    trace_max = np.max(trace)
    trace_min = np.min(trace)

    trace_axis_height = (plot_params['field_traces_coord'][3] - (0.01 * (plot_params['traces_panels'] - 1))) \
                        / plot_params['traces_panels']
    for trace_i in range(plot_params['traces_panels']):
        curr_trace_axis = f.add_axes([
            plot_params['field_traces_coord'][0],
            plot_params['field_traces_coord'][1] + trace_i * (0.01 + trace_axis_height),
            plot_params['field_traces_coord'][2],
            trace_axis_height
        ])
        curr_trace_chunk = trace[trace_i * trace_chunk_length: (trace_i + 1) * trace_chunk_length]
        curr_trace_axis.plot(curr_trace_chunk, color=plot_params['traces_color'],
                             lw=plot_params['traces_line_width'])
        curr_trace_axis.set_xlim([0, trace_chunk_length])
        curr_trace_axis.set_ylim([trace_min, trace_max])
        curr_trace_axis.set_axis_off()

    # plot receptive field
    if srf_pos_on is not None:
        ax_rf_pos = f.add_axes(plot_params['ax_rf_pos_coord'])
        zscore_pos = render_rb(rf_on=srf_pos_on.get_weighted_mask(),
                               rf_off=srf_pos_off.get_weighted_mask(), vmax=plot_params['rf_zscore_vmax'])
        ax_rf_pos.imshow(zscore_pos, interpolation='nearest')
        ax_rf_pos.set_axis_off()

        # plotting negative ON and OFF receptive fields
        ax_rf_neg = f.add_axes(plot_params['ax_rf_neg_coord'])
        zscore_neg = render_rb(rf_on=-srf_neg_on.get_weighted_mask(),
                               rf_off=-srf_neg_off.get_weighted_mask(), vmax=plot_params['rf_zscore_vmax'])
        ax_rf_neg.imshow(zscore_neg, interpolation='nearest')
        ax_rf_neg.set_axis_off()

    # select dgc response matrix and response table for plotting
    if plot_params['response_type_for_plot'] == 'df':
        dgcrm_plot = dgcrm_df
        dgcrt_plot = dgcrt_df
    elif plot_params['response_type_for_plot'] == 'dff':
        dgcrm_plot = dgcrm_dff
        dgcrt_plot = dgcrt_dff
    elif plot_params['response_type_for_plot'] == 'zscore':
        dgcrm_plot = dgcrm_z
        dgcrt_plot = dgcrt_z
    else:
        raise LookupError("Do not understand 'response_type_for_plot': {}. Should be "
                          "'df', 'dff' or 'zscore'.".format(params['response_type_for_plot']))

    if dgcrm_plot is not None:
        # plot peak condition traces
        ax_peak_traces_pos = f.add_axes(plot_params['ax_peak_traces_pos_coord'])
        ax_peak_traces_neg = f.add_axes(plot_params['ax_peak_traces_neg_coord'])

        ymin_pos, ymax_pos = dgcrm_plot.plot_traces(condi_ind=dgcrt_plot.peak_condi_ind_pos,
                                                    axis=ax_peak_traces_pos,
                                                    blank_ind=dgcrt_plot.blank_condi_ind,
                                                    block_dur=dgc_block_dur,
                                                    response_window=params['response_window_dgc'],
                                                    baseline_window=params['baseline_window_dgc'],
                                                    trace_color=plot_params['peak_traces_pos_color'],
                                                    block_face_color=plot_params['block_face_color'],
                                                    response_window_color=plot_params['response_window_color'],
                                                    baseline_window_color=plot_params['baseline_window_color'],
                                                    blank_trace_color=plot_params['blank_traces_color'],
                                                    lw_single=plot_params['single_traces_lw'],
                                                    lw_mean=plot_params['mean_traces_lw'])

        ymin_neg, ymax_neg = dgcrm_plot.plot_traces(condi_ind=dgcrt_plot.peak_condi_ind_neg,
                                                    axis=ax_peak_traces_neg,
                                                    blank_ind=dgcrt_plot.blank_condi_ind,
                                                    block_dur=dgc_block_dur,
                                                    response_window=params['response_window_dgc'],
                                                    baseline_window=params['baseline_window_dgc'],
                                                    trace_color=plot_params['peak_traces_neg_color'],
                                                    block_face_color=plot_params['block_face_color'],
                                                    response_window_color=plot_params['response_window_color'],
                                                    baseline_window_color=plot_params['baseline_window_color'],
                                                    blank_trace_color=plot_params['blank_traces_color'],
                                                    lw_single=plot_params['single_traces_lw'],
                                                    lw_mean=plot_params['mean_traces_lw'])

        ax_peak_traces_pos.set_ylim(min([ymin_pos, ymin_neg]), max([ymax_pos, ymax_neg]))
        ax_peak_traces_neg.set_ylim(min([ymin_pos, ymin_neg]), max([ymax_pos, ymax_neg]))
        ax_peak_traces_pos.set_xticks([])
        ax_peak_traces_pos.set_yticks([])
        ax_peak_traces_neg.set_xticks([])
        ax_peak_traces_neg.set_yticks([])

        # plot sf-tf matrix
        ax_sftf_pos = f.add_axes(plot_params['ax_sftf_pos_coord'])
        ax_sftf_neg = f.add_axes(plot_params['ax_sftf_neg_coord'])

        dgcrt_plot.plot_sf_tf_matrix(response_dir='pos',
                                     axis=ax_sftf_pos,
                                     cmap=plot_params['sftf_cmap'],
                                     vmax=plot_params['sftf_vmax'],
                                     vmin=plot_params['sftf_vmin'])
        dgcrt_plot.plot_sf_tf_matrix(response_dir='neg',
                                     axis=ax_sftf_neg,
                                     cmap=plot_params['sftf_cmap'],
                                     vmax=plot_params['sftf_vmax'],
                                     vmin=plot_params['sftf_vmin'])

        # plot direction tuning curve
        ax_dire_pos = f.add_axes(plot_params['ax_dire_pos_coord'], projection='polar')
        ax_dire_neg = f.add_axes(plot_params['ax_dire_neg_coord'], projection='polar')

        r_max_pos = dgcrt_plot.plot_dire_tuning(response_dir='pos', axis=ax_dire_pos,
                                                is_collapse_sf=params['is_collapse_sf'],
                                                is_collapse_tf=params['is_collapse_tf'],
                                                trace_color=plot_params['dire_color_pos'],
                                                lw=plot_params['dire_line_width'],
                                                postprocess=plot_params['dgc_postprocess'])

        r_max_neg = dgcrt_plot.plot_dire_tuning(response_dir='neg', axis=ax_dire_neg,
                                                is_collapse_sf=params['is_collapse_sf'],
                                                is_collapse_tf=params['is_collapse_tf'],
                                                trace_color=plot_params['dire_color_neg'],
                                                lw=plot_params['dire_line_width'],
                                                postprocess=plot_params['dgc_postprocess'])

        rmax = max([r_max_pos, r_max_neg])

        ax_dire_pos.set_rlim([0, rmax])
        ax_dire_pos.set_rticks([rmax])
        ax_dire_neg.set_rlim([0, rmax])
        ax_dire_neg.set_rticks([rmax])

    # print text
    ax_text = f.add_axes(plot_params['ax_text_coord'])
    ax_text.set_xticks([])
    ax_text.set_yticks([])

    file_n = os.path.splitext(os.path.split(nwb_f.filename)[1])[0]

    txt = '{}\n'.format(file_n)
    txt += '\n'
    txt += 'plane name:          {}\n'.format(plane_n)
    txt += 'roi name:            {}\n'.format(axon_n)
    txt += 'depth (um):          {}\n'.format(roi_properties['depth'])
    txt += 'roi area (um^2):     {:.2f}\n'.format(roi_properties['roi_area'])
    # txt += '\n'
    txt += 'trace type:{:>19}\n'.format(params['trace_type'])
    txt += 'response type:{:>14}\n'.format(plot_params['response_type_for_plot'])
    txt += 'dgc postprocess:{:>13}\n'.format(plot_params['dgc_postprocess'])
    txt += '\n'
    txt += 'skewness raw:        {:.2f}\n'.format(roi_properties['skew_raw'])
    txt += 'skewness fil:        {:.2f}\n'.format(roi_properties['skew_fil'])
    # txt += '\n'

    rf_pos_peak_z = max([roi_properties['rf_pos_on_peak_z'],
                         roi_properties['rf_pos_off_peak_z']])
    rf_neg_peak_z = max([roi_properties['rf_neg_on_peak_z'],
                         roi_properties['rf_neg_off_peak_z']])

    if plot_params['response_type_for_plot'] == 'df':
        surfix1 = 'df'
    elif plot_params['response_type_for_plot'] == 'dff':
        surfix1 = 'dff'
    elif plot_params['response_type_for_plot'] == 'zscore':
        surfix1 = 'z'
    else:
        raise LookupError("Do not ',understand 'response_type_for_plot': {}. Should be "
                          "'df 'dff' or 'zscore'.".format(plot_params['response_type_for_plot']))

    if plot_params['dgc_postprocess'] == 'raw':
        surfix2 = 'raw'
    elif plot_params['dgc_postprocess'] == 'elevate':
        surfix2 = 'ele'
    elif plot_params['dgc_postprocess'] == 'rectify':
        surfix2 = 'rec'
    else:
        raise LookupError("Do not ',understand 'response_type_for_plot': {}. Should be "
                          "'raw', 'elevate' or 'rectify'.".format(plot_params['dgc_postprocess']))

    txt += 'dgc_p_anova:         {:.2f}\n'.format(roi_properties['dgc_p_anova_{}'.format(surfix1)])
    txt += '\n'
    txt += 'positive response:\n'
    txt += 'rf_peak_z:           {:.2f}\n'.format(rf_pos_peak_z)
    txt += 'rf_lsi:              {:.2f}\n'.format(roi_properties['rf_pos_lsi'])
    txt += 'dgc_p_ttest:         {:.2f}\n'.format(roi_properties['dgc_pos_p_ttest_{}'.format(surfix1)])
    txt += 'dgc_peak_resp:       {:.2f}\n'.format(roi_properties['dgc_pos_peak_{}'.format(surfix1)])
    txt += 'dgc_OSI:             {:.2f}\n'.format(roi_properties['dgc_pos_osi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_gOSI:            {:.2f}\n'.format(roi_properties['dgc_pos_gosi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_DSI:             {:.2f}\n'.format(roi_properties['dgc_pos_dsi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_gDSI:            {:.2f}\n'.format(roi_properties['dgc_pos_gdsi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_vs_dire:         {:.2f}\n'.format(roi_properties['dgc_pos_vs_dire_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_sf:     {:.2f}\n'.format(roi_properties['dgc_pos_weighted_sf'
                                                                 '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_sf_log: {:.2f}\n'.format(roi_properties['dgc_pos_weighted_sf_log'
                                                                 '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_tf:     {:.2f}\n'.format(roi_properties['dgc_pos_weighted_tf'
                                                                 '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_tf_log: {:.2f}\n'.format(roi_properties['dgc_pos_weighted_tf_log'
                                                                 '_{}_{}'.format(surfix2, surfix1)])
    txt += '\nnegative response:\n'
    txt += 'rf_peak_z:           {:.2f}\n'.format(rf_neg_peak_z)
    txt += 'rf_lsi:              {:.2f}\n'.format(roi_properties['rf_neg_lsi'])
    txt += 'dgc_p_ttest:         {:.2f}\n'.format(roi_properties['dgc_neg_p_ttest_{}'.format(surfix1)])
    txt += 'dgc_peak_resp:       {:.2f}\n'.format(roi_properties['dgc_neg_peak_{}'.format(surfix1)])
    txt += 'dgc_OSI:             {:.2f}\n'.format(roi_properties['dgc_neg_osi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_gOSI:            {:.2f}\n'.format(roi_properties['dgc_neg_gosi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_DSI:             {:.2f}\n'.format(roi_properties['dgc_neg_dsi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_gDSI:            {:.2f}\n'.format(roi_properties['dgc_neg_gdsi_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_vs_dire:         {:.2f}\n'.format(roi_properties['dgc_neg_vs_dire_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_sf:     {:.2f}\n'.format(roi_properties['dgc_neg_weighted_sf'
                                                                 '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_sf_log: {:.2f}\n'.format(roi_properties['dgc_neg_weighted_sf_log'
                                                                 '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_tf:     {:.2f}\n'.format(roi_properties['dgc_neg_weighted_tf'
                                                                 '_{}_{}'.format(surfix2, surfix1)])
    txt += 'dgc_weighted_tf_log: {:.2f}\n'.format(roi_properties['dgc_neg_weighted_tf_log'
                                                                 '_{}_{}'.format(surfix2, surfix1)])

    ax_text.text(0.01, 0.99, txt, horizontalalignment='left', verticalalignment='top', family='monospace')

    # plt.show()
    return f


class BoutonClassifier(object):

    def __init__(self, skew_filter_sigma=5., skew_thr=0.6, lowpass_sigma=0.1, detrend_sigma=3.,
                 event_std_thr=3., peri_event_dur=(-3., 3.), corr_len_thr=300., corr_abs_thr=0.5,
                 corr_std_thr=3., distance_measure='dis_corr_coef', distance_metric='euclidean',
                 linkage_method='weighted', distance_thr=1.0):
        """
        initiate the object. setup a bunch of analysis parameters.

        for detailed the bouton classification method, please see: Liang et al, Cell, 2018, 173:1343

        There are a few simplifications that I found fitted better to my data.
        1. not necessary to run cosine similarity to get distance matrix.
        2. not necessary to use x std above mean to threshold correlation coefficient matrix, one absolute value is
           enough
        3. the application of distance matrix is somewhat different from the scipy documentation. The documentation
           says feed y matrix to scipy.cluster.hierarchy.linkage() will generate the corrected linkage. But it is
           not the clear what

        :param skew_filter_sigma: float, in second, sigma for gaussian filter for skewness
        :param skew_thr: float, threshold of skewness of filtered trace to pickup responsive traces
        :param lowpass_sigma: float, in second, sigma for gaussian filter to highpass single trace
        :param detrend_sigma: float, in second, sigma for gaussian filter to remove slow trend
        :param event_std_thr: float, how many standard deviation above mean to detect events
        :param peri_event_dur: list of two floats, in seconds, pre- and post- duration to be included into detected
                               events
        :param corr_len_thr: float, in seconds, length threshold to calculate correlation between a pair
                                       of two traces. if the length is too short (detected events are too few),
                                       their correlation coefficient will be set to 0.
        :param corr_abs_thr: float, [0, 1], absolute threshold to treat correlation coefficient matrix
        :param corr_std_thr: float, how many standard deviation above the mean to threshold correlation
                                    coefficient matrix for each roi (currently not implemented)
        :param distance_measure: str, options 'cosine_similarity', 'corr_coef', 'dis_corr_coef'. Default: 'corr_coef'
                                 'corr_coef': use '1 -  correlation coefficient' as distance directly
                                 'dis_corr_coef': use correlation coefficient as observation and calculate euclidean
                                                  distance between each pair
                                 'cosine_similarity': use '1 - cosine_similarity' as distance, this is described in
                                                      Liang et al., 2018 cell paper, but did not work well for my
                                                      data.
        :param distance_metric: str, metric for scipy to get distance. "metric" input to scipy.spatial.distance.pdist()
                                method and scipy.cluster.hierarchy.linkage method.
        :param linkage_method: str, method argument to the scipy.cluster.hierarchy.linkage() method
        :param distance_thr: float, positive, the distance threshold to classify boutons into axons from the linkage
                             array
        """

        self.skew_filter_sigma = float(skew_filter_sigma)
        self.skew_thr = float(skew_thr)
        self.lowpass_sigma = float(lowpass_sigma)
        self.detrend_sigma = float(detrend_sigma)
        self.event_std_thr = float(event_std_thr)
        self.peri_event_dur = tuple(peri_event_dur)
        self.corr_len_thr = float(corr_len_thr)

        if corr_abs_thr >= 1.:
            raise ValueError('input "corr_abs_thr" should be smaller than 1.')
        elif corr_abs_thr < 0.:
            print('input "corr_abs_thr" is less than 0. Setting it to be 0.')
            self.corr_abs_thr = 0.
        else:
            self.corr_abs_thr = float(corr_abs_thr)

        self.corr_std_thr = float(corr_std_thr)

        if distance_measure in ['cosine_similarity', 'corr_coef', 'dis_corr_coef']:
            self.distance_measure = str(distance_measure)
        else:
            raise LookupError("the input 'distance measure' should be in "
                              "['cosine_similarity', 'corr_coef', 'dis_corr_coef'].")

        self.distance_metric = str(distance_metric)
        self.linkage_method = str(linkage_method)
        self.distance_thr = float(distance_thr)

    def filter_traces(self, traces, roi_ns, sample_dur):
        """
        filter traces by filtered skewness, also detect events for each traces
        :param traces: n x m array, n: roi numbers, m: time points
        :param roi_ns: list of strings, length = n, name of all rois
        :param sample_dur: float, duration of each sample in second.
        :return traces_res: l x m array, l number of rois that pass the skewness thresold, self.skew_thr
        :return roi_ns_res: list of strings, length = l, name of these rois
        :return event_masks: n x m array, dtype: np.bool, event masks for each roi. events are deteced by
                             larger than trace_mean + self.event_std_thr * trace_std
        """

        if traces.shape[0] != len(roi_ns):
            raise ValueError('traces.shape[0] ({}) should be the same as len(roi_ns) ({})'.format(traces.shape[0],
                                                                                                  len(roi_ns)))

        lowpass_sig_pt = self.lowpass_sigma / sample_dur
        detrend_sig_pt = self.detrend_sigma / sample_dur

        trace_ts = np.arange(traces.shape[1]) * sample_dur

        event_start_pt = int(np.floor(self.peri_event_dur[0] / sample_dur))
        event_end_pt = int(np.ceil(self.peri_event_dur[1] / sample_dur))

        roi_ns_res = []
        traces_res = []
        event_masks = []

        for trace_i, trace in enumerate(traces):
            _, skew_fil = sca.get_skewness(trace=trace, ts=trace_ts,
                                           filter_length=self.skew_filter_sigma)

            if skew_fil >= self.skew_thr:

                trace_l = ni.gaussian_filter1d(trace, sigma=lowpass_sig_pt) # lowpass
                trace_d = trace_l - ni.gaussian_filter1d(trace_l, sigma=detrend_sig_pt) # detrend

                # get event masks
                event_mask = np.zeros(trace_ts.shape, dtype=np.bool)

                trace_mean = np.mean(trace_d)
                trace_std = np.std(trace_d)
                event_intervals = ta.threshold_to_intervals(trace=trace_d,
                                                            thr=trace_mean + self.event_std_thr * trace_std,
                                                            comparison='>=')

                if len(event_intervals) != 0: # filter out the rois that have no detected event
                    for inte in event_intervals:
                        start_ind = max([0, inte[0] + event_start_pt])
                        end_ind = min([inte[1] + event_end_pt, traces.shape[1]])
                        event_mask[start_ind : end_ind] = True

                    roi_ns_res.append(roi_ns[trace_i])
                    traces_res.append(trace_d)
                    event_masks.append(event_mask)

        return np.array(traces_res), roi_ns_res, np.array(event_masks)

    def get_correlation_coefficient_matrix(self, traces, event_masks, sample_dur, is_plot=False):
        """
        calculate event based correcation coefficient matrix of a set of rois.
        ideally, the traces and event_masks will be the output of self.filter_traces() method.

        :param traces: l x m array, l: number of rois, m: number of time points
        :param event_masks: array same size of traces, dtype=np.bool, masks of event for each trace.
        :param sample_dur:  float, duration of each sample in second.
        :param is_plot: bool
        :return mat_corr: l x l array, correlation coefficient matrix
        """

        roi_num_res = traces.shape[0]
        mat_corr = np.zeros((roi_num_res, roi_num_res))
        np.fill_diagonal(mat_corr, 1.)

        for i in range(0, roi_num_res - 1):
            for j in range(i + 1, roi_num_res):
                trace_i = traces[i]
                event_mask_i = event_masks[i]

                trace_j = traces[j]
                event_mask_j = event_masks[j]

                res_ind_merge = np.logical_or(event_mask_i, event_mask_j)
                #         print("({}, {}), trace_length: {}".format(i, j, np.sum(res_ind_merge)))

                if np.sum(res_ind_merge) < self.corr_len_thr // sample_dur:
                    mat_corr[i, j] = 0
                    mat_corr[j, i] = 0
                else:
                    trace_i = trace_i[res_ind_merge]
                    trace_j = trace_j[res_ind_merge]
                    coeff = np.corrcoef(np.array([trace_i, trace_j]), rowvar=True)
                    mat_corr[i, j] = coeff[1, 0]
                    mat_corr[j, i] = coeff[1, 0]

        if is_plot:
            f = plt.figure(figsize=(8, 6))
            ax = f.add_subplot(111)
            ax.set_title('corr coef matrix')
            fig = ax.imshow(mat_corr, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
            f.colorbar(fig)
            plt.show()

        return mat_corr

    def threshold_correlation_coefficient_matrix(self, mat_corr, is_plot=False):
        """
        threshold correlation coefficient matrix based on each roi

        for each roi, the corr coeff smaller than min([self.corr_abs_thr, mean + self.corr_std_thr * std])
        will be set zero

        :param mat_corr:
        :param is_plot:
        :return:
        """

        mask = np.ones(mat_corr.shape)

        for row_i, row in enumerate(mat_corr):
            curr_std = np.std(row)
            curr_mean = np.mean(row)
            curr_thr = min([self.corr_abs_thr, curr_mean + self.corr_std_thr * curr_std])

            mask[row_i, :][row < curr_thr] = 0.
            mask[:, row_i][row < curr_thr] = 0.

        mat_corr_thr = mat_corr * mask

        if is_plot:
            f = plt.figure(figsize=(8, 6))
            ax = f.add_subplot(111)
            ax.set_title('thresholded corr coef matrix')
            fig = ax.imshow(mat_corr, cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
            f.colorbar(fig)
            plt.show()

        return mat_corr_thr

    def get_distance_matrix(self, mat_corr, is_plot=False):
        """
        calculated the square form of distance matrix based on self.distance_measure and self.distance_metric

        :param mat_corr:
        :param is_plot:
        :return:
        """

        if self.distance_measure == 'cosine_similarity':

            mat_dis = np.zeros(mat_corr.shape)
            roi_num = mat_dis.shape[0]
            # print('total roi number: {}'.format(roi_num))
            for i in range(roi_num):
                for j in range(i+1, roi_num, 1):

                    ind = np.ones(roi_num, dtype=np.bool)
                    ind[i] = 0
                    ind[j] = 0

                    row_i = mat_corr[i][ind]
                    row_j = mat_corr[j][ind]

                    if ((row_i * row_j) == 0).all():
                            mat_dis[i, j] = 1
                            mat_dis[j, i] = 1
                    else:
                        cos = spatial.distance.cosine(row_i, row_j)
                        mat_dis[i, j] = 1 - cos
                        mat_dis[j, i] = 1 - cos
        elif self.distance_measure == 'corr_coef':
            mat_dis = 1 - mat_corr
        elif self.distance_measure == 'dis_corr_coef':
            mat_dis = spatial.distance.squareform(spatial.distance.pdist(mat_corr, metric=self.distance_metric))
        else:
            raise LookupError('do not understand self.distance_measure: {}. should be in '
                              '["cosine_similarity", "corr_coef", "dis_corr_coef"].'.format(self.distance_measure))

        if is_plot:
            f = plt.figure(figsize=(8, 6))
            ax = f.add_subplot(111)
            ax.set_title('distance matrix')
            fig = ax.imshow(mat_dis, cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
            f.colorbar(fig)
            plt.show()

        return mat_dis

    def hierarchy_clustering(self, mat_dis, is_plot=False, **kwargs):
        """

        cluster the boutons based on the distance matrix using scipy.cluster.hierarchy.linkage function
        the "method" argument of this function is defined by self.linkage_method

        :param mat_dis: 2d array, distance matrix
        :param is_plot: bool
        :param kwargs: other inputs to scipy.cluster.hierarchy.dendrogram function
        :return linkage_z: 2d array, the linkage array Z from scipy.cluster.hierarchy.linkage method
        :return mat_dis_reorg: 2d array, reorganized the distance matrix based on the clustering
        :return c: float, the cophenetic correlation distance of the clustering. Value range: [0, 1].
                   Better if it is more close to 1.
        """

        linkage_z = cluster.hierarchy.linkage(mat_dis, method=self.linkage_method, metric=self.distance_metric)

        if len(mat_dis.shape) == 1:
            c, _ = cluster.hierarchy.cophenet(linkage_z, mat_dis)
        else:
            c, _ = cluster.hierarchy.cophenet(linkage_z, spatial.distance.pdist(mat_dis, metric=self.distance_metric))

        print('\tCophentic correlation distance of clustering: {}'.format(c))


        if is_plot:
            f_den = plt.figure(figsize=(20, 8))
            ax_den = f_den.add_subplot(111)
            _ = cluster.hierarchy.dendrogram(linkage_z, ax=ax_den, **kwargs)
            ax_den.axhline(y=self.distance_thr)
            plt.show()

        return linkage_z, c

    @staticmethod
    def reorganize_matrix_by_cluster(linkage_z, mat):
        """
        :param linkage_z:
        :param mat:
        :return:
        """

        if len(mat.shape) != 2:
            raise ValueError('input "mat" should be a 2d array.')

        if mat.shape[0] != mat.shape[1]:
            raise ValueError('input "mat" should have same rows as columns.')

        clu = np.array(cluster.hierarchy.fcluster(linkage_z, t=0, criterion='distance'))
        clu = clu - 1

        mat_0 = np.zeros(mat.shape)

        for l_i, l in enumerate(clu):
            mat_0[l, :] = mat[l_i, :]

        mat_reorg = np.zeros(mat.shape)
        for l_i, l in enumerate(clu):
            mat_reorg[:, l] = mat_0[:, l_i]

        return mat_reorg

    def get_axon_dict(self, linkage_z, roi_ns):
        """
        generate a dictionary of clustered axons.

        :param linkage_z: 2d array, the linkage array genearted by scipy.cluster.hierarchy.linkage method
        :param roi_ns: list of strings, roi names for the rois used for clustering
        :return axon_dict: dictionary of axons, each entry is {<axon_name> : list of roi names belong to that axon}
        :return clu_axon: list of integers, cluster index list generated by scipy.cluster.hierarchy.fcluster method
        """

        clu_axon = cluster.hierarchy.fcluster(linkage_z, t=self.distance_thr, criterion='distance')
        clu_axon = np.array(clu_axon)
        clu_axon = clu_axon - 1 # change to zero based indexing

        axon_num = max(clu_axon) + 1

        roi_ns = np.array(roi_ns)

        # get axon dictionary
        axon_dict = {}
        axon_num_multi_roi = 0

        for axon_i in range(axon_num):
            axon_n = 'axon_{:04d}'.format(axon_i)
            axon_lst = roi_ns[clu_axon == axon_i]
            axon_dict.update({axon_n: axon_lst})

            if len(axon_lst) > 1:
                axon_num_multi_roi += 1

        print('\ttotal number of rois: {}'
              '\n\ttotal number of axons: {} '
              '\n\tnumber of axons with multiple rois: {}'.format(len(roi_ns), axon_num, axon_num_multi_roi))

        return axon_dict, clu_axon

    @staticmethod
    def plot_chunked_traces_with_intervals(traces, trace_center=None, event_masks=None, chunk_num=4,
                                           fig_obj=None, colors=None, is_normalize_traces=False, **kwarg):
        """
        plot traces in defined number of chunks. Also mark the defined period indicated by the marked_inds

        :param traces: 2d array, float, each row is a single trace
        :param trace_center: 1d array, merged trace to plot, should have same length as traces.shape[1]
        :param event_masks: 2d array, bool, same size as traces, masks of detected events for each roi
        :param chunk_num: int, number of chunks for plotting
        :param fig_obj: matplotlib.figure object
        :param **kwarg: inputs to matplotlib.axes.plot() function
        """

        if len(traces.shape) == 1:
            traces_p = np.array([traces])
        elif len(traces.shape) == 2:
            traces_p = np.array(traces)
        else:
            raise ValueError("the input 'traces' should be a 2d array.")

        if is_normalize_traces:
            #=================================================================
            mins = np.min(traces_p, axis=1, keepdims=True)
            maxs = np.max(traces_p, axis=1, keepdims=True)
            traces_p = (traces_p - mins) / (maxs - mins)
            # =================================================================

            # =================================================================
            # mins = np.min(traces_p, axis=1, keepdims=True)
            # stds = np.std(traces_p, axis=1, keepdims=True)
            # traces_p = (traces_p - mins) / stds
            # =================================================================

            # =================================================================
            # medians = np.median(traces_p, axis=1, keepdims=True)
            # traces_p = (traces_p - medians) / medians
            # =================================================================

        if trace_center is not None:
            if is_normalize_traces:
                # =================================================================
                min_c = np.min(trace_center)
                max_c = np.max(trace_center)
                trace_center_p = (trace_center - min_c) / (max_c - min_c)
                # =================================================================

                # =================================================================
                # min_c = np.min(trace_center)
                # std_c = np.std(trace_center)
                # trace_center_p = (trace_center - min_c) / std_c
                # =================================================================

                # =================================================================
                # median_c = np.median(trace_center)
                # trace_center_p = (trace_center - median_c) / median_c
                # =================================================================
            else:
                trace_center_p = np.array(trace_center)

        if event_masks is not None and len(event_masks.shape) == 1:
            event_masks = np.array([event_masks])

        if event_masks is not None and traces_p.shape != event_masks.shape:
            raise ValueError(
                'the shape of input "traces" ({}) and "event_masks" ({}) are not the same.'.format(traces_p.shape,
                                                                                                   event_masks.shape))

        if fig_obj is None:
            fig_obj = plt.figure(figsize=(15, 10))

        if colors is None:
            colors = pt.random_color(traces_p.shape[0])

        len_tot = traces_p.shape[1]
        len_chunk = len_tot // chunk_num

        for chunk_i in range(chunk_num):
            chunk_traces = traces_p[:, chunk_i * len_chunk: (chunk_i + 1) * len_chunk]
            chunk_ax = fig_obj.add_subplot(chunk_num, 1, chunk_i + 1)
            chunk_ax.set_xlim([0, len_chunk])

            if is_normalize_traces:
                chunk_ax.set_ylim([-0.1, 1.1])
                chunk_ax.set_axis_off()

            for trace_i in range(traces_p.shape[0]):
                chunk_ax.plot(chunk_traces[trace_i], color=colors[trace_i], **kwarg)

            if trace_center is not None:
                chunk_trace_c = trace_center_p[chunk_i * len_chunk: (chunk_i + 1) * len_chunk]
                chunk_ax.plot(chunk_trace_c, color='#ff0000', lw=1, alpha=1)

            if event_masks is not None:
                chunk_inds = event_masks[:, chunk_i * len_chunk: (chunk_i + 1) * len_chunk]
                chunk_ind = np.any(chunk_inds, axis=0)
                chunk_intes = ta.threshold_to_intervals(trace=chunk_ind.astype(np.float32), thr=0.5, comparison='>=')
                for chunk_int in chunk_intes:
                    chunk_ax.axvspan(chunk_int[0], chunk_int[1], color='#000000', lw=None, alpha=0.2)

    def process_plane(self, nwb_f, save_folder, plane_n='plane0', trace_type='f_center_subtracted',
                      trace_window='AllStimuli', is_normalize_traces=False):
        """

        :param nwb_f:
        :param save_folder:
        :param plane_n:
        :param trace_type:
        :param trace_window:
        :return:
        """

        print('\tclustering ...')

        nwb_id = nwb_f['identifier'][()]
        date = nwb_id.split('_')[0]
        mid = nwb_id.split('_')[1]

        # the timestamp index for spontaneous activity
        traces, trace_ts = get_traces(nwb_f=nwb_f, plane_n=plane_n, trace_type=trace_type)
        sample_dur = np.mean(np.diff(trace_ts))

        if trace_window == 'AllStimuli':
            win_mask = np.ones(trace_ts.shape, dtype=np.bool)
            has_stim = True
        elif trace_window == 'UniformContrast':
            win_mask, has_stim = get_UC_ts_mask(nwb_f=nwb_f, plane_n=plane_n)
        elif trace_window == 'LocallySparseNoise':
            win_mask, has_stim = get_LSN_ts_mask(nwb_f=nwb_f, plane_n=plane_n)
        elif trace_window == 'DriftingGratingSpont':
            win_mask, has_stim = get_DGC_spont_ts_mask(nwb_f=nwb_f, plane_n=plane_n)
        else:
            raise LookupError('do not understand input "trace_window".')

        if not has_stim:
            print('the nwb file does not contain the specified stimulus: {}. Do nothing.'.format(trace_window))
            return

        traces_sub = traces[:, win_mask]

        roi_ns = get_roi_ns(nwb_f=nwb_f, plane_n=plane_n)

        traces_res, roi_ns_res, event_masks = self.filter_traces(traces=traces_sub, roi_ns=roi_ns,
                                                                 sample_dur=sample_dur)
        mat_corr = self.get_correlation_coefficient_matrix(traces=traces_res, event_masks=event_masks,
                                                           sample_dur=sample_dur, is_plot=False)

        mat_corr_thr = self.threshold_correlation_coefficient_matrix(mat_corr=mat_corr, is_plot=False)
        mat_dis = self.get_distance_matrix(mat_corr=mat_corr_thr, is_plot=False)
        mat_dis_dense = spatial.distance.squareform(mat_dis)
        linkage_z, c = self.hierarchy_clustering(mat_dis=mat_dis_dense, is_plot=False)

        # reorganize matrix
        mat_dis_reorg = self.reorganize_matrix_by_cluster(linkage_z=linkage_z, mat=mat_dis)
        mat_corr_reorg = self.reorganize_matrix_by_cluster(linkage_z=linkage_z, mat=mat_corr)

        # get axon clusters
        axon_dict, clu_axon = self.get_axon_dict(linkage_z=linkage_z, roi_ns=roi_ns_res)

        axon_ns = list(axon_dict.keys())
        axon_ns.sort()
        roi_num_per_axon = [len(axon_dict[axon_n]) for axon_n in axon_ns]
        # print(roi_num_per_axon)

        # save data
        print('\tsaving results ...')
        save_f = h5py.File(os.path.join(save_folder, '{}_{}_{}_axon_grouping.hdf5'.format(date, mid, plane_n)), 'a')

        meta_grp = save_f.create_group('meta')
        meta_grp.create_dataset('date', data=date)
        meta_grp.create_dataset('mouse_id', data=mid)
        meta_grp.create_dataset('plane_n', data=plane_n)
        meta_grp.create_dataset('trace_type', data=trace_type)
        meta_grp.create_dataset('trace_window', data=trace_window)

        bc_grp = save_f.create_group('classifier_parameters')
        for attr_n, attr in self.__dict__.items():
            bc_grp.create_dataset(attr_n, data=attr)

        save_f.create_dataset('matrix_corr_coef', data=mat_corr)
        save_f.create_dataset('matrix_corr_coef_thr', data=mat_corr_thr)
        save_f.create_dataset('matrix_distance', data=mat_dis)
        save_f.create_dataset('matrix_distance_reorg', data=mat_dis_reorg)
        save_f.create_dataset('matrix_corr_coef_thr_reorg', data=mat_corr_reorg)
        save_f.create_dataset('linkage_z', data=linkage_z)
        save_f.create_dataset('responsive_roi_ns', data=[r.encode('utf-8') for r in roi_ns_res])
        save_f.create_dataset('cluster_indices', data=clu_axon)
        axon_grp = save_f.create_group('axons')
        for axon_n, roi_lst in axon_dict.items():
            axon_grp.create_dataset(axon_n, data=[r.encode('utf-8') for r in roi_lst])

        # adding rois and traces
        trace_grp = nwb_f['processing/rois_and_traces_{}/Fluorescence'.format(plane_n)]
        seg_grp = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane'.format(plane_n)]

        axon_lst = []
        axon_masks = []
        axon_traces_raw = []
        axon_traces_sub = []

        for axon_n in axon_ns:
            roi_lst = axon_dict[axon_n]

            if len(roi_lst) > 1:

                curr_mask = np.zeros((512, 512), dtype=np.float32)
                curr_trace_raw = None
                curr_trace_sub = None
                total_weight = 0.

                for roi_n in roi_lst:

                    roi_i = int(roi_n[-4:])
                    curr_mask = curr_mask + seg_grp[roi_n]['img_mask'][()]
                    curr_weight = np.sum(seg_grp[roi_n]['pix_mask_weight'])
                    total_weight = total_weight + curr_weight

                    if curr_trace_raw is None:
                        curr_trace_raw = trace_grp['f_center_raw/data'][roi_i, :] * curr_weight
                    else:
                        curr_trace_raw = curr_trace_raw + trace_grp['f_center_raw/data'][roi_i, :] * curr_weight

                    if curr_trace_sub is None:
                        curr_trace_sub = trace_grp['f_center_subtracted/data'][roi_i, :] * curr_weight
                    else:
                        curr_trace_sub = curr_trace_sub + trace_grp['f_center_subtracted/data'][roi_i, :] * curr_weight

                axon_lst.append(axon_n)
                axon_masks.append(curr_mask)
                axon_traces_raw.append(curr_trace_raw / total_weight)
                axon_traces_sub.append(curr_trace_sub / total_weight)

        axon_masks = np.array(axon_masks)
        axon_traces_raw = np.array(axon_traces_raw)
        axon_traces_sub = np.array(axon_traces_sub)
        rat_grp = save_f.create_group('rois_and_traces')
        rat_grp.attrs['description'] = 'this group only list axons with more than one rois'
        rat_grp.create_dataset('axon_list', data=[a.encode('utf-8') for a in axon_lst])
        rat_grp.create_dataset('masks_center', data=axon_masks)
        rat_grp.create_dataset('traces_center_raw', data=axon_traces_raw)
        rat_grp.create_dataset('traces_center_subtracted', data=axon_traces_sub)
        save_f.close()

        # plot matrices
        sup_title_mat = '{}_{}_{}, {}, dis_thr={:.2f}'.format(date,
                                                              mid,
                                                              plane_n,
                                                              trace_window,
                                                              self.distance_thr)

        f_mat = self.plot_matrices(mat_corr=mat_corr, mat_corr_thr=mat_corr_thr, mat_dis=mat_dis,
                                   mat_corr_reorg=mat_corr_reorg, mat_dis_reorg=mat_dis_reorg,
                                   linkage_z=linkage_z, roi_num_per_axon=roi_num_per_axon,
                                   distance_thr=self.distance_thr, sup_title=sup_title_mat)

        f_mat.savefig(os.path.join(save_folder,
                                   '{}_{}_{}_{}_clustering.pdf'.format(date, mid, plane_n, trace_window)))

        plt.close(f_mat)

        # plot contours
        title_contour = '{}_{}_{}, {}, dis_thr={:.2f}'.format(date, mid, plane_n, trace_window,
                                                              self.distance_thr)
        f_contour = self.plot_countours_single_plane(nwb_f=nwb_f, plane_n=plane_n, axon_dict=axon_dict,
                                                     title_str=title_contour)

        f_contour.savefig(os.path.join(save_folder,
                                       '{}_{}_{}_{}_axon_contours.pdf'.format(date, mid, plane_n, trace_window)))

        plt.close(f_contour)

        # plot axon traces
        save_path_trace = os.path.join(save_folder, '{}_{}_{}_{}_axon_traces.pdf'.format(date,
                                                                                         mid,
                                                                                         plane_n,
                                                                                         trace_window))

        self.plot_traces_single_plane(save_path=save_path_trace, axon_dict=axon_dict, nwb_f=nwb_f,
                                           plane_n=plane_n, roi_num_per_axon=roi_num_per_axon,
                                           mat_corr_reorg=mat_corr_reorg, mat_dis_reorg=mat_dis_reorg,
                                           event_masks=event_masks, clu_axon=clu_axon, win_mask=win_mask,
                                           axon_lst=axon_lst, trace_type=trace_type,
                                           axon_traces_raw=axon_traces_raw, axon_traces_sub=axon_traces_sub,
                                           is_normalize_traces=is_normalize_traces)

        print('\tfinished.')

    @staticmethod
    def plot_matrices(mat_corr, mat_corr_thr, mat_dis, mat_corr_reorg, mat_dis_reorg, linkage_z,
                      roi_num_per_axon, distance_thr, sup_title=None, is_truncate=False):

        print('\tplotting clustering matrices ...')

        f_mat = plt.figure(figsize=(15, 12))
        if sup_title is not None:
            f_mat.suptitle(sup_title)

        ax_corr = f_mat.add_axes([0.02, 0.66, 0.3, 0.3])
        fig_corr = ax_corr.imshow(mat_corr, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        f_mat.colorbar(fig_corr)
        ax_corr.set_ylabel('corr coef')
        ax_corr.set_xticks([])
        ax_corr.set_yticks([])

        ax_corr_thr = f_mat.add_axes([0.34, 0.66, 0.3, 0.3])
        fig_corr_thr = ax_corr_thr.imshow(mat_corr_thr, cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
        f_mat.colorbar(fig_corr_thr)
        ax_corr_thr.set_ylabel('corr coef thr')
        ax_corr_thr.set_xticks([])
        ax_corr_thr.set_yticks([])

        ax_dis = f_mat.add_axes([0.66, 0.66, 0.3, 0.3])
        fig_dis = ax_dis.imshow(mat_dis, cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
        f_mat.colorbar(fig_dis)
        ax_dis.set_ylabel('distance')
        ax_dis.set_xticks([])
        ax_dis.set_yticks([])

        ax_corr_reorg = f_mat.add_axes([0.34, 0.34, 0.3, 0.3])
        fig_corr_reorg = ax_corr_reorg.imshow(mat_corr_reorg, cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
        f_mat.colorbar(fig_corr_reorg)
        ax_corr_reorg.set_ylabel('corr coef reorg')

        axon_roi_num_cum_sum = np.cumsum(roi_num_per_axon)
        for axon_i, roi_num in enumerate(roi_num_per_axon):
            if roi_num > 1:
                if axon_i == 0:
                    axon_start = 0
                else:
                    axon_start = axon_roi_num_cum_sum[axon_i - 1]
                axon_end = axon_roi_num_cum_sum[axon_i]
                axon_corr_mask = np.zeros(mat_corr_reorg.shape, dtype=np.uint8)
                axon_corr_mask[axon_start: axon_end, axon_start: axon_end] = 1
                pt.plot_mask_borders(axon_corr_mask, plotAxis=ax_corr_reorg, color='#009900', borderWidth=1)
        ax_corr_reorg.set_xticks([])
        ax_corr_reorg.set_yticks([])

        ax_dis_reorg = f_mat.add_axes([0.66, 0.34, 0.3, 0.3])
        fig_dis_reorg = ax_dis_reorg.imshow(mat_dis_reorg, cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
        f_mat.colorbar(fig_dis_reorg)
        ax_dis_reorg.set_ylabel('distance reorg')
        ax_dis_reorg.set_xticks([])
        ax_dis_reorg.set_yticks([])

        ax_den = f_mat.add_axes([0.02, 0.02, 0.96, 0.3])

        if not is_truncate:
            _ = cluster.hierarchy.dendrogram(linkage_z, ax=ax_den, color_threshold=distance_thr, no_labels=True)
        else:
            _ = cluster.hierarchy.dendrogram(linkage_z, ax=ax_den, color_threshold=distance_thr, no_labels=True,
                                             truncate_mode='level', p=20)
        ax_den.axhline(y=distance_thr)
        ax_den.set_title('dendrogram')

        return f_mat

    @staticmethod
    def plot_countours_single_plane(nwb_f, plane_n, axon_dict, title_str=''):
        print('\tplot axon contours ...')
        f = plt.figure(figsize=(8, 8))
        ax = f.add_subplot(111)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_axis_off()
        bg_img = get_background_img(nwb_f=nwb_f, plane_n=plane_n)

        if bg_img is not None:
            ax.imshow(ia.array_nor(bg_img), vmin=0, vmax=0.8, cmap='gray', interpolation='nearest')
        else:
            ax.imshow(np.zeros(512, 512), vmin=0, vmax=1, cmap='gray', interpolation='nearest')
        ax.set_title(title_str)

        for axon_n, roi_lst in axon_dict.items():

            if len(roi_lst) > 1:
                curr_color = pt.random_color(1)
                for roi_n in roi_lst:
                    curr_roi = get_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
                    pt.plot_mask_borders(curr_roi.get_binary_mask(), plotAxis=ax, color=curr_color[0],
                                         borderWidth=0.5)
            if len(roi_lst) == 1:
                curr_roi = get_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_lst[0])
                pt.plot_mask_borders(curr_roi.get_binary_mask(), plotAxis=ax, color='#aaaaaa',
                                     borderWidth=0.5)
        return f

    def plot_traces_single_plane(self, save_path, axon_dict, nwb_f, plane_n, roi_num_per_axon, mat_corr_reorg,
                                 mat_dis_reorg, event_masks, clu_axon, win_mask, axon_lst, trace_type,
                                 axon_traces_raw, axon_traces_sub, is_normalize_traces):

        # trace_f = PdfPages(os.path.join(save_folder,
        #                                 '{}_{}_{}_{}_axon_traces.pdf'.format(date, mid, plane_n, trace_window)))

        print('\tplot axon traces ...')

        trace_f = PdfPages(save_path)

        axon_ns = list(axon_dict.keys())
        axon_ns.sort()

        for axon_n in axon_ns:

            roi_lst = axon_dict[axon_n]

            if len(roi_lst) > 1:

                f = plt.figure(figsize=(10, 15), tight_layout=True)
                axon_int = int(axon_n[-4:])

                # get the mean correlation coefficient for an axon
                roi_ind_start = int(np.sum(roi_num_per_axon[0: axon_int]))
                roi_num = len(roi_lst)
                mean_corr = np.mean(mat_corr_reorg[roi_ind_start: roi_ind_start + roi_num,
                                    roi_ind_start: roi_ind_start + roi_num].flat)

                mean_dis = np.mean(mat_dis_reorg[roi_ind_start: roi_ind_start + roi_num,
                                   roi_ind_start: roi_ind_start + roi_num].flat)

                f.suptitle(
                    '{}: {} rois; mean corr coef: {:4.2f}; mean distance: {:6.4f}'.format(axon_n, len(roi_lst),
                                                                                          mean_corr,
                                                                                          mean_dis))

                axon_event_masks = event_masks[clu_axon == axon_int]

                traces_p = []
                for roi_n in roi_lst:
                    trace, _ = get_single_trace(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n, trace_type=trace_type)
                    traces_p.append(trace)
                traces_p = np.array(traces_p)
                traces_p = traces_p[:, win_mask]

                axon_plot_ind = axon_lst.index(axon_n)
                if trace_type == 'f_center_raw':
                    trace_center = axon_traces_raw[axon_plot_ind, :]
                elif trace_type == 'f_center_subtracted':
                    trace_center = axon_traces_sub[axon_plot_ind, :]
                else:
                    raise LookupError("do not under stand 'trace_type' ({}). Should be "
                                      "'f_center_raw' or 'f_center_subtracted'.".format(trace_type))

                self.plot_chunked_traces_with_intervals(traces_p,
                                                        trace_center=trace_center,
                                                        event_masks=axon_event_masks,
                                                        chunk_num=10,
                                                        fig_obj=f,
                                                        lw=0.5,
                                                        is_normalize_traces=is_normalize_traces)

                trace_f.savefig(f)
                plt.close(f)

        trace_f.close()

    @staticmethod
    def add_axon_strf_single_plane(nwb_f, clu_f, plane_n, t_win, verbose=False):
        """
        This is a very high level function to add spatial temporal receptive field of grouped axons into saved
        h5 log files.

        :param nwb_f: h5py.File object. Should be the same nwb file on which clustering was perfomed.
        :param clu_f: h5py.File object. Saved clustering file, generated by the 'process_plane()' function.
        :param plane_n: str
        :param t_win: list/tuple of two floats, peri-event time window to extract strf
        :param verbose: bool
        :return:
        """

        def get_sta(arr, arr_ts, trigger_ts, frame_start, frame_end):
            sta_arr = []

            for trig in trigger_ts:
                trig_ind = ta.find_nearest(arr_ts, trig)

                if trig_ind + frame_end < arr.shape[1]:
                    curr_sta = arr[:, (trig_ind + frame_start): (trig_ind + frame_end)]
                    # print(curr_sta.shape)
                    sta_arr.append(curr_sta.reshape((curr_sta.shape[0], 1, curr_sta.shape[1])))

            sta_arr = np.concatenate(sta_arr, axis=1)
            return sta_arr

        stim_ns = nwb_f['analysis/photodiode_onsets'].keys()
        lsn_stim_n = [n for n in stim_ns if 'LocallySparseNoiseRetinotopicMapping' in n]
        if len(lsn_stim_n) == 0:
            print('\tno locally sparse noise data, skip.')
            return
        elif len(lsn_stim_n) > 1:
            raise ('more than one locally sparse noise stimuli found.')
        else:
            lsn_stim_n = lsn_stim_n[0]

        if t_win[0] >= t_win[1]:
            raise ValueError('time window should be from early time to late time.')

        probe_onsets_path = 'analysis/photodiode_onsets/{}'.format(lsn_stim_n)
        probe_ns = list(nwb_f[probe_onsets_path].keys())
        probe_ns.sort()

        trace_ts = nwb_f['processing/motion_correction/MotionCorrection/{}/corrected/timestamps'.format(plane_n)]

        traces = {}
        for f_type in clu_f['rois_and_traces'].keys():
            if f_type[0:7] == 'traces_':
                if clu_f['rois_and_traces/{}'.format(f_type)].shape[0] == 0:
                    print('no clustered axons, skip')
                    return
                else:
                    traces.update({f_type: clu_f['rois_and_traces/{}'.format(f_type)][()]})

        frame_dur = np.mean(np.diff(trace_ts))
        frame_start = int(np.floor(t_win[0] / frame_dur))
        frame_end = int(np.ceil(t_win[1] / frame_dur))
        t_axis = np.arange(frame_end - frame_start) * frame_dur + (frame_start * frame_dur)

        strf_grp = clu_f.create_group('strf_{}'.format(lsn_stim_n))
        strf_plane_grp = strf_grp.create_group(plane_n)

        strf_plane_grp.attrs['sta_timestamps'] = t_axis

        for probe_i, probe_n in enumerate(probe_ns):

            if verbose:
                print('\tprocessing probe {} / {}'.format(probe_i + 1, len(probe_ns)))

            onsets_probe_grp = nwb_f['{}/{}'.format(probe_onsets_path, probe_n)]

            curr_probe_grp = strf_plane_grp.create_group(probe_n)

            probe_onsets = onsets_probe_grp['pd_onset_ts_sec'][()]

            curr_probe_grp['global_trigger_timestamps'] = nwb_f['/{}/{}/pd_onset_ts_sec'
                .format(probe_onsets_path, probe_n)][()]
            curr_probe_grp.attrs['sta_traces_dimenstion'] = 'roi x trial x timepoint'

            for trace_n, trace in traces.items():
                sta = get_sta(arr=trace, arr_ts=trace_ts, trigger_ts=probe_onsets, frame_start=frame_start,
                              frame_end=frame_end)
                # curr_probe_grp.create_dataset('sta_' + trace_n, data=sta, compression='lzf')
                curr_probe_grp.create_dataset('sta_' + trace_n, data=sta)

    @staticmethod
    def add_axon_dgcrm_single_plane(nwb_f, clu_f, plane_n, t_win, verbose=False):
        """
        This is a very high level function to add drifting grating response matrix of grouped axons into saved
        h5 log files.

        :param nwb_f: h5py.File object. Should be the same nwb file on which clustering was perfomed.
        :param clu_f: h5py.File object. Saved clustering file, generated by the 'process_plane()' function.
        :param plane_n: str
        :param t_win: list/tuple of two floats, peri-event time window to extract dgcrm
        :param verbose: bool
        :return:
        """

        def get_sta(arr, arr_ts, trigger_ts, frame_start, frame_end):
            sta_arr = []

            for trig in trigger_ts:
                trig_ind = ta.find_nearest(arr_ts, trig)

                if trig_ind + frame_end < arr.shape[1]:
                    curr_sta = arr[:, (trig_ind + frame_start): (trig_ind + frame_end)]
                    # print(curr_sta.shape)
                    sta_arr.append(curr_sta.reshape((curr_sta.shape[0], 1, curr_sta.shape[1])))

            sta_arr = np.concatenate(sta_arr, axis=1)
            return sta_arr

        if t_win[0] >= t_win[1]:
            raise ValueError('time window should be from early time to late time.')

        stim_ns = nwb_f['analysis/photodiode_onsets'].keys()
        dgc_stim_n = [n for n in stim_ns if 'DriftingGratingCircleRetinotopicMapping' in n]
        if len(dgc_stim_n) == 0:
            print('\tno drifting grating circle data, skip.')
            return
        elif len(dgc_stim_n) > 1:
            raise ('more than one drifting grating cricle stimuli found.')
        else:
            dgc_stim_n = dgc_stim_n[0]

        grating_onsets_path = 'analysis/photodiode_onsets/{}'.format(dgc_stim_n)
        grating_ns = list(nwb_f[grating_onsets_path].keys())
        grating_ns.sort()
        # print('\n'.join(grating_ns))

        trace_ts = nwb_f['processing/motion_correction/MotionCorrection/{}/corrected/timestamps'.format(plane_n)]

        traces = {}
        for f_type in clu_f['rois_and_traces'].keys():
            if f_type[0:7] == 'traces_':
                if clu_f['rois_and_traces/{}'.format(f_type)].shape[0] == 0:
                    print('no clustered axons, skip')
                    return
                else:
                    traces.update({f_type: clu_f['rois_and_traces/{}'.format(f_type)][()]})

        dgcrm_grp = clu_f.create_group('response_table_{}'.format(dgc_stim_n))
        dgcrm_plane_grp = dgcrm_grp.create_group(plane_n)

        frame_dur = np.mean(np.diff(trace_ts))
        frame_start = int(np.floor(t_win[0] / frame_dur))
        frame_end = int(np.ceil(t_win[1] / frame_dur))
        t_axis = np.arange(frame_end - frame_start) * frame_dur + (frame_start * frame_dur)
        dgcrm_plane_grp.attrs['sta_timestamps'] = t_axis

        for grating_i, grating_n in enumerate(grating_ns):

            if verbose:
                print('\tprocessing grating {} / {}'.format(grating_i + 1, len(grating_ns)))

            onsets_grating_grp = nwb_f['{}/{}'.format(grating_onsets_path, grating_n)]

            curr_grating_grp = dgcrm_plane_grp.create_group(grating_n)

            grating_onsets = onsets_grating_grp['pd_onset_ts_sec'][()]

            curr_grating_grp.attrs['global_trigger_timestamps'] = grating_onsets
            curr_grating_grp.attrs['sta_traces_dimenstion'] = 'roi x trial x timepoint'

            for trace_n, trace in traces.items():
                sta = get_sta(arr=trace, arr_ts=trace_ts, trigger_ts=grating_onsets, frame_start=frame_start,
                              frame_end=frame_end)
                # curr_grating_grp.create_dataset('sta_' + trace_n, data=sta, compression='lzf')
                curr_grating_grp.create_dataset('sta_' + trace_n, data=sta)


    def process_file(self, nwb_f, save_folder, trace_type='f_center_subtracted',
                     trace_window='AllStimuli', is_normalize_traces=False):
        """

        :param nwb_f:
        :param save_folder:
        :param trace_type:
        :param trace_window:
        :param is_normalize_traces:
        :return:
        """

        print('\tclustering ...')

        nwb_id = nwb_f['identifier'][()].decode('utf-8')
        date = nwb_id.split('_')[0]
        mid = nwb_id.split('_')[1]

        # combine all planes
        plane_ns = get_plane_ns(nwb_f=nwb_f)
        plane_ns.sort()

        if len(plane_ns) == 1:
            print('This nwb file has only one plane, please use "process_plane()" method.')
            return

        # use the middle plane to decide time.
        plane_n_mid = plane_ns[len(plane_ns) // 2]
        _, trace_ts = get_traces(nwb_f=nwb_f, plane_n=plane_n_mid, trace_type=trace_type)
        sample_dur = np.mean(np.diff(trace_ts))

        if trace_window == 'AllStimuli':
            win_mask = np.ones(trace_ts.shape, dtype=np.bool)
            has_stim = True
        elif trace_window == 'UniformContrast':
            win_mask, has_stim = get_UC_ts_mask(nwb_f=nwb_f, plane_n=plane_n_mid)
        elif trace_window == 'LocallySparseNoise':
            win_mask, has_stim = get_LSN_ts_mask(nwb_f=nwb_f, plane_n=plane_n_mid)
        elif trace_window == 'DriftingGratingSpont':
            win_mask, has_stim = get_DGC_spont_ts_mask(nwb_f=nwb_f, plane_n=plane_n_mid)
        else:
            raise LookupError('do not understand input "trace_window".')

        if not has_stim:
            print('the nwb file does not contain the specified stimulus: {}. Do nothing.'.format(trace_window))
            return

        # combine planes
        roi_ns = []
        traces = []
        min_len = [len(trace_ts)]

        for plane_n in plane_ns:

            curr_traces, _ = get_traces(nwb_f=nwb_f, plane_n=plane_n, trace_type=trace_type)
            traces.append(curr_traces)
            min_len.append(curr_traces.shape[1])

            curr_roi_ns = get_roi_ns(nwb_f=nwb_f, plane_n=plane_n)
            curr_roi_ns = ['{}_{}'.format(plane_n, r) for r in curr_roi_ns]
            roi_ns = roi_ns + curr_roi_ns

        min_len = np.min(min_len)
        traces = [t[:, 0: min_len] for t in traces]
        traces = np.concatenate(traces, axis=0)
        trace_ts = trace_ts[:min_len]
        win_mask = win_mask[:min_len]

        traces_sub = traces[:, win_mask]

        # # for debugging
        # traces_sub = traces_sub[:100]
        # roi_ns = roi_ns[:100]

        traces_res, roi_ns_res, event_masks = self.filter_traces(traces=traces_sub, roi_ns=roi_ns,
                                                                 sample_dur=sample_dur)
        mat_corr = self.get_correlation_coefficient_matrix(traces=traces_res, event_masks=event_masks,
                                                           sample_dur=sample_dur, is_plot=False)

        mat_corr_thr = self.threshold_correlation_coefficient_matrix(mat_corr=mat_corr, is_plot=False)
        mat_dis = self.get_distance_matrix(mat_corr=mat_corr_thr, is_plot=False)
        mat_dis_dense = spatial.distance.squareform(mat_dis)
        linkage_z, c = self.hierarchy_clustering(mat_dis=mat_dis_dense, is_plot=False)

        # if self.is_cosine_similarity:
        #     mat_dis_dense = mat_dis[np.triu_indices(n=mat_dis.shape[0], k=1)]
        #     linkage_z, c = self.hierarchy_clustering(mat_dis=mat_dis_dense, is_plot=False)
        # else:
        #     linkage_z, c = self.hierarchy_clustering(mat_dis=mat_dis, is_plot=False)

        # reorganize matrix
        mat_dis_reorg = self.reorganize_matrix_by_cluster(linkage_z=linkage_z, mat=mat_dis)
        mat_corr_reorg = self.reorganize_matrix_by_cluster(linkage_z=linkage_z, mat=mat_corr)

        # get axon clusters
        axon_dict, clu_axon = self.get_axon_dict(linkage_z=linkage_z, roi_ns=roi_ns_res)

        axon_ns = list(axon_dict.keys())
        axon_ns.sort()
        roi_num_per_axon = [len(axon_dict[axon_n]) for axon_n in axon_ns]
        # print(roi_num_per_axon)

        # save data
        print('\tsaving results ...')
        save_f = h5py.File(os.path.join(save_folder, '{}_{}_axon_grouping_multiplane.hdf5'.format(date, mid)), 'a')

        meta_grp = save_f.create_group('meta')
        meta_grp.create_dataset('date', data=date)
        meta_grp.create_dataset('mouse_id', data=mid)
        meta_grp.create_dataset('trace_type', data=trace_type)
        meta_grp.create_dataset('trace_window', data=trace_window)

        bc_grp = save_f.create_group('classifier_parameters')
        for attr_n, attr in self.__dict__.items():
            bc_grp.create_dataset(attr_n, data=attr)

        save_f.create_dataset('matrix_corr_coef', data=mat_corr)
        save_f.create_dataset('matrix_corr_coef_thr', data=mat_corr_thr)
        save_f.create_dataset('matrix_distance', data=mat_dis)
        save_f.create_dataset('matrix_distance_reorg', data=mat_dis_reorg)
        save_f.create_dataset('matrix_corr_coef_thr_reorg', data=mat_corr_reorg)
        save_f.create_dataset('linkage_z', data=linkage_z)
        save_f.create_dataset('responsive_roi_ns', data=[n.encode('utf-8') for n in roi_ns_res])
        save_f.create_dataset('cluster_indices', data=clu_axon)
        axon_grp = save_f.create_group('axons')
        for axon_n, roi_lst in axon_dict.items():
            axon_grp.create_dataset(axon_n, data=[r.encode('utf-8') for r in roi_lst])

        # adding rois and traces
        axon_lst = []
        axon_traces_raw = []
        axon_traces_sub = []

        for axon_n in axon_ns:
            roi_lst = axon_dict[axon_n]

            if len(roi_lst) > 1:

                curr_trace_raw = None
                curr_trace_sub = None
                total_weight = 0.

                for plane_roi_n in roi_lst:

                    plane_n = plane_roi_n.split('_')[0]
                    roi_n = '_'.join(plane_roi_n.split('_')[1:3])

                    trace_grp = nwb_f['processing/rois_and_traces_{}/Fluorescence'.format(plane_n)]
                    seg_grp = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane'.format(plane_n)]

                    roi_i = int(roi_n[-4:])
                    curr_weight = np.sum(seg_grp[roi_n]['pix_mask_weight'])
                    total_weight = total_weight + curr_weight

                    roi_trace_raw = trace_grp['f_center_raw/data'][roi_i, :min_len] * curr_weight
                    if curr_trace_raw is None:
                        curr_trace_raw = roi_trace_raw
                    else:
                        curr_trace_raw = curr_trace_raw + roi_trace_raw

                    roi_trace_sub = trace_grp['f_center_subtracted/data'][roi_i, :min_len] * curr_weight
                    if curr_trace_sub is None:
                        curr_trace_sub = roi_trace_sub
                    else:
                        curr_trace_sub = curr_trace_sub + roi_trace_sub

                axon_lst.append(axon_n)
                axon_traces_raw.append(curr_trace_raw / total_weight)
                axon_traces_sub.append(curr_trace_sub / total_weight)

        axon_traces_raw = np.array(axon_traces_raw)
        axon_traces_sub = np.array(axon_traces_sub)
        rat_grp = save_f.create_group('rois_and_traces')
        rat_grp.attrs['description'] = 'this group only list axons with more than one rois'
        rat_grp.create_dataset('axon_list', data=[a.encode('utf-8') for a in axon_lst])
        rat_grp.create_dataset('traces_center_raw', data=axon_traces_raw)
        rat_grp.create_dataset('traces_center_subtracted', data=axon_traces_sub)
        save_f.close()

        # plot matrices
        sup_title_mat = '{}_{}, {}, dis_thr={:.2f}'.format(date, mid, trace_window, self.distance_thr)

        f_mat = self.plot_matrices(mat_corr=mat_corr, mat_corr_thr=mat_corr_thr, mat_dis=mat_dis,
                                   mat_corr_reorg=mat_corr_reorg, mat_dis_reorg=mat_dis_reorg,
                                   linkage_z=linkage_z, roi_num_per_axon=roi_num_per_axon,
                                   distance_thr=self.distance_thr, sup_title=sup_title_mat,
                                   is_truncate=False)

        f_mat.savefig(os.path.join(save_folder,
                                   '{}_{}_{}_clustering.pdf'.format(date, mid, trace_window)))

        plt.close(f_mat)

        # plot contours
        title_contour = '{}_{}, {}, dis_thr={:.2f}'.format(date, mid, trace_window, self.distance_thr)
        f_contour = self.plot_countours_multiplane(nwb_f=nwb_f, plane_ns=plane_ns, axon_dict=axon_dict,
                                                   title_str=title_contour)

        f_contour.savefig(os.path.join(save_folder,
                                       '{}_{}_{}_axon_contours.pdf'.format(date, mid, trace_window)))

        plt.close(f_contour)

        # plot axon traces
        save_path_trace = os.path.join(save_folder, '{}_{}_{}_axon_traces.pdf'.format(date,
                                                                                      mid,
                                                                                      trace_window))

        self.plot_traces_multiplane(save_path=save_path_trace, axon_dict=axon_dict, nwb_f=nwb_f,
                                    roi_num_per_axon=roi_num_per_axon, mat_corr_reorg=mat_corr_reorg,
                                    mat_dis_reorg=mat_dis_reorg, event_masks=event_masks, clu_axon=clu_axon,
                                    win_mask=win_mask, axon_lst=axon_lst, trace_type=trace_type,
                                    axon_traces_raw=axon_traces_raw, axon_traces_sub=axon_traces_sub,
                                    is_normalize_traces=is_normalize_traces)

    @staticmethod
    def plot_countours_multiplane(nwb_f, plane_ns, axon_dict, title_str=None):

        print('\tplot axon contours ...')

        f, ax = plt.subplots(1, len(plane_ns), figsize=(4 * len(plane_ns) + 1, 4))
        if title_str is not None:
            f.suptitle(title_str)

        for plane_i, plane_n in enumerate(plane_ns):
            curr_ax = ax[plane_i]
            curr_ax.set_xticks([])
            curr_ax.set_yticks([])
            curr_ax.set_title(plane_n)

            bg_img = get_background_img(nwb_f=nwb_f, plane_n=plane_n)

            if bg_img is not None:
                curr_ax.imshow(ia.array_nor(bg_img), vmin=0, vmax=0.8, cmap='gray', interpolation='nearest')
            else:
                curr_ax.imshow(np.zeros(512, 512), vmin=0, vmax=1, cmap='gray', interpolation='nearest')
            curr_ax.set_title(plane_n)

        for axon_n, roi_lst in axon_dict.items():

            if len(roi_lst) == 1:
                plane_n = roi_lst[0].split('_')[0]
                roi_n = '_'.join(roi_lst[0].split('_')[1:3])
                curr_roi = get_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
                plane_i = plane_ns.index(plane_n)
                pt.plot_mask_borders(curr_roi.get_binary_mask(), plotAxis=ax[plane_i], color='#aaaaaa',
                                     borderWidth=0.5)
            elif len(roi_lst) > 1:
                curr_color = pt.random_color(1)
                for plane_roi_n in roi_lst:
                    plane_n = plane_roi_n.split('_')[0]
                    roi_n = '_'.join(plane_roi_n.split('_')[1:3])
                    curr_roi = get_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
                    plane_i = plane_ns.index(plane_n)
                    pt.plot_mask_borders(curr_roi.get_binary_mask(), plotAxis=ax[plane_i], color=curr_color[0],
                                         borderWidth=0.5)
            else:
                raise ValueError('no roi in the axon.')

        return f

    def plot_traces_multiplane(self, save_path, axon_dict, nwb_f, roi_num_per_axon, mat_corr_reorg,
                               mat_dis_reorg, event_masks, clu_axon, win_mask, axon_lst, trace_type,
                               axon_traces_raw, axon_traces_sub, is_normalize_traces):

        # trace_f = PdfPages(os.path.join(save_folder,
        #                                 '{}_{}_{}_{}_axon_traces.pdf'.format(date, mid, plane_n, trace_window)))

        print('\tplot axon traces ...')

        trace_f = PdfPages(save_path)

        axon_ns = list(axon_dict.keys())
        axon_ns.sort()

        for axon_n in axon_ns:

            roi_lst = axon_dict[axon_n]

            if len(roi_lst) > 1:

                f = plt.figure(figsize=(10, 15), tight_layout=True)
                axon_int = int(axon_n[-4:])

                # get the mean correlation coefficient for an axon
                roi_ind_start = int(np.sum(roi_num_per_axon[0: axon_int]))
                roi_num = len(roi_lst)
                mean_corr = np.mean(mat_corr_reorg[roi_ind_start: roi_ind_start + roi_num,
                                    roi_ind_start: roi_ind_start + roi_num].flat)

                mean_dis = np.mean(mat_dis_reorg[roi_ind_start: roi_ind_start + roi_num,
                                   roi_ind_start: roi_ind_start + roi_num].flat)

                f.suptitle(
                    '{}: {} rois; mean corr coef: {:4.2f}; mean distance: {:6.4f}'.format(axon_n, len(roi_lst),
                                                                                          mean_corr,
                                                                                          mean_dis))

                axon_event_masks = event_masks[clu_axon == axon_int]

                traces_p = []
                for plane_roi_n in roi_lst:
                    plane_n = plane_roi_n.split('_')[0]
                    roi_n = '_'.join(plane_roi_n.split('_')[1:3])
                    trace, _ = get_single_trace(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n, trace_type=trace_type)
                    traces_p.append(trace[:len(win_mask)])

                traces_p = np.array(traces_p)
                traces_p = traces_p[:, win_mask]

                axon_plot_ind = axon_lst.index(axon_n)
                if trace_type == 'f_center_raw':
                    trace_center = axon_traces_raw[axon_plot_ind, :]
                elif trace_type == 'f_center_subtracted':
                    trace_center = axon_traces_sub[axon_plot_ind, :]
                else:
                    raise LookupError("do not under stand 'trace_type' ({}). Should be "
                                      "'f_center_raw' or 'f_center_subtracted'.".format(trace_type))

                self.plot_chunked_traces_with_intervals(traces_p,
                                                        trace_center=trace_center,
                                                        event_masks=axon_event_masks,
                                                        chunk_num=10,
                                                        fig_obj=f,
                                                        lw=0.5,
                                                        is_normalize_traces=is_normalize_traces)

                trace_f.savefig(f)
                plt.close(f)

        trace_f.close()

    @staticmethod
    def add_axon_strf_multiplane(nwb_f, clu_f, t_win, verbose=False):
        """
        This is a very high level function to add spatial temporal receptive field of grouped axons into saved
        h5 log files. This is for the resulting file from multiplane clustering by "process_file" method

        :param nwb_f: h5py.File object. Should be the same nwb file on which clustering was perfomed.
        :param clu_f: h5py.File object. Saved clustering file, generated by the 'process_plane()' function.
        :param t_win: list/tuple of two floats, peri-event time window to extract strf
        :param verbose: bool
        :return:
        """

        def get_sta(arr, arr_ts, trigger_ts, frame_start, frame_end):
            sta_arr = []

            for trig in trigger_ts:
                trig_ind = ta.find_nearest(arr_ts, trig)

                if trig_ind + frame_end < arr.shape[1]:
                    curr_sta = arr[:, (trig_ind + frame_start): (trig_ind + frame_end)]
                    # print(curr_sta.shape)
                    sta_arr.append(curr_sta.reshape((curr_sta.shape[0], 1, curr_sta.shape[1])))

            sta_arr = np.concatenate(sta_arr, axis=1)
            return sta_arr

        stim_ns = nwb_f['analysis/photodiode_onsets'].keys()
        lsn_stim_n = [n for n in stim_ns if 'LocallySparseNoiseRetinotopicMapping' in n]
        if len(lsn_stim_n) == 0:
            print('\tno locally sparse noise data, skip.')
            return
        elif len(lsn_stim_n) > 1:
            raise ('more than one locally sparse noise stimuli found.')
        else:
            lsn_stim_n = lsn_stim_n[0]

        if t_win[0] >= t_win[1]:
            raise ValueError('time window should be from early time to late time.')

        probe_onsets_path = 'analysis/photodiode_onsets/{}'.format(lsn_stim_n)
        probe_ns = list(nwb_f[probe_onsets_path].keys())
        probe_ns.sort()

        plane_ns = get_plane_ns(nwb_f)
        plane_n_mid = plane_ns[len(plane_ns) // 2]
        trace_ts = nwb_f['processing/motion_correction/MotionCorrection/{}/corrected/timestamps'.format(plane_n_mid)]

        traces = {}
        for f_type in clu_f['rois_and_traces'].keys():
            if f_type[0:7] == 'traces_':
                if clu_f['rois_and_traces/{}'.format(f_type)].shape[0] == 0:
                    print('no clustered axons, skip')
                    return
                else:
                    traces.update({f_type: clu_f['rois_and_traces/{}'.format(f_type)][()]})

        frame_dur = np.mean(np.diff(trace_ts))
        frame_start = int(np.floor(t_win[0] / frame_dur))
        frame_end = int(np.ceil(t_win[1] / frame_dur))
        t_axis = np.arange(frame_end - frame_start) * frame_dur + (frame_start * frame_dur)

        strf_grp = clu_f.create_group('strf_{}'.format(lsn_stim_n))
        strf_plane_grp = strf_grp.create_group('multiplane')

        strf_plane_grp.attrs['sta_timestamps'] = t_axis

        for probe_i, probe_n in enumerate(probe_ns):

            if verbose:
                print('\tprocessing probe {} / {}'.format(probe_i + 1, len(probe_ns)))

            onsets_probe_grp = nwb_f['{}/{}'.format(probe_onsets_path, probe_n)]

            curr_probe_grp = strf_plane_grp.create_group(probe_n)

            probe_onsets = onsets_probe_grp['pd_onset_ts_sec'][()]

            curr_probe_grp['global_trigger_timestamps'] = nwb_f['/{}/{}/pd_onset_ts_sec'
                .format(probe_onsets_path, probe_n)][()]
            curr_probe_grp.attrs['sta_traces_dimenstion'] = 'roi x trial x timepoint'

            for trace_n, trace in traces.items():
                sta = get_sta(arr=trace, arr_ts=trace_ts, trigger_ts=probe_onsets, frame_start=frame_start,
                              frame_end=frame_end)
                # curr_probe_grp.create_dataset('sta_' + trace_n, data=sta, compression='lzf')
                curr_probe_grp.create_dataset('sta_' + trace_n, data=sta)

    @staticmethod
    def add_axon_dgcrm_multiplane(nwb_f, clu_f, t_win, verbose=False):
        """
        This is a very high level function to add drifting grating response matrix of grouped axons into saved
        h5 log files. This is for the resulting file from multiplane clustering by "process_file" method

        :param nwb_f: h5py.File object. Should be the same nwb file on which clustering was perfomed.
        :param clu_f: h5py.File object. Saved clustering file, generated by the 'process_plane()' function.
        :param plane_n: str
        :param t_win: list/tuple of two floats, peri-event time window to extract dgcrm
        :param verbose: bool
        :return:
        """

        def get_sta(arr, arr_ts, trigger_ts, frame_start, frame_end):
            sta_arr = []

            for trig in trigger_ts:
                trig_ind = ta.find_nearest(arr_ts, trig)

                if trig_ind + frame_end < arr.shape[1]:
                    curr_sta = arr[:, (trig_ind + frame_start): (trig_ind + frame_end)]
                    # print(curr_sta.shape)
                    sta_arr.append(curr_sta.reshape((curr_sta.shape[0], 1, curr_sta.shape[1])))

            sta_arr = np.concatenate(sta_arr, axis=1)
            return sta_arr

        if t_win[0] >= t_win[1]:
            raise ValueError('time window should be from early time to late time.')

        stim_ns = nwb_f['analysis/photodiode_onsets'].keys()
        dgc_stim_n = [n for n in stim_ns if 'DriftingGratingCircleRetinotopicMapping' in n]
        if len(dgc_stim_n) == 0:
            print('\tno drifting grating circle data, skip.')
            return
        elif len(dgc_stim_n) > 1:
            raise ('more than one drifting grating cricle stimuli found.')
        else:
            dgc_stim_n = dgc_stim_n[0]

        grating_onsets_path = 'analysis/photodiode_onsets/{}'.format(dgc_stim_n)
        grating_ns = list(nwb_f[grating_onsets_path].keys())
        grating_ns.sort()
        # print('\n'.join(grating_ns))

        plane_ns = get_plane_ns(nwb_f)
        plane_n_mid = plane_ns[len(plane_ns) // 2]
        trace_ts = nwb_f['processing/motion_correction/MotionCorrection/{}/corrected/timestamps'.format(plane_n_mid)]

        traces = {}
        for f_type in clu_f['rois_and_traces'].keys():
            if f_type[0:7] == 'traces_':
                if clu_f['rois_and_traces/{}'.format(f_type)].shape[0] == 0:
                    print('no clustered axons, skip')
                    return
                else:
                    traces.update({f_type: clu_f['rois_and_traces/{}'.format(f_type)][()]})

        dgcrm_grp = clu_f.create_group('response_table_{}'.format(dgc_stim_n))
        dgcrm_plane_grp = dgcrm_grp.create_group('multiplane')

        frame_dur = np.mean(np.diff(trace_ts))
        frame_start = int(np.floor(t_win[0] / frame_dur))
        frame_end = int(np.ceil(t_win[1] / frame_dur))
        t_axis = np.arange(frame_end - frame_start) * frame_dur + (frame_start * frame_dur)
        dgcrm_plane_grp.attrs['sta_timestamps'] = t_axis

        for grating_i, grating_n in enumerate(grating_ns):

            if verbose:
                print('\tprocessing grating {} / {}'.format(grating_i + 1, len(grating_ns)))

            onsets_grating_grp = nwb_f['{}/{}'.format(grating_onsets_path, grating_n)]

            curr_grating_grp = dgcrm_plane_grp.create_group(grating_n)

            grating_onsets = onsets_grating_grp['pd_onset_ts_sec'][()]

            curr_grating_grp.attrs['global_trigger_timestamps'] = grating_onsets
            curr_grating_grp.attrs['sta_traces_dimenstion'] = 'roi x trial x timepoint'

            for trace_n, trace in traces.items():
                sta = get_sta(arr=trace, arr_ts=trace_ts, trigger_ts=grating_onsets, frame_start=frame_start,
                              frame_end=frame_end)
                # curr_grating_grp.create_dataset('sta_' + trace_n, data=sta, compression='lzf')
                curr_grating_grp.create_dataset('sta_' + trace_n, data=sta)


class BulkPaperFunctions(object):
    """
    for the bulk paper the cell classification are as follows:

    for the rois with both RF and DGC measurement
    there are 6 types in total:
    ("RF" means significant RF, "DGC" means significant DGC response
    "DS" means direction selective)


    1. RFnDGC
    2. RFDGCnDS
    3. RFDS
    4. nRFnDGC
    5. nRFDGCnDS
    6. nRFDS

    in which:

    3 and 6 are combined into "DS" group
    1 and 2 are combined into "RFnDS" group
    5 is defined as "nRFnDS" group
    4 is excluded from the study
    """

    def __init__(self):
        pass

    @staticmethod
    def get_dataframe_has_dgc(df):
        """
        return a subset dataframe of input dataframe the has dgc measurement
        """
        return df.dropna(axis=0, how='any', subset=['dgc_pos_peak_z'])

    @staticmethod
    def get_dataframe_has_rf(df):
        """
        return a subset dataframe of input dataframe the has dgc measurement
        """
        return df.dropna(axis=0, how='all', subset=['rf_pos_on_peak_z', 'rf_pos_off_peak_z'])

    def get_dataframe_rf(self, df, response_dir='pos', rf_z_thr_abs=1.6):
        """
        return two subsets of the input df
            1. rows that have significant rf response
            2. rows that do not have significant rf response

        rows that do not have rf measurement will be excluded
        """

        df_has_rf = self.get_dataframe_has_rf(df=df)

        df_rf = df_has_rf[(df_has_rf['rf_{}_on_peak_z'.format(response_dir)] >= rf_z_thr_abs) |
                          (df_has_rf['rf_{}_off_peak_z'.format(response_dir)] >= rf_z_thr_abs)]

        df_nrf = df_has_rf[(df_has_rf['rf_{}_on_peak_z'.format(response_dir)] < rf_z_thr_abs) &
                           (df_has_rf['rf_{}_off_peak_z'.format(response_dir)] < rf_z_thr_abs)]

        assert(len(df_has_rf) == len(df_rf) + len(df_nrf))

        return df_rf, df_nrf

    def get_dataframe_rf_type(self, df, response_dir='pos', rf_z_thr_abs=1.6):
        """
        return three subsets of the input df
            1. rows that have S1 ON receptive field
            2. rows that have S1 OFF receptive field
            3. rows that have S2 receptive field
        """

        df_rf, _ = self.get_dataframe_rf(df=df, response_dir=response_dir, rf_z_thr_abs=rf_z_thr_abs)

        df_s1on = df_rf[(df_rf['rf_{}_on_peak_z'.format(response_dir)] >= rf_z_thr_abs) &
                        (df_rf['rf_{}_off_peak_z'.format(response_dir)] < rf_z_thr_abs)]

        df_s1off = df_rf[(df_rf['rf_{}_on_peak_z'.format(response_dir)] < rf_z_thr_abs) &
                         (df_rf['rf_{}_off_peak_z'.format(response_dir)] >= rf_z_thr_abs)]

        df_s2 = df_rf[(df_rf['rf_{}_on_peak_z'.format(response_dir)] >= rf_z_thr_abs) &
                      (df_rf['rf_{}_off_peak_z'.format(response_dir)] >= rf_z_thr_abs)]

        assert(len(df_rf) == len(df_s1on) + len(df_s1off) + len(df_s2))

        return df_s1on, df_s1off, df_s2

    def get_dataframe_dgc(self, df, response_dir='pos', response_type='dff', dgc_peak_z_thr=3.,
                          dgc_p_anova_thr=0.01):
        """
        return two subsets of the input df
            1. rows that have significant dgc response
            2. rows that do not have significant dgc response

        rows that do not have dgc measurement will be excluded
        """

        df_has_dgc = self.get_dataframe_has_dgc(df=df)

        df_dgc = df_has_dgc[(df_has_dgc['dgc_{}_peak_z'.format(response_dir)] >= dgc_peak_z_thr) &
                            (df_has_dgc['dgc_p_anova_{}'.format(response_type)] <= dgc_p_anova_thr)]

        df_ndgc = df_has_dgc[(df_has_dgc['dgc_{}_peak_z'.format(response_dir)] < dgc_peak_z_thr) |
                            (df_has_dgc['dgc_p_anova_{}'.format(response_type)] > dgc_p_anova_thr)]

        assert(len(df_has_dgc) == len(df_dgc) + len(df_ndgc))

        return df_dgc, df_ndgc

    def get_dataframe_ds(self, df, response_dir='pos', response_type='dff', dgc_peak_z_thr=3.,
                         dgc_p_anova_thr=0.01, post_process_type='ele', dsi_type='gdsi', dsi_thr=0.5):
        """
        return two subsets of the input df
            1. rows that have significant dgc response and that are also direction selective
            2. rows that have significant dgc response but not direction selective

        rows that do not have dgc measurement or do not have significant dgc response will be excluded
        """

        df_dgc, _ = self.get_dataframe_dgc(df=df, response_dir=response_dir, response_type=response_type,
                                           dgc_peak_z_thr=dgc_peak_z_thr, dgc_p_anova_thr=dgc_p_anova_thr)

        df_dgcds = df_dgc[df_dgc['dgc_{}_{}_{}_{}'.format(response_dir,
                                                          dsi_type,
                                                          post_process_type,
                                                          response_type)] >= dsi_thr]

        df_dgcnds = df_dgc[df_dgc['dgc_{}_{}_{}_{}'.format(response_dir,
                                                           dsi_type,
                                                           post_process_type,
                                                           response_type)] <= dsi_thr]

        assert(len(df_dgc) == len(df_dgcds)+ len(df_dgcnds))

        return df_dgcds, df_dgcnds

    def get_dataframe_all_groups(self, df, response_dir='pos', rf_z_thr_abs=1.6, response_type='dff',
                                 dgc_peak_z_thr=3., dgc_p_anova_thr=0.01, post_process_type='ele', dsi_type='gdsi',
                                 dsi_thr=0.5):
        """
        return 6 subsets of the input df
            1. rfndgc
            2. rfdgcds
            3. rfdgcnds
            4. nrfndgc
            5. nrfdgcds
            6. nrfdgcnds

        rows that do not have rf measurement or do not have dgc measurement will be excluded
        """

        df_has_rf = self.get_dataframe_has_rf(df=df)
        df_has_rfdgc = self.get_dataframe_has_dgc(df=df_has_rf)

        df_rf, df_nrf = self.get_dataframe_rf(df=df_has_rfdgc, response_dir=response_dir, rf_z_thr_abs=rf_z_thr_abs)

        df_rfdgc, df_rfndgc = self.get_dataframe_dgc(df=df_rf, response_dir=response_dir, response_type=response_type,
                                                     dgc_peak_z_thr=dgc_peak_z_thr, dgc_p_anova_thr=dgc_p_anova_thr)

        df_rfdgcds, df_rfdgcnds = self.get_dataframe_ds(df=df_rfdgc, response_dir=response_dir,
                                                        response_type=response_type, dgc_peak_z_thr=dgc_peak_z_thr,
                                                        dgc_p_anova_thr=dgc_p_anova_thr,
                                                        post_process_type=post_process_type, dsi_type=dsi_type,
                                                        dsi_thr=dsi_thr)

        df_nrfdgc, df_nrfndgc = self.get_dataframe_dgc(df=df_nrf, response_dir=response_dir,
                                                       response_type=response_type,
                                                       dgc_peak_z_thr=dgc_peak_z_thr, dgc_p_anova_thr=dgc_p_anova_thr)

        df_nrfdgcds, df_nrfdgcnds = self.get_dataframe_ds(df=df_nrfdgc, response_dir=response_dir,
                                                          response_type=response_type, dgc_peak_z_thr=dgc_peak_z_thr,
                                                          dgc_p_anova_thr=dgc_p_anova_thr,
                                                          post_process_type=post_process_type, dsi_type=dsi_type,
                                                          dsi_thr=dsi_thr)

        assert(len(df_has_rfdgc) == len(df_rfndgc) + len(df_rfdgcds) + len(df_rfdgcnds) + len(df_nrfndgc) +
               len(df_nrfdgcds) + len(df_nrfdgcnds))

        return df_rfdgcds, df_rfdgcnds, df_rfndgc, df_nrfdgcds, df_nrfdgcnds, df_nrfndgc

    def get_dataframe_final_groups(self, df, response_dir='pos', rf_z_thr_abs=1.6, response_type='dff',
                                   dgc_peak_z_thr=3., dgc_p_anova_thr=0.01, post_process_type='ele', dsi_type='gdsi',
                                   dsi_thr=0.5):
        """
        return 4 subsets of the input df (these will be final groups used in the paper)
            1. nrfdgcds --> DSnRF group
            2. rfdgcnds + rfndgc --> RFnDS group
            3. rfdgcds --> RFDS group
            4. nrfdgcnds --> nRFnDS group

        rows that do not have rf measurement or do not have dgc measurement will be excluded

        nrfndgc group will also be excluded
        """

        _ = self.get_dataframe_all_groups(df=df, response_dir=response_dir, rf_z_thr_abs=rf_z_thr_abs,
                                          response_type=response_type, dgc_peak_z_thr=dgc_peak_z_thr,
                                          dgc_p_anova_thr=dgc_p_anova_thr, post_process_type=post_process_type,
                                          dsi_type=dsi_type, dsi_thr=dsi_thr)

        df_rfdgcds, df_rfdgcnds, df_rfndgc, df_nrfdgcds, df_nrfdgcnds, df_nrfndgc = _

        df_DSnRF = df_nrfdgcds
        df_RFnDS = pd.concat([df_rfdgcnds, df_rfndgc])
        df_RFDS = df_rfdgcds
        df_nRFnDS = df_nrfdgcnds

        return df_DSnRF, df_RFnDS, df_RFDS, df_nRFnDS

    @staticmethod
    def break_into_planes(df):
        plane_dfs = []

        planes = df[['date', 'mouse_id', 'plane_n']].drop_duplicates().reset_index()

        for plane_i, plane_row in planes.iterrows():
            date = plane_row['date']
            mid = plane_row['mouse_id']
            plane_n = plane_row['plane_n']

            plane_df = df[(df['date'] == date) &
                          (df['mouse_id'] == mid) &
                          (df['plane_n'] == plane_n)]

            if len(plane_df) > 0:
                plane_dfs.append(plane_df)

        return plane_dfs

    @staticmethod
    def break_into_volumes(df):
        vol_dfs = []

        vols = df['vol_n'].drop_duplicates().reset_index()

        for vol_i, vol_row in vols.iterrows():
            vol_df = df[df['vol_n'] == vol_row['vol_n']]

            if len(vol_df) > 0:
                vol_dfs.append(vol_df)

        return vol_dfs


if __name__ == '__main__':

    # ===================================================================================================
    nwb_f = h5py.File(r"G:\bulk_LGN_database\nwbs\190221_M426525_110_repacked.nwb", 'r')
    clu_f = h5py.File(r"G:\bulk_LGN_database\intermediate_results\bouton_clustering"
                      r"\AllStimuli_DistanceThr_1.30\190221_M426525_plane0_axon_grouping.hdf5", 'r')
    plane_n = 'plane0'
    axon_n = 'axon_0003'
    roi_s = get_axon_roi(clu_f=clu_f, nwb_f=nwb_f, plane_n=plane_n, axon_n=axon_n)
    scope = get_scope(nwb_f=nwb_f)
    roi_t = get_normalized_binary_roi(roi=roi_s, scope=scope, canvas_size=300., pixel_res=600, is_center=True)

    f = plt.figure(figsize=(10, 4))
    ax1 = f.add_subplot(121)
    ax1.imshow(roi_s.get_binary_mask(), interpolation='nearest', cmap='gray', vmax=1, vmin=0)
    ax2 = f.add_subplot(122)
    ax2.imshow(roi_t.get_binary_mask(), interpolation='nearest', cmap='gray', vmax=1, vmin=0)
    plt.show()
    # ===================================================================================================


    # ===================================================================================================
    # nwb_f = h5py.File(r"G:\bulk_LGN_database\nwbs\190221_M426525_110_repacked.nwb", 'r')
    # clu_f = h5py.File(r"G:\bulk_LGN_database\intermediate_results\bouton_clustering"
    #                   r"\AllStimuli_DistanceThr_1.30\190221_M426525_plane0_axon_grouping.hdf5", 'r')
    # plane_n = 'plane0'
    # axon_n = 'axon_0007'
    # axon_morph = get_axon_morphology(clu_f=clu_f, nwb_f=nwb_f, plane_n=plane_n, axon_n=axon_n)
    #
    # keys = axon_morph.keys()
    # keys.sort()
    # for key in keys:
    #     print('{}: {}'.format(key, axon_morph[key]))
    # ===================================================================================================

    # ===================================================================================================
    # nwb_f = h5py.File(r"G:\bulk_LGN_database\nwbs\190404_M439939_110_repacked.nwb")
    # uc_inds, _ = get_UC_ts_mask(nwb_f=nwb_f, plane_n='plane0')
    # plt.plot(uc_inds)
    # plt.show()
    #
    # dgc_spont_inds, _ = get_DGC_spont_ts_mask(nwb_f=nwb_f, plane_n='plane0')
    # plt.plot(dgc_spont_inds)
    # plt.show()
    # ===================================================================================================

    # ===================================================================================================
    # nwb_f = h5py.File(r"Z:\chandelier_cell_project\M447219\2019-06-25-deepscope\190625_M447219_110.nwb", 'r')
    # triplets = get_roi_triplets(nwb_f=nwb_f, overlap_ratio=0.9)
    # print('\n'.join([str(t) for t in triplets]))
    # nwb_f.close()
    # ===================================================================================================

    # ===================================================================================================
    # nwb_path = r"F:\data2\chandelier_cell_project\M455115\2019-06-06-deepscope\190606_M455115_110.nwb"
    # nwb_f = h5py.File(nwb_path, 'r')
    # pupil_area, pupil_ts = get_pupil_area(nwb_f=nwb_f,
    #                                       module_name='eye_tracking_right',
    #                                       ell_thr=0.5,
    #                                       median_win=3.)
    # plt.figure(figsize=(20, 5))
    # plt.plot(pupil_ts, pupil_area)
    # plt.show()
    # ===================================================================================================

    # ===================================================================================================
    # nwb_path = r"F:\data2\chandelier_cell_project\M441626\2019-03-26-deepscope\190326_M441626_110.nwb"
    # nwb_path = r"G:\repacked\190326_M439939_110_repacked.nwb"
    # nwb_path = r"F:\data2\rabies_tracing_project\M439939\2019-04-03-2p\190403_M439939_110.nwb"
    # nwb_path = r"/media/nc-ophys/Jun/bulk_LGN_database/nwbs/190508_M439939_110_repacked.nwb"
    # plane_n = 'plane0'
    # roi_n = 'roi_0000'
    # nwb_f = h5py.File(nwb_path, 'r')
    #
    # roi_properties, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
    #     get_everything_from_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
    #
    # keys = roi_properties.keys()
    # keys.sort()
    # for key in keys:
    #     print('{}: {}'.format(key, roi_properties[key]))
    #
    # roi_page_report(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
    #
    # nwb_f.close()
    # plt.show()
    # ===================================================================================================

    #===================================================================================================
    # coords_roi = np.array([[50, 60], [100, 200], [300, 400]])
    # coords_rf = np.array([[0., 35.], [10., 70.], [0., 70.]])
    # f = plt.figure()
    # ax_alt = f.add_subplot(121)
    # ax_azi = f.add_subplot(122)
    # plot_roi_retinotopy(coords_roi=coords_roi, coords_rf=coords_rf, ax_alt=ax_alt, ax_azi=ax_azi,
    #                     cmap='viridis', canvas_shape=(512, 512), edgecolors='#000000', linewidths=0.5)
    # plt.show()
    # ===================================================================================================

    print('for debug ...')