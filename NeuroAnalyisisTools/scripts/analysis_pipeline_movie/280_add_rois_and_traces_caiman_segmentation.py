import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
import corticalmapping.NwbTools as nt
import corticalmapping.core.FileTools as ft
import corticalmapping.core.ImageAnalysis as ia

# for deepscope
# plane_ns = ['plane0', 'plane1', 'plane2']
# plane_depths = [150, 100, 50] # or [300, 250, 200]

# for single plane
plane_ns = ['plane0']
plane_depths = [150]

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

def add_rois_and_traces(data_folder, nwb_f, plane_n, imaging_depth,
                        mov_path='/processing/motion_correction/MotionCorrection'):

    mov_grp = nwb_f.file_pointer[mov_path + '/' + plane_n + '/corrected']

    data_f = h5py.File(os.path.join(data_folder, 'rois_and_traces.hdf5'), 'r')
    mask_arr_c = data_f['masks_center'].value
    mask_arr_s = data_f['masks_surround'].value
    traces_center_raw = data_f['traces_center_raw'].value
    # traces_center_demixed = data_f['traces_center_demixed'].value
    traces_center_subtracted = data_f['traces_center_subtracted'].value
    # traces_center_dff = data_f['traces_center_dff'].value
    traces_surround_raw = data_f['traces_surround_raw'].value
    neuropil_r = data_f['neuropil_r'].value
    neuropil_err = data_f['neuropil_err'].value
    data_f.close()


    if traces_center_raw.shape[1] != mov_grp['num_samples'].value:
        raise ValueError('number of trace time points ({}) does not match frame number of '
                         'corresponding movie ({}).'.format(traces_center_raw.shape[1], mov_grp['num_samples'].value))

    # traces_center_raw = traces_center_raw[:, :mov_grp['num_samples'].value]
    # traces_center_subtracted = traces_center_subtracted[:, :mov_grp['num_samples'].value]
    # traces_surround_raw = traces_surround_raw[:, :mov_grp['num_samples'].value]

    rf_img_max = tf.imread(os.path.join(data_folder, 'corrected_max_projection.tif'))
    rf_img_mean = tf.imread(os.path.join(data_folder, 'corrected_mean_projection.tif'))

    print 'adding segmentation results ...'
    rt_mo = nwb_f.create_module('rois_and_traces_' + plane_n)
    rt_mo.set_value('imaging_depth_micron', imaging_depth)
    is_if = rt_mo.create_interface('ImageSegmentation')
    is_if.create_imaging_plane('imaging_plane', description='')
    is_if.add_reference_image('imaging_plane', 'max_projection', rf_img_max)
    is_if.add_reference_image('imaging_plane', 'mean_projection', rf_img_mean)

    for i in range(mask_arr_c.shape[0]):
        curr_cen = mask_arr_c[i]
        curr_cen_n = 'roi_' + ft.int2str(i, 4)
        curr_cen_roi = ia.WeightedROI(curr_cen)
        curr_cen_pixels_yx = curr_cen_roi.get_pixel_array()
        curr_cen_pixels_xy = np.array([curr_cen_pixels_yx[:, 1], curr_cen_pixels_yx[:, 0]]).transpose()
        is_if.add_roi_mask_pixels(image_plane='imaging_plane', roi_name=curr_cen_n, desc='',
                                  pixel_list=curr_cen_pixels_xy, weights=curr_cen_roi.weights, width=512, height=512)

        curr_sur = mask_arr_s[i]
        curr_sur_n = 'surround_' + ft.int2str(i, 4)
        curr_sur_roi = ia.ROI(curr_sur)
        curr_sur_pixels_yx = curr_sur_roi.get_pixel_array()
        curr_sur_pixels_xy = np.array([curr_sur_pixels_yx[:, 1], curr_sur_pixels_yx[:, 0]]).transpose()
        is_if.add_roi_mask_pixels(image_plane='imaging_plane', roi_name=curr_sur_n, desc='',
                                  pixel_list=curr_sur_pixels_xy, weights=None, width=512, height=512)
    is_if.finalize()



    trace_f_if = rt_mo.create_interface('Fluorescence')
    seg_if_path = '/processing/rois_and_traces_' + plane_n + '/ImageSegmentation/imaging_plane'
    # print seg_if_path
    ts_path = mov_path + '/' + plane_n + '/corrected'

    print 'adding center fluorescence raw'
    trace_raw_ts = nwb_f.create_timeseries('RoiResponseSeries', 'f_center_raw')
    trace_raw_ts.set_data(traces_center_raw, unit='au', conversion=np.nan, resolution=np.nan)
    trace_raw_ts.set_value('data_format', 'roi (row) x time (column)')
    trace_raw_ts.set_value('data_range', '[-8192, 8191]')
    trace_raw_ts.set_description('fluorescence traces extracted from the center region of each roi')
    trace_raw_ts.set_time_as_link(ts_path)
    trace_raw_ts.set_value_as_link('segmentation_interface', seg_if_path)
    roi_names = ['roi_' + ft.int2str(ind, 4) for ind in range(traces_center_raw.shape[0])]
    trace_raw_ts.set_value('roi_names', roi_names)
    trace_raw_ts.set_value('num_samples', traces_center_raw.shape[1])
    trace_f_if.add_timeseries(trace_raw_ts)
    trace_raw_ts.finalize()

    print 'adding neuropil fluorescence raw'
    trace_sur_ts = nwb_f.create_timeseries('RoiResponseSeries', 'f_surround_raw')
    trace_sur_ts.set_data(traces_surround_raw, unit='au', conversion=np.nan, resolution=np.nan)
    trace_sur_ts.set_value('data_format', 'roi (row) x time (column)')
    trace_sur_ts.set_value('data_range', '[-8192, 8191]')
    trace_sur_ts.set_description('neuropil traces extracted from the surroud region of each roi')
    trace_sur_ts.set_time_as_link(ts_path)
    trace_sur_ts.set_value_as_link('segmentation_interface', seg_if_path)
    sur_names = ['surround_' + ft.int2str(ind, 4) for ind in range(traces_center_raw.shape[0])]
    trace_sur_ts.set_value('roi_names', sur_names)
    trace_sur_ts.set_value('num_samples', traces_surround_raw.shape[1])
    trace_f_if.add_timeseries(trace_sur_ts)
    trace_sur_ts.finalize()

    roi_center_n_path = '/processing/rois_and_traces_' + plane_n + '/Fluorescence/f_center_raw/roi_names'
    # print 'adding center fluorescence demixed'
    # trace_demix_ts = nwb_f.create_timeseries('RoiResponseSeries', 'f_center_demixed')
    # trace_demix_ts.set_data(traces_center_demixed, unit='au', conversion=np.nan, resolution=np.nan)
    # trace_demix_ts.set_value('data_format', 'roi (row) x time (column)')
    # trace_demix_ts.set_description('center traces after overlapping demixing for each roi')
    # trace_demix_ts.set_time_as_link(mov_path + '/' + plane_n + '/corrected')
    # trace_demix_ts.set_value_as_link('segmentation_interface', seg_if_path)
    # trace_demix_ts.set_value('roi_names', roi_names)
    # trace_demix_ts.set_value('num_samples', traces_center_demixed.shape[1])
    # trace_f_if.add_timeseries(trace_demix_ts)
    # trace_demix_ts.finalize()

    print 'adding center fluorescence after neuropil subtraction'
    trace_sub_ts = nwb_f.create_timeseries('RoiResponseSeries', 'f_center_subtracted')
    trace_sub_ts.set_data(traces_center_subtracted, unit='au', conversion=np.nan, resolution=np.nan)
    trace_sub_ts.set_value('data_format', 'roi (row) x time (column)')
    trace_sub_ts.set_description('center traces after overlap demixing and neuropil subtraction for each roi')
    trace_sub_ts.set_time_as_link(mov_path + '/' + plane_n + '/corrected')
    trace_sub_ts.set_value_as_link('segmentation_interface', seg_if_path)
    trace_sub_ts.set_value_as_link('roi_names', roi_center_n_path)
    trace_sub_ts.set_value('num_samples', traces_center_subtracted.shape[1])
    trace_sub_ts.set_value('r', neuropil_r, dtype='float32')
    trace_sub_ts.set_value('rmse', neuropil_err, dtype='float32')
    trace_sub_ts.set_comments('value "r": neuropil contribution ratio for each roi. '
                              'value "rmse": RMS error of neuropil subtraction for each roi')
    trace_f_if.add_timeseries(trace_sub_ts)
    trace_sub_ts.finalize()

    trace_f_if.finalize()

    # print 'adding global dF/F traces for each roi'
    # trace_dff_if = rt_mo.create_interface('DfOverF')
    #
    # trace_dff_ts = nwb_f.create_timeseries('RoiResponseSeries', 'dff_center')
    # trace_dff_ts.set_data(traces_center_dff, unit='au', conversion=np.nan, resolution=np.nan)
    # trace_dff_ts.set_value('data_format', 'roi (row) x time (column)')
    # trace_dff_ts.set_description('global df/f traces for each roi center, input fluorescence is the trace after demixing'
    #                              ' and neuropil subtraction. global df/f is calculated by '
    #                              'allensdk.brain_observatory.dff.compute_dff() function.')
    # trace_dff_ts.set_time_as_link(ts_path)
    # trace_dff_ts.set_value_as_link('segmentation_interface', seg_if_path)
    # trace_dff_ts.set_value('roi_names', roi_names)
    # trace_dff_ts.set_value('num_samples', traces_center_dff.shape[1])
    # trace_dff_if.add_timeseries(trace_dff_ts)
    # trace_dff_ts.finalize()
    # trace_dff_if.finalize()

    rt_mo.finalize()

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = nt.RecordedFile(nwb_fn)

for plane_i, plane_n in enumerate(plane_ns):

    print('\n\n' + plane_n)

    data_folder = os.path.join(curr_folder, plane_n)
    add_rois_and_traces(data_folder, nwb_f, plane_n, imaging_depth=plane_depths[plane_i])

nwb_f.close()


