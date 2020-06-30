import os
import cv2
import NeuroAnalysisTools.PreprocessPipeline as pp

def run():
    date = '200625'
    mid = '513381'
    scope = 'sutter' # 'sutter', 'deepscope' or 'scientifica'
    plane_num = 1
    plane_depths = [361]
    zoom = 4
    file_identifier_2p = '110_LSNDGCUC'
    experimenter = 'Jun'
    genotype = 'Vipr2-IRES2-Cre-neo'
    sex = 'female'
    age = '204'
    indicator = 'GCaMP6s'
    imaging_rate = '30'
    imaging_location = 'visual cortex'

    vasmap_2p_chs = ['green', 'red']
    movie_2p_chs = ['green', 'red']
    reference_ch_n = 'red'
    apply_ch_n = 'green'
    online_td_rate = 5 # movie temporal downsample rate during recording
    reorg_td_rate = 1 # movie temporal downsample rate during reorganization
    post_correction_td_rate = 1 # movie temporal downsample rate after motion correction (usually 1)
    movie_save_frames_per_file = 500 # frames per file after reorganization
    movie_low_threshold = -500. # threshold for minimum value of the movie
    mc_process_num = 6
    pd_filter_size = 0.01
    pd_segment_thr = 0.02
    pd_smallest_interval = 0.05

    pd_color_thr = -0.5 # photodiode onset color threshold for analyze display delay, [-1, 1]
    ccg_t_range = (0, 0.1)
    ccg_bins = 100
    is_plot_ccg = True

    et_side = 'right'
    et_confidence_thr = 0.7
    et_point_num_thr = 11
    et_ellipse_fit_function = cv2.fitEllipse
    et_is_generate_labeled_movie = True
    et_diagonal_length = 9.0

    roi_filter_sigma = 0. # float, pixel, gaussian filter of original caiman results, for bouton use 0.
    roi_cut_thr = 0.1 # [0., 1.], low for more rois, high for less rois
    roi_margin_pix_num = 5. # int, pixel, margin to add from motion correction edge.
    roi_area_range = [5, 500] # pixel, min and max areas allowed for valid rois
    roi_overlap_thr = 0.2 # [0., 1.], the smaller rois with overlap ratio larger than this value will be discarded
    roi_surround_limit = [1, 8] # pixel, inner and outer edge of surround donut.
    roi_chunk_size = 2000 # int, number of frames for multiprocessing
    roi_process_num = 4 # int, number of processes
    roi_lambda = 0.5 # float, lambda for neuropil subtraction
    roi_plot_chunk_size = 5000 # int, traces plot chunk size
    roi_plot_process_num = 4 # int, trace plotting process number
    roi_label_movie_downsample_rate = 10 # int, downsample rate
    roi_label_movie_process_num = 4 # int, process number for downsampling movie
    roi_label_movie_chunk_size = 1000  # chunk frame size for each process
    roi_label_movie_frame_size = 8 # float, movie frame size, inch

    strf_t_win = [-0.5, 2.] # time window for STRF response table
    dgc_t_win = [-1., 2.5] # time window for drifting grating response table
    lsn_stim_name = '001_LocallySparseNoiseRetinotopicMapping'
    dgc_stim_name = '003_DriftingGratingCircleRetinotopicMapping'

    plot_trace_type = 'sta_f_center_subtracted'
    # if the minimum of whold trace is less than this value,
    # a constant will be added to make this value minimum
    plot_bias = 1.
    plot_rf_t_window = [0., 0.5] # seconds, time window for calculating receptive field
    plot_rf_filter_sigma = 1. # pixel, gaussian filter for receptive field
    plot_rf_interoplation_rate = 10 # int, pixel, interpolation rate for receptive
    # float, zscore, only the rf maps with a peak larger than this value will be considered to have a significant rf
    plot_rf_absolute_thr = 1.6
    # float, the ratio of the rf map peak to get the rf. if the result (rf_peak * thr_ratio)
    # is less than the absolute threshold (above), than the absolute threshold will be used instead
    plot_rf_thr_ratio = 0.4
    plot_rf_zscore_range = [-4., 4.]
    plot_dgc_t_window_response = [0., 1.5]
    plot_dgc_t_window_baseline = [-0.5, 0.]

    source_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data"

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curr_folder)

    sess_id = file_identifier_2p.split('_')[0]

    psr = pp.Preprocessor()

    if scope == 'sutter':
        session_folder = os.path.join(source_folder, '{}-M{}-2p'.format(date, mid))
        vasmap_wf_identifier = 'JCam'
        imaging_excitation_lambda = '92n nm'
        pd_digitize_thr = -0.15
        vsync_frame_path = 'acquisition/timeseries/digital_vsync_visual_rise'
        et_mov_fn_identifier = '-0.avi'
        et_nasal_dir = 'left'
        et_ts_name = 'digital_vsync_right_eye_mon_rise'
    elif scope == 'deepscope':
        session_folder = os.path.join(source_folder, '{}-M{}-deepscope'.format(date, mid))
        vasmap_wf_identifier = 'vasmap_wf'
        imaging_excitation_lambda = '940 nm'
        pd_digitize_thr = 0.15
        vsync_frame_path = 'acquisition/timeseries/digital_vsync_stim_rise'
        et_mov_fn_identifier = '-1.avi'
        et_nasal_dir = 'right'
        et_ts_name = 'digital_cam_eye_rise'
    else:
        raise LookupError('Do not understand scope type.')

    plane_ns = ['plane{:1d}'.format(i) for i in range(plane_num)]
    reorg_folder = os.path.join(session_folder, file_identifier_2p + '_reorged')

    # # copy relevant data to local folder
    # psr.copy_initial_files(session_folder, curr_folder, date=date, mouse_id=mid)

    # get 2p vasculature maps as tif
    # _ = psr.get_vasmap_2p(data_folder=os.path.join(session_folder, 'vasmap_2p'),
    #                       scope=scope,
    #                       channels=vasmap_2p_chs,
    #                       identifier='vasmap_2p',
    #                       is_equalize=False,
    #                       save_folder=curr_folder)
    #
    # # save aligned 2p vasculature maps as png
    # for vasmap_2p_ch in vasmap_2p_chs:
    #     psr.save_png_vasmap(tif_path=os.path.join(curr_folder, 'vasmap_2p_{}_rotated.tif'.format(vasmap_2p_ch)),
    #                         prefix='{}_{}'.format(date, mid), saturation_level=10)
    #
    # # get widefield vasculature maps as tif
    # _ = psr.get_vasmap_wf(data_folder=os.path.join(session_folder, 'vasmap_wf'),
    #                       save_folder=curr_folder,
    #                       scope=scope,
    #                       identifier=vasmap_wf_identifier)
    #
    # # save aligned widefield vasculature maps as png
    # psr.save_png_vasmap(tif_path=os.path.join(curr_folder, 'vasmap_wf_rotated.tif'),
    #                     prefix='{}_{}'.format(date, mid), saturation_level=5)
    #
    # # reorganized raw 2p movies
    # psr.reorganize_raw_2p_data(data_folder=os.path.join(session_folder, 'movie'),
    #                            save_folder=session_folder,
    #                            identifier=file_identifier_2p,
    #                            scope=scope,
    #                            channels=movie_2p_chs,
    #                            plane_num=plane_num,
    #                            temporal_downsample_rate=reorg_td_rate,
    #                            frames_per_file=movie_save_frames_per_file,
    #                            low_thr=movie_low_threshold)

    # # motion correction
    # psr.motion_correction(data_folder=reorg_folder,
    #                       reference_channel_name=reference_ch_n,
    #                       apply_channel_names=movie_2p_chs,
    #                       process_num=mc_process_num)
    #
    # # copy correction results to local folder
    # psr.copy_correction_results(reorg_folder, curr_folder, apply_channel_name=apply_ch_n,
    #                             reference_channel_name=reference_ch_n)

    # # downsample corrected files
    # psr.get_downsampled_small_movies(reorg_folder, curr_folder,
    #                                  identifier=file_identifier_2p,
    #                                  xy_downsample_rate=2,
    #                                  t_downsample_rate=10,
    #                                  channel_names=movie_2p_chs)
    #
    # # generate 2p data hdf5 file for nwb
    # psr.get_2p_data_file_for_nwb(reorg_folder, curr_folder, identifier=file_identifier_2p,
    #                              file_prefix='{}_M{}_{}'.format(date, mid, sess_id))
    #
    # # get .mmap file for caiman bouton segmentation
    # psr.get_mmap_files_for_caiman_bouton(reorg_folder,
    #                                      save_name_base='{}_M{}_{}'.format(date, mid, sess_id),
    #                                      identifier=file_identifier_2p,
    #                                      temporal_downsample_rate=5, channel_name='green')

    # # generate nwb file
    # psr.generate_nwb_file(save_folder=curr_folder, date=date, mouse_id=mid, session_id=sess_id,
    #                       experimenter=experimenter, genotype=genotype, sex=sex, age=age,
    #                       indicator=indicator, imaging_rate=imaging_rate,
    #                       imaging_depth='{} um'.format(plane_depths),
    #                       imaging_location=imaging_location, imaging_device=scope,
    #                       imaging_excitation_lambda=imaging_excitation_lambda)
    #
    # # add vasmaps to nwb file
    # psr.add_vasmap_to_nwb(nwb_folder=curr_folder)
    #
    # # adding sync data to nwb file
    # psr.add_sync_data_to_nwb(nwb_folder=curr_folder, sync_identifier='{}-M{}'.format(date, mid))
    #
    # # get photodiode onset timestamps
    # psr.get_photodiode_onset_in_nwb(nwb_folder=curr_folder, digitize_thr=pd_digitize_thr,
    #                                 filter_size=pd_filter_size, segment_thr=pd_segment_thr,
    #                                 smallest_interval=pd_smallest_interval, is_interact=True)
    #
    # # add visual stim log to nwb file
    # psr.add_visual_stimuli_to_nwb(nwb_folder=curr_folder)
    #
    # # analyze display log and photodiode signal
    # psr.analyze_visual_display_in_nwb(nwb_folder=curr_folder,
    #                                   vsync_frame_path=vsync_frame_path,
    #                                   photodiode_ts_path='analysis/digital_photodiode_rise',
    #                                   photodiode_thr=pd_color_thr, ccg_t_range=ccg_t_range,
    #                                   ccg_bins=ccg_bins, is_plot=is_plot_ccg)

    # # add eyetracking data
    # psr.add_eyetracking_to_nwb_deeplabcut(nwb_folder=curr_folder,
    #                                       eyetracking_folder=os.path.join(curr_folder, 'videomon'),
    #                                       mov_fn_identifier=et_mov_fn_identifier,
    #                                       confidence_thr=et_confidence_thr, point_num_thr=et_point_num_thr,
    #                                       ellipse_fit_function=et_ellipse_fit_function,
    #                                       is_generate_labeled_movie=et_is_generate_labeled_movie,
    #                                       side=et_side, nasal_dir=et_nasal_dir,
    #                                       diagonal_length=et_diagonal_length,
    #                                       eyetracking_ts_name=et_ts_name)
    #
    # # adding 2p imaging data
    # psr.add_2p_image_to_nwb(nwb_folder=curr_folder, image_identifier='{}_M{}_{}'.format(date, mid, sess_id),
    #                         zoom=zoom, scope=scope, plane_ns=plane_ns, plane_depths=plane_depths,
    #                         temporal_downsample_rate=online_td_rate * reorg_td_rate)
    #
    # # add motion correction module to nwb file
    # psr.add_motion_correction_module_to_nwb(nwb_folder=curr_folder,
    #                                         movie_fn='{}_M{}_{}_2p_movies.hdf5'.format(date, mid, sess_id),
    #                                         plane_num=plane_num,
    #                                         post_correction_td_rate=post_correction_td_rate)

    # ppsr = pp.PlaneProcessor()

    # # get rois from caiman segmentation results
    # for plane_n in plane_ns:
    #     plane_folder = os.path.join(curr_folder, plane_n)
    #     ppsr.get_rois_from_caiman_results(plane_folder=plane_folder, filter_sigma=roi_filter_sigma,
    #                                       cut_thr=roi_cut_thr, bg_fn='corrected_mean_projections.tif')

    # # filter rois
    # for plane_n in plane_ns:
    #     plane_folder = os.path.join(curr_folder, plane_n)
    #     ppsr.filter_rois(plane_folder=plane_folder, margin_pix_num=roi_margin_pix_num,
    #                      area_range=roi_area_range, overlap_thr=roi_overlap_thr,
    #                      bg_fn='corrected_mean_projections.tif')

    # # get center and surround rois
    # for plane_n in plane_ns:
    #     plane_folder = os.path.join(curr_folder, plane_n)
    #     ppsr.get_rois_and_surrounds(plane_folder=plane_folder, surround_limit=roi_surround_limit,
    #                                 bg_fn='corrected_mean_projections.tif')

    # # extract traces
    # for plane_n in plane_ns:
    #     plane_folder = os.path.join(curr_folder, plane_n)
    #     ppsr.get_raw_center_and_surround_traces(plane_folder=plane_folder, chunk_size=roi_chunk_size,
    #                                             process_num=roi_process_num)

    # # neuropil subtraction
    # for plane_n in plane_ns:
    #     plane_folder = os.path.join(curr_folder, plane_n)
    #     ppsr.get_neuropil_subtracted_traces(plane_folder=plane_folder, lam=roi_lambda,
    #                                         plot_chunk_size=roi_plot_chunk_size,
    #                                         plot_process_num=roi_plot_process_num)

    # # generate downsampled labeled movie
    # for plane_n in plane_ns:
    #     plane_folder = os.path.join(curr_folder, plane_n)
    #     ppsr.generate_labeled_movie(plane_folder=plane_folder,
    #                                 downsample_rate=roi_label_movie_downsample_rate,
    #                                 process_num=roi_label_movie_process_num,
    #                                 chunk_size=roi_label_movie_chunk_size,
    #                                 frame_size=roi_label_movie_frame_size)

    # # add rois and traces to nwb file
    # psr.add_rois_and_traces_to_nwb_caiman(nwb_folder=curr_folder, plane_ns=plane_ns,
    #                                       plane_depths=plane_depths)

    # # add response tables
    # psr.add_response_tables(nwb_folder=curr_folder, strf_response_window=strf_t_win,
    #                         dgc_response_window=dgc_t_win, lsn_stim_name=lsn_stim_name,
    #                         dgc_stim_name=dgc_stim_name)

    # ==================================Plotting==========================================
    # plot_strf_grp_name = 'strf_{}'.format(lsn_stim_name)
    # plot_dgc_grp_name = 'response_table_{}'.format(dgc_stim_name)

    # # plot drifting grating tuning curves
    # psr.plot_dgc_tuning_curves(nwb_folder=curr_folder, response_table_name=plot_dgc_grp_name,
    #                            t_window_response=plot_dgc_t_window_response,
    #                            t_window_baseline=plot_dgc_t_window_baseline,
    #                            trace_type=plot_trace_type, bias=plot_bias)

    # # plot receptive field contours
    # psr.plot_RF_contours(nwb_folder=curr_folder, response_table_name=plot_strf_grp_name,
    #                      trace_type=plot_trace_type, bias=plot_bias, t_window=plot_rf_t_window,
    #                      filter_sigma=plot_rf_filter_sigma, interpolation_rate=plot_rf_interoplation_rate,
    #                      absolute_thr=plot_rf_absolute_thr, thr_ratio=plot_rf_thr_ratio)

    # # plot zscore receptive field maps
    # psr.plot_zscore_RF_maps(nwb_folder=curr_folder, response_table_name=plot_strf_grp_name,
    #                         t_window=plot_rf_t_window, trace_type=plot_trace_type, bias=plot_bias,
    #                         zscore_range=plot_rf_zscore_range)

    # # plot drifting grating mean responses
    # psr.plot_dgc_mean_responses(nwb_folder=curr_folder, response_table_name=plot_dgc_grp_name,
    #                             t_window_response=plot_dgc_t_window_response,
    #                             t_window_baseline=plot_dgc_t_window_baseline,
    #                             trace_type=plot_trace_type, bias=plot_bias)

    # # plot drifting grating trial responses
    # psr.plot_dgc_trial_responses(nwb_folder=curr_folder, response_table_name=plot_dgc_grp_name,
    #                              t_window_response=plot_dgc_t_window_response,
    #                              t_window_baseline=plot_dgc_t_window_baseline,
    #                              trace_type=plot_trace_type, bias=plot_bias)

    # # plot spatial temporal receptive field
    # psr.plot_STRF(nwb_folder=curr_folder, response_table_name=plot_strf_grp_name,
    #               trace_type=plot_trace_type, bias=plot_bias)


if __name__ == '__main__':
    run()