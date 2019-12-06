import os
import numpy as np
import matplotlib.pyplot as plt
import corticalmapping.NwbTools as nt
import retinotopic_mapping.DisplayLogAnalysis as dla
import corticalmapping.core.TimingAnalysis as ta

# for deepscope
vsync_frame_path='acquisition/timeseries/digital_vsync_stim_rise'

# for sutter
# vsync_frame_path='acquisition/timeseries/digital_vsync_visual_rise'

pd_ts_pd_path = 'analysis/digital_photodiode_rise'
pd_thr = -0.5 # this is color threshold, not analog photodiode threshold
ccg_t_range = (0., 0.1)
ccg_bins = 100
is_plot = True

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = nt.RecordedFile(nwb_fn)

stim_pkl_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.pkl'][0]
stim_log = dla.DisplayLogAnalyzer(stim_pkl_fn)

# get display lag
display_delay = nwb_f.get_display_delay_retinotopic_mapping(stim_log=stim_log, indicator_color_thr=pd_thr,
                                                            ccg_t_range=ccg_t_range, ccg_bins=ccg_bins,
                                                            is_plot=is_plot, pd_onset_ts_path=pd_ts_pd_path,
                                                            vsync_frame_ts_path=vsync_frame_path)

# analyze photodiode onset
stim_dict = stim_log.get_stim_dict()
pd_onsets_seq = stim_log.analyze_photodiode_onsets_sequential(stim_dict=stim_dict, pd_thr=pd_thr)
pd_onsets_com = stim_log.analyze_photodiode_onsets_combined(pd_onsets_seq=pd_onsets_seq,
                                                            is_dgc_blocked=True)
nwb_f.add_photodiode_onsets_combined_retinotopic_mapping(pd_onsets_com=pd_onsets_com,
                                                         display_delay=display_delay,
                                                         vsync_frame_path=vsync_frame_path)
nwb_f.close()