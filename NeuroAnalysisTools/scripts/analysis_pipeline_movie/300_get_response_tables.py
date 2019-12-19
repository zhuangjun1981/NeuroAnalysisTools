import os
import NeuroAnalysisTools.NwbTools as nt


strf_t_win = [-0.5, 2.]
dgc_t_win = [-1., 2.5]

lsn_stim_name = '001_LocallySparseNoiseRetinotopicMapping'
dgc_stim_name = '003_DriftingGratingCircleRetinotopicMapping'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = nt.RecordedFile(nwb_fn)

nwb_f.get_drifting_grating_response_table_retinotopic_mapping(stim_name=dgc_stim_name, time_window=dgc_t_win)
nwb_f.get_spatial_temporal_receptive_field_retinotopic_mapping(stim_name=lsn_stim_name, time_window=strf_t_win)

nwb_f.close()



