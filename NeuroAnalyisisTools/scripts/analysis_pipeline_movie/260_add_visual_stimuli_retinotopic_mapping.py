import os
import retinotopic_mapping.DisplayLogAnalysis as dla
import corticalmapping.NwbTools as nt

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = nt.RecordedFile(nwb_fn)

stim_pkl_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.pkl'][0]
stim_log = dla.DisplayLogAnalyzer(stim_pkl_fn)

nwb_f.add_visual_display_log_retinotopic_mapping(stim_log=stim_log)
nwb_f.close()
