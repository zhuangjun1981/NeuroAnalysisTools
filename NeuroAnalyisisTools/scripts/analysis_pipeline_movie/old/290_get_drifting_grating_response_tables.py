import os
import h5py
import numpy as np
import corticalmapping.NwbTools as nt

# plane_ns = ['plane0', 'plane1', 'plane2', 'plane3', 'plane4']
stim_name = '003_DriftingGratingCircleRetinotopicMapping'
t_win = [-1., 2.5]

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = nt.RecordedFile(nwb_fn)

nwb_f.get_drifting_grating_response_table_retinotopic_mapping(stim_name=stim_name, time_window=t_win)

nwb_f.close()