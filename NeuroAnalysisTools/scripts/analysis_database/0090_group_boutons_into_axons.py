import os
import numpy as np
import h5py
import corticalmapping.DatabaseTools as dt
import corticalmapping.SingleCellAnalysis as sca
import matplotlib.pyplot as plt
import pandas as pd


nwb_folder = "nwbs"
save_folder = r"intermediate_results\bouton_clustering"
trace_type = 'f_center_subtracted'
trace_window = 'UniformContrast' # 'AllStimuli', 'UniformContrast', 'LocallySparseNoise', or 'DriftingGratingSpont'

# BoutonClassifier parameters
skew_filter_sigma = 5.
skew_thr = 0.6
lowpass_sigma=0.1
detrend_sigma=3.
event_std_thr = 3.
peri_event_dur = (-1., 3.)
corr_len_thr = 30.
corr_abs_thr = 0.7
corr_std_thr = 3.
is_cosine_similarity = False
distance_metric = 'euclidean'
linkage_method = 'weighted'
distance_thr = 1.3

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

save_folder = os.path.join(save_folder, '{}_DistanceThr_{:.2f}'.format(trace_window, distance_thr))
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

nwb_fns = [f for f in os.listdir(nwb_folder) if f[-4:] == '.nwb']
nwb_fns.sort()

bc = dt.BoutonClassifier(skew_filter_sigma=skew_filter_sigma,
                         skew_thr=skew_thr,
                         lowpass_sigma=lowpass_sigma,
                         detrend_sigma=detrend_sigma,
                         event_std_thr=event_std_thr,
                         peri_event_dur=peri_event_dur,
                         corr_len_thr=corr_len_thr,
                         corr_abs_thr=corr_abs_thr,
                         corr_std_thr=corr_std_thr,
                         is_cosine_similarity=is_cosine_similarity,
                         distance_metric=distance_metric,
                         linkage_method=linkage_method,
                         distance_thr=distance_thr)

for nwb_fi, nwb_fn in enumerate(nwb_fns):

    print('processing {}, {}/{}'.format(nwb_fn, nwb_fi + 1, len(nwb_fns)))

    nwb_f = h5py.File(os.path.join(nwb_folder, nwb_fn), 'r')

    plane_ns = dt.get_plane_ns(nwb_f=nwb_f)
    plane_ns.sort()

    for plane_i, plane_n in enumerate(plane_ns):

        print('\n\t{}, {}/{}'.format(plane_n, plane_i + 1, len(plane_ns)))

        bc.process_plane(nwb_f=nwb_f, save_folder=save_folder, plane_n=plane_n, trace_type=trace_type,
                         trace_window=trace_window)

    nwb_f.close()
