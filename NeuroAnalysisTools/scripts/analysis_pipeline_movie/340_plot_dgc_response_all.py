import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import NeuroAnalysisTools.DatabaseTools as dt
import NeuroAnalysisTools.core.PlottingTools as pt

trace_type = 'f_center_subtracted'
response_table_path = 'analysis/response_table_003_DriftingGratingCircleRetinotopicMapping'

baseline_span = [-0.5, 0.]
response_span = [0., 1.]
bias = 1.

face_cmap = 'RdBu_r'

def get_dff(traces, t_axis, response_span, baseline_span):
    """

    :param traces: dimension, trial x timepoint
    :param t_axis:
    :return:
    """

    baseline_ind = np.logical_and(t_axis > baseline_span[0], t_axis <= baseline_span[1])
    response_ind = np.logical_and(t_axis > response_span[0], t_axis <= response_span[1])
    baseline = np.mean(traces[:, baseline_ind], axis=1, keepdims=True)
    dff_traces = (traces - baseline) / baseline

    trace_mean = np.mean(traces, axis=0)
    baseline_mean = np.mean(trace_mean[baseline_ind])
    dff_trace_mean = (trace_mean - baseline_mean) / baseline_mean
    dff_mean = np.mean(dff_trace_mean[response_ind])

    return dff_traces, dff_trace_mean, dff_mean


curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

save_folder = os.path.join(curr_folder, 'figures')
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
print(nwb_fn)
nwb_f = h5py.File(nwb_fn, 'r')

plane_ns = nwb_f[response_table_path].keys()
plane_ns.sort()

for plane_n in plane_ns:

    print('\nprocessing {} ...'.format(plane_n))

    res_grp = nwb_f['{}/{}'.format(response_table_path, plane_n)]
    t_axis = res_grp.attrs['sta_timestamps']

    roi_lst = nwb_f['processing/rois_and_traces_' + plane_n + '/ImageSegmentation/imaging_plane/roi_list'].value
    roi_lst = [r for r in roi_lst if r[:4] == 'roi_']
    roi_lst.sort()

    grating_ns = res_grp.keys()

    # remove blank sweep
    grating_ns = [gn for gn in grating_ns if gn[-37:] != '_sf0.00_tf00.0_dire000_con0.00_rad000']

    dire_lst = np.array(list(set([str(gn[38:41]) for gn in grating_ns])))
    dire_lst.sort()
    tf_lst = np.array(list(set([str(gn[29:33]) for gn in grating_ns])))
    tf_lst.sort()
    sf_lst = np.array(list(set([str(gn[22:26]) for gn in grating_ns])))
    sf_lst.sort()

    print('\nall directions (deg): {}'.format(dire_lst))
    print('all temporal frequencies (Hz): {}'.format(tf_lst))
    print('all spatial frequencies (dpd): {}\n'.format(sf_lst))

    pdff = PdfPages(os.path.join(save_folder, 'STA_DriftingGrating_' + plane_n + '_all.pdf'))

    for roi_i, roi_n in enumerate(roi_lst):
        print('plotting: {} ...'.format(roi_n))

        curr_trace, _ = dt.get_single_trace(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
        if np.min(curr_trace) < bias:
            add_to_trace = -np.min(curr_trace) + bias
        else:
            add_to_trace = 0.

        f = plt.figure(figsize=(8.5, 11))
        gs_out = gridspec.GridSpec(len(tf_lst), 1)
        gs_in_dict = {}
        for gs_ind, gs_o in enumerate(gs_out):
            curr_gs_in = gridspec.GridSpecFromSubplotSpec(len(sf_lst), len(dire_lst), subplot_spec=gs_o,
                                                          wspace=0.0, hspace=0.0)
            gs_in_dict[gs_ind] = curr_gs_in

        v_max = 0
        v_min = 0
        dff_mean_max=0
        dff_mean_min=0

        for grating_n in grating_ns:
            grating_grp = res_grp[grating_n]

            curr_sta = grating_grp['sta_' + trace_type].value[roi_i] + add_to_trace
            dff_traces, dff_trace_mean, dff_mean = get_dff(traces=curr_sta, t_axis=t_axis, response_span=response_span,
                                                           baseline_span=baseline_span)
            v_max = max([np.amax(dff_traces), v_max])
            v_min = min([np.amin(dff_traces), v_min])
            dff_mean_max = max([dff_mean, dff_mean_max])
            dff_mean_min = min([dff_mean, dff_mean_min])

        dff_mean_max = max([abs(dff_mean_max), abs(dff_mean_min)])
        dff_mean_min = - dff_mean_max


        for grating_n in grating_ns:
            grating_grp = res_grp[grating_n]

            curr_sta = grating_grp['sta_' + trace_type].value[roi_i] + add_to_trace
            dff_traces, dff_trace_mean, dff_mean = get_dff(traces=curr_sta, t_axis=t_axis, response_span=response_span,
                                                           baseline_span=baseline_span)

            curr_tf = grating_n[29:33]
            tf_i = np.where(tf_lst == curr_tf)[0][0]
            curr_sf = grating_n[22:26]
            sf_i = np.where(sf_lst == curr_sf)[0][0]
            curr_dire = grating_n[38:41]
            dire_i = np.where(dire_lst == curr_dire)[0][0]
            ax = plt.Subplot(f, gs_in_dict[tf_i][sf_i * len(dire_lst) + dire_i])
            f_color = pt.value_2_rgb(value=(dff_mean - dff_mean_min) / (dff_mean_max - dff_mean_min),
                                     cmap=face_cmap)

            # f_color = pt.value_2_rgb(value=dff_mean / dff_mean_max, cmap=face_cmap)

            # print f_color
            ax.set_axis_bgcolor(f_color)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.axhline(y=0, ls='--', color='#888888', lw=1)
            ax.axvspan(response_span[0], response_span[1], alpha=0.5, color='#888888', ec='none')
            for t in dff_traces:
                ax.plot(t_axis, t, '-', color='#888888', lw=0.5)
            ax.plot(t_axis, dff_trace_mean, '-r', lw=1)
            f.add_subplot(ax)

        all_axes = f.get_axes()
        for ax in all_axes:
            ax.set_ylim([v_min, v_max])
            ax.set_xlim([t_axis[0], t_axis[-1]])

        f.suptitle('roi:{:04d}; trace type:{}; baseline:{}; response:{}; \ntrace range:{}; color range:{}'
                   .format(roi_i, trace_type, baseline_span, response_span, [v_min, v_max],
                           [dff_mean_min, dff_mean_max]), fontsize=8)
        # plt.show()
        pdff.savefig(f)
        f.clear()
        plt.close(f)

    pdff.close()
nwb_f.close()

print('done!')