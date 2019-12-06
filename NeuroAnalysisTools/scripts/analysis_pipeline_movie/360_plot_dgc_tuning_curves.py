import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import corticalmapping.DatabaseTools as dt

trace_type = 'f_center_subtracted'
response_table_path = 'analysis/response_table_003_DriftingGratingCircleRetinotopicMapping'

baseline_span = [-0.5, 0.]
response_span = [0., 1.5]
bias = 1.

def get_response(traces, t_axis, response_span, baseline_span):
    """

    :param traces: dimension, trial x timepoint
    :param t_axis:
    :return:
    """

    baseline_ind = np.logical_and(t_axis > baseline_span[0], t_axis <= baseline_span[1])
    response_ind = np.logical_and(t_axis > response_span[0], t_axis <= response_span[1])

    trace_mean = np.mean(traces, axis=0)
    baseline_mean = np.mean(trace_mean[baseline_ind])
    dff_trace_mean = (trace_mean - baseline_mean) / baseline_mean
    dff_mean = np.mean(dff_trace_mean[response_ind])

    baselines = np.mean(traces[:, baseline_ind], axis=1, keepdims=True)
    dff_traces = (traces - baselines) / baselines
    dffs = np.mean(dff_traces[:, response_ind], axis=1)
    dff_std = np.std(dffs)
    dff_sem = dff_std / np.sqrt(traces.shape[0])

    return dff_mean, dff_std, dff_sem


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

    pdff = PdfPages(os.path.join(save_folder, 'tuning_curve_DriftingGrating_' + plane_n + '_mean.pdf'))

    for roi_i, roi_n in enumerate(roi_lst):
        print(roi_n)

        curr_trace, _ = dt.get_single_trace(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
        if np.min(curr_trace) < bias:
            add_to_trace = -np.min(curr_trace) + bias
        else:
            add_to_trace = 0.

        # get response table
        res_tab = pd.DataFrame(columns=['con', 'tf', 'sf', 'dire', 'dff_mean', 'dff_std', 'dff_sem'])
        row_ind = 0

        for grating_n in grating_ns:
            grating_grp = res_grp[grating_n]
            curr_sta = grating_grp['sta_' + trace_type].value[roi_i] + add_to_trace
            _ = get_response(traces=curr_sta, t_axis=t_axis, response_span=response_span, baseline_span=baseline_span)
            dff_mean, dff_std, dff_sem = _

            con = float(grating_n.split('_')[5][3:])
            tf = float(grating_n.split('_')[3][2:])
            sf = float(grating_n.split('_')[2][2:])
            dire = int(grating_n.split('_')[4][4:])

            res_tab.loc[row_ind] = [con, tf, sf, dire, dff_mean, dff_std, dff_sem]
            row_ind += 1

        # find the preferred condition
        top_condition = res_tab[res_tab.dff_mean == max(res_tab.dff_mean)]

        # make figure
        f = plt.figure(figsize=(8.5, 11))

        # get tf plot
        tf_conditions = res_tab[(res_tab.sf == float(top_condition.sf)) & \
                                (res_tab.dire == int(top_condition.dire))]
        tf_conditions = tf_conditions.sort_values(by='tf')

        tf_log = np.log(tf_conditions.tf)

        ax_tf = f.add_subplot(311)
        ax_tf.fill_between(x=tf_log, y1=tf_conditions.dff_mean + tf_conditions.dff_sem,
                           y2=tf_conditions.dff_mean - tf_conditions.dff_sem, edgecolor='none',
                           facecolor='#888888', alpha=0.5)
        ax_tf.axhline(y=0, ls='--', color='k', lw=1)
        ax_tf.plot(tf_log, tf_conditions.dff_mean, 'r-', lw=2)
        ax_tf.set_title('temporal frequency tuning', rotation='vertical', x=-0.4, y=0.5, va='center', ha='center',
                        size=10)
        ax_tf.set_xticks(tf_log)
        ax_tf.set_xticklabels(list(tf_conditions.tf))
        ax_tf.set_xlim(np.log([0.9, 16]))
        ax_tf_xrange = ax_tf.get_xlim()[1] - ax_tf.get_xlim()[0]
        ax_tf_yrange = ax_tf.get_ylim()[1] - ax_tf.get_ylim()[0]
        ax_tf.set_aspect(aspect=(ax_tf_xrange / ax_tf_yrange))
        ax_tf.set_ylabel('mean df/f', size=10)
        ax_tf.set_xlabel('temporal freqency (Hz)', size=10)
        ax_tf.tick_params(axis='both', which='major', labelsize=10)

        # get sf plot
        sf_conditions = res_tab[(res_tab.tf == float(top_condition.tf)) & \
                                (res_tab.dire == int(top_condition.dire))]
        sf_conditions = sf_conditions.sort_values(by='sf')

        sf_log = np.log(sf_conditions.sf)

        ax_sf = f.add_subplot(312)
        ax_sf.fill_between(x=sf_log, y1=sf_conditions.dff_mean + sf_conditions.dff_sem,
                           y2=sf_conditions.dff_mean - sf_conditions.dff_sem, edgecolor='none',
                           facecolor='#888888', alpha=0.5)
        ax_sf.axhline(y=0, ls='--', color='k', lw=1)
        ax_sf.plot(sf_log, sf_conditions.dff_mean, '-r', lw=2)
        ax_sf.set_title('spatial frequency tuning', rotation='vertical', x=-0.4, y=0.5, va='center', ha='center',
                        size=10)
        ax_sf.set_xticks(sf_log)
        ax_sf.set_xticklabels(['{:04.2f}'.format(s) for s in list(sf_conditions.sf)])
        ax_sf.set_xlim(np.log([0.008, 0.4]))
        ax_sf_xrange = ax_sf.get_xlim()[1] - ax_sf.get_xlim()[0]
        ax_sf_yrange = ax_sf.get_ylim()[1] - ax_sf.get_ylim()[0]
        ax_sf.set_aspect(aspect=(ax_sf_xrange / ax_sf_yrange))
        ax_sf.set_ylabel('mean df/f', size=10)
        ax_sf.set_xlabel('spatial freqency (cpd)', size=10)
        ax_sf.tick_params(axis='both', which='major', labelsize=10)

        # get dire plot
        dire_conditions = res_tab[(res_tab.tf == float(top_condition.tf)) & \
                                  (res_tab.sf == float(top_condition.sf))]
        dire_conditions = dire_conditions.sort_values(by='dire')
        dire_arc = list(dire_conditions.dire * np.pi / 180.)
        dire_arc.append(dire_arc[0])
        dire_dff = np.array(dire_conditions.dff_mean)
        dire_dff[dire_dff < 0.] = 0.
        dire_dff = list(dire_dff)
        dire_dff.append(dire_dff[0])
        dire_dff_sem = list(dire_conditions.dff_sem)
        dire_dff_sem.append(dire_dff_sem[0])
        dire_dff_low = np.array(dire_dff) - np.array(dire_dff_sem)
        dire_dff_low[dire_dff_low < 0.] = 0.
        dire_dff_high = np.array(dire_dff) + np.array(dire_dff_sem)

        r_ticks = [0, round(max(dire_dff) * 10000.) / 10000.]

        ax_dire = f.add_subplot(313, projection='polar')
        ax_dire.fill_between(x=dire_arc, y1=dire_dff_low, y2=dire_dff_high, edgecolor='none', facecolor='#888888',
                             alpha=0.5)
        ax_dire.plot(dire_arc, dire_dff, '-r', lw=2)
        ax_dire.set_title('orientation tuning', rotation='vertical', x=-0.4, y=0.5, va='center', ha='center', size=10)
        ax_dire.set_rticks(r_ticks)
        ax_dire.set_xticks(dire_arc)
        ax_dire.tick_params(axis='both', which='major', labelsize=10)

        f.suptitle('roi:{:04d}; trace type:{}; baseline:{}; response:{}'
                   .format(roi_i, trace_type, baseline_span, response_span), fontsize=10)
        # plt.show()
        pdff.savefig(f)
        f.clear()
        plt.close(f)

    pdff.close()
nwb_f.close()