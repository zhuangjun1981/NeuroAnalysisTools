import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import NeuroAnalysisTools.DatabaseTools as dt
import NeuroAnalysisTools.SingleCellAnalysis as sca

trace_type = 'sta_f_center_subtracted'
roi_t_window = [0., 0.5]
zscore_range = [0., 4.]
save_folder = 'figures'
bias = 1.

# plot control
thr_ratio = 0.4
filter_sigma = 1.
interpolate_rate = 10
absolute_thr = 1.6
level_num = 1

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

save_folder = os.path.join(curr_folder, save_folder)
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = h5py.File(nwb_fn, 'r')

strf_grp = nwb_f['analysis/strf_001_LocallySparseNoiseRetinotopicMapping']
plane_ns = strf_grp.keys()
plane_ns.sort()
print('planes:')
print('\n'.join(plane_ns))

X = None
Y = None

for plane_n in plane_ns:
    print('plotting rois in {} ...'.format(plane_n))

    plane_grp = strf_grp[plane_n]

    f_all = plt.figure(figsize=(10, 10))
    f_all.suptitle('t window: {}; z threshold: {}'.format(roi_t_window, absolute_thr / thr_ratio))
    ax_all = f_all.add_subplot(111)

    pdff = PdfPages(os.path.join(save_folder, 'RF_contours_' + plane_n + '.pdf'))

    roi_lst = nwb_f['processing/rois_and_traces_' + plane_n + '/ImageSegmentation/imaging_plane/roi_list'].value
    roi_lst = [r for r in roi_lst if r[:4] == 'roi_']
    roi_lst.sort()

    for roi_ind, roi_n in enumerate(roi_lst):
        print('roi: {} / {}'.format(roi_ind + 1, len(roi_lst)))

        curr_trace, _ = dt.get_single_trace(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
        if np.min(curr_trace) < bias:
            add_to_trace = -np.min(curr_trace) + bias
        else:
            add_to_trace = 0.

        curr_strf = sca.get_strf_from_nwb(h5_grp=strf_grp[plane_n], roi_ind=roi_ind, trace_type=trace_type)
        curr_strf_dff = curr_strf.get_local_dff_strf(is_collaps_before_normalize=True, add_to_trace=add_to_trace)
        rf_on, rf_off, _ = curr_strf_dff.get_zscore_thresholded_receptive_fields(timeWindow=roi_t_window,
                                                                                 thr_ratio=thr_ratio,
                                                                                 filter_sigma=filter_sigma,
                                                                                 interpolate_rate=interpolate_rate,
                                                                                 absolute_thr=absolute_thr)

        if X is None and Y is None:
            X, Y = np.meshgrid(np.arange(len(rf_on.aziPos)),
                               np.arange(len(rf_on.altPos)))

        levels_on = [np.max(rf_on.get_weighted_mask().flat) * thr_ratio]
        levels_off = [np.max(rf_off.get_weighted_mask().flat) * thr_ratio]
        ax_all.contour(X, Y, rf_on.get_weighted_mask(), levels=levels_on, colors='r', lw=5)
        ax_all.contour(X, Y, rf_off.get_weighted_mask(), levels=levels_off, colors='b', lw=5)

        f_single = plt.figure(figsize=(10, 10))
        ax_single = f_single.add_subplot(111)
        ax_single.contour(X, Y, rf_on.get_weighted_mask(), levels=levels_on, colors='r', lw=5)
        ax_single.contour(X, Y, rf_off.get_weighted_mask(), levels=levels_off, colors='b', lw=5)
        ax_single.set_xticks(range(len(rf_on.aziPos))[::20])
        ax_single.set_xticklabels(['{:3d}'.format(int(round(l))) for l in rf_on.aziPos[::20]])
        ax_single.set_yticks(range(len(rf_on.altPos))[::20])
        ax_single.set_yticklabels(['{:3d}'.format(int(round(l))) for l in rf_on.altPos[::-1][::20]])
        ax_single.set_aspect('equal')
        ax_single.set_title('{}: {}. t_window: {}; ON thr:{}; OFF thr:{}.'.format(plane_n, roi_n, roi_t_window,
                                                                                  rf_on.thr, rf_off.thr))
        pdff.savefig(f_single)
        f_single.clear()
        plt.close(f_single)

    pdff.close()

    ax_all.set_xticks(range(len(rf_on.aziPos))[::20])
    ax_all.set_xticklabels(['{:3d}'.format(int(round(l))) for l in rf_on.aziPos[::20]])
    ax_all.set_yticks(range(len(rf_on.altPos))[::20])
    ax_all.set_yticklabels(['{:3d}'.format(int(round(l))) for l in rf_on.altPos[::-1][::20]])
    ax_all.set_aspect('equal')
    ax_all.set_title('{}, abs_zscore_thr:{}'.format(plane_n, absolute_thr))

    f_all.savefig(os.path.join(save_folder, 'RF_contours_' + plane_n + '_all.pdf'), dpi=300)

nwb_f.close()

