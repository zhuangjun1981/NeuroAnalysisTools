import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import corticalmapping.NwbTools as nt
import corticalmapping.core.TimingAnalysis as ta
import corticalmapping.SingleCellAnalysis as sca
import corticalmapping.core.FileTools as ft
import corticalmapping.core.ImageAnalysis as ia
from matplotlib.backends.backend_pdf import PdfPages

roi_t_window = [0., 0.5]
zscore_range = [0., 4.]
save_folder = 'figures'
is_add_to_traces = True

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

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb']
print('\n'.join(nwb_fn))

if len(nwb_fn) != 1:
    raise LookupError

nwb_fn = nwb_fn[0]
rff = h5py.File(nwb_fn, 'r')

strf_grp = rff['analysis/STRFs']
plane_ns = strf_grp.keys()
plane_ns.sort()
print('planes:')
print('\n'.join(plane_ns))

X = None
Y = None

for plane_n in plane_ns:
    print('plotting rois in {} ...'.format(plane_n))

    if is_add_to_traces:
        add_to_trace = h5py.File(os.path.join(plane_n, "caiman_segmentation_results.hdf5"),
                                 'r')['bias_added_to_movie'].value
    else:
        add_to_trace = 0.

    plane_grp = strf_grp[plane_n]

    roi_ns = [rn[-8:] for rn in plane_grp.keys()]
    roi_ns.sort()

    f_all = plt.figure(figsize=(10, 10))
    f_all.suptitle('t window: {}; z threshold: {}'.format(roi_t_window, absolute_thr/thr_ratio))
    ax_all = f_all.add_subplot(111)

    pdff = PdfPages(os.path.join(save_folder, 'RF_contours_' + plane_n + '.pdf'))

    for roi_ind, roi_n in enumerate(roi_ns):
        print('roi: {} / {}'.format(roi_ind + 1, len(roi_ns)))
        curr_strf = sca.SpatialTemporalReceptiveField.from_h5_group(plane_grp['strf_' + roi_n])
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

rff.close()

