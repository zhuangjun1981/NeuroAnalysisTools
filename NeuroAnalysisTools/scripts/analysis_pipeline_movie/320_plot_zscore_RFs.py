import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.backends.backend_pdf import PdfPages
import NeuroAnalysisTools.DatabaseTools as dt
import NeuroAnalysisTools.SingleCellAnalysis as sca

trace_type = 'sta_f_center_subtracted'
save_folder = 'figures'
is_local_dff = True
zscore_range = [-4., 4.]
t_window = [0., 0.5]
bias = 1.

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

for plane_n in plane_ns:
    print('plotting rois in {} ...'.format(plane_n))

    plane_grp = strf_grp[plane_n]
    pdff = PdfPages(os.path.join(save_folder, 'zscore_RFs_' + plane_n + '.pdf'))

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
        # v_min, v_max = curr_strf_dff.get_data_range()

        rf_on, rf_off = curr_strf_dff.get_zscore_receptive_field(timeWindow=t_window)
        f = plt.figure(figsize=(15, 4))
        f.suptitle('{}: t_window: {}'.format(roi_n, t_window))
        ax_on = f.add_subplot(121)
        rf_on.plot_rf(plot_axis=ax_on, is_colorbar=True, cmap='RdBu_r', vmin=zscore_range[0], vmax=zscore_range[1])
        ax_on.set_title('ON zscore RF')
        ax_off = f.add_subplot(122)
        rf_off.plot_rf(plot_axis=ax_off, is_colorbar=True, cmap='RdBu_r', vmin=zscore_range[0], vmax=zscore_range[1])
        ax_off.set_title('OFF zscore RF')
        plt.close()

        # plt.show()
        pdff.savefig(f)
        f.clear()
        plt.close(f)

    pdff.close()

nwb_f.close()
