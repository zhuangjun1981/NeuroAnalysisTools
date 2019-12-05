import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import corticalmapping.core.TimingAnalysis as ta
import corticalmapping.SingleCellAnalysis as sca
import corticalmapping.core.FileTools as ft
import corticalmapping.core.ImageAnalysis as ia
from matplotlib.backends.backend_pdf import PdfPages
import corticalmapping.DatabaseTools as dt

trace_type = 'sta_f_center_subtracted'
save_folder = 'figures'
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
    pdff = PdfPages(os.path.join(save_folder, 'STRFs_' + plane_n + '.pdf'))

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

        v_min, v_max = curr_strf_dff.get_data_range()
        f = curr_strf_dff.plot_traces(yRange=(v_min, v_max * 1.1), figSize=(16, 10),
                                      columnSpacing=0.002, rowSpacing=0.002)
        # plt.show()
        pdff.savefig(f)
        f.clear()
        plt.close(f)

    pdff.close()

nwb_f.close()
