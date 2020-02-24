import os
import NeuroAnalysisTools.DatabaseTools as dt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py

area_lim = 100.
trace_type = 'f_center_raw'
save_folder = 'figures'

analysis_params = dt.ANALYSIS_PARAMS
plot_params = dt.PLOTTING_PARAMS

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

save_folder = os.path.join(curr_folder, save_folder)
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

nwb_fn = [fn for fn in os.listdir(curr_folder) if fn[-4:] == '.nwb']
if len(nwb_fn) == 0:
    raise LookupError('cannot find .nwb file.')
elif len(nwb_fn) > 1:
    raise LookupError('more than one .nwb files found.')

nwb_fn = nwb_fn[0]

pdf_path = os.path.join(save_folder, r'page_report_{}.pdf'.format(os.path.splitext(nwb_fn)[0]))
pdff = PdfPages(pdf_path)

nwb_f = h5py.File(nwb_fn, 'r')

plane_ns = dt.get_plane_ns(nwb_f)

for plane_n in plane_ns:

    roi_ns = dt.get_roi_ns(nwb_f, plane_n)

    for roi_n in roi_ns:

        roi = dt.get_roi(nwb_f, plane_n, roi_n)

        if roi.get_binary_area() >= area_lim:

            print('plotting {}; {}; {}'.format(nwb_fn, plane_n, roi_n))

            f = dt.roi_page_report(nwb_f=nwb_f,
                                   plane_n=plane_n,
                                   roi_n=roi_n,
                                   params=analysis_params,
                                   plot_params=plot_params)

            pdff.savefig(f)
            f.clear()
            plt.close()

nwb_f.close()
pdff.close()