import os
import numpy as np
import h5py
import tifffile as tf
import NeuroAnalysisTools.core.ImageAnalysis as ia
import NeuroAnalysisTools.core.PlottingTools as pt
import matplotlib.pyplot as plt
import NeuroAnalysisTools.HighLevel as hl

plt.ioff()

def run():
    isSave = True

    filter_sigma = 2.  # parameters only used if filter the rois
    thr_high = 0.0
    thr_low = 0.1

    bg_fn = "corrected_mean_projections.tif"
    save_folder = 'figures'

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curr_folder)

    data_f = h5py.File('caiman_segmentation_results.hdf5')
    masks = data_f['masks'].value
    data_f.close()

    bg = ia.array_nor(np.max(tf.imread(bg_fn), axis=0))

    final_roi_dict = {}

    roi_ind = 0
    for i, mask in enumerate(masks):
        mask_dict = hl.threshold_mask_by_energy(mask, sigma=filter_sigma, thr_high=thr_high, thr_low=thr_low)
        for mask_roi in mask_dict.values():
            final_roi_dict.update({'roi_{:04d}'.format(roi_ind): mask_roi})
            roi_ind += 1

    print('Total number of ROIs:',len(final_roi_dict))

    f = plt.figure(figsize=(15, 8))
    ax1 = f.add_subplot(121)
    ax1.imshow(bg, vmin=0, vmax=0.5, cmap='gray', interpolation='nearest')
    colors1 = pt.random_color(masks.shape[0])
    for i, mask in enumerate(masks):
        pt.plot_mask_borders(mask, plotAxis=ax1, color=colors1[i])
    ax1.set_title('original ROIs')
    ax1.set_axis_off()
    ax2 = f.add_subplot(122)
    ax2.imshow(ia.array_nor(bg), vmin=0, vmax=0.5, cmap='gray', interpolation='nearest')
    colors2 = pt.random_color(len(final_roi_dict))
    i = 0
    for roi in final_roi_dict.values():
        pt.plot_mask_borders(roi.get_binary_mask(), plotAxis=ax2, color=colors2[i])
        i = i + 1
    ax2.set_title('filtered ROIs')
    ax2.set_axis_off()
    # plt.show()

    if isSave:

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        f.savefig(os.path.join(save_folder, 'caiman_segmentation_filtering.pdf'), dpi=300)

        cell_file = h5py.File('cells.hdf5', 'w')

        i = 0
        for key, value in sorted(final_roi_dict.iteritems()):
            curr_grp = cell_file.create_group('cell{:04d}'.format(i))
            curr_grp.attrs['name'] = key
            value.to_h5_group(curr_grp)
            i += 1

        cell_file.close()

if __name__ == '__main__':
    run()