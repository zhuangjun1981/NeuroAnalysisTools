# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:44:42 2015

@author: junz
"""

import os
import numpy as np
import h5py
import tifffile as tf
import NeuroAnalysisTools.core.ImageAnalysis as ia
import NeuroAnalysisTools.core.PlottingTools as pt
import scipy.ndimage as ni
import matplotlib.pyplot as plt

plt.ioff()


def run():
    data_file_name = 'cells_refined.hdf5'
    background_file_name = "corrected_mean_projections.tif"
    save_folder = 'figures'

    overlap_threshold = 0.9
    surround_limit = [1, 8]

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curr_folder)

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    print('reading cells file ...')
    data_f = h5py.File(data_file_name, 'r')

    cell_ns = data_f.keys()
    cell_ns.sort()

    binary_mask_array = []
    weight_mask_array = []

    for cell_n in cell_ns:
        curr_roi = ia.ROI.from_h5_group(data_f[cell_n]['roi'])
        binary_mask_array.append(curr_roi.get_binary_mask())
        weight_mask_array.append(curr_roi.get_weighted_mask())

    data_f.close()
    binary_mask_array = np.array(binary_mask_array)
    weight_mask_array = np.array(weight_mask_array)
    print('starting mask_array shape:', weight_mask_array.shape)

    print('getting total mask ...')
    total_mask = np.zeros((binary_mask_array.shape[1], binary_mask_array.shape[2]), dtype=np.uint8)
    for curr_mask in binary_mask_array:
        total_mask = np.logical_or(total_mask, curr_mask)
    total_mask = np.logical_not(total_mask)

    plt.imshow(total_mask, interpolation='nearest')
    plt.title('total_mask')
    # plt.show()

    print('getting and surround masks ...')
    binary_surround_array = []
    for binary_center in binary_mask_array:
        curr_surround = np.logical_xor(ni.binary_dilation(binary_center, iterations=surround_limit[1]),
                                       ni.binary_dilation(binary_center, iterations=surround_limit[0]))
        curr_surround = np.logical_and(curr_surround, total_mask).astype(np.uint8)
        binary_surround_array.append(curr_surround)
        # plt.imshow(curr_surround)
        # plt.show()
    binary_surround_array = np.array(binary_surround_array)

    print("saving rois ...")
    center_areas = []
    surround_areas = []
    for mask_ind in range(binary_mask_array.shape[0]):
        center_areas.append(np.sum(binary_mask_array[mask_ind].flat))
        surround_areas.append(np.sum(binary_surround_array[mask_ind].flat))
    roi_f = h5py.File('rois_and_traces.hdf5')
    roi_f['masks_center'] = weight_mask_array
    roi_f['masks_surround'] = binary_surround_array

    roi_f.close()
    print('minimum surround area:', min(surround_areas), 'pixels.')

    f = plt.figure(figsize=(10, 10))
    ax_center = f.add_subplot(211)
    ax_center.hist(center_areas, bins=30)
    ax_center.set_title('roi center area distribution')
    ax_surround = f.add_subplot(212)
    ax_surround.hist(surround_areas, bins=30)
    ax_surround.set_title('roi surround area distribution')
    # plt.show()

    print('plotting ...')
    colors = pt.random_color(weight_mask_array.shape[0])
    bg = ia.array_nor(np.max(tf.imread(background_file_name), axis=0))

    f_c_bg = plt.figure(figsize=(10, 10))
    ax_c_bg = f_c_bg.add_subplot(111)
    ax_c_bg.imshow(bg, cmap='gray', vmin=0, vmax=0.5, interpolation='nearest')
    f_c_nbg = plt.figure(figsize=(10, 10))
    ax_c_nbg = f_c_nbg.add_subplot(111)
    ax_c_nbg.imshow(np.zeros(bg.shape,dtype=np.uint8),vmin=0,vmax=1,cmap='gray',interpolation='nearest')
    f_s_nbg = plt.figure(figsize=(10, 10))
    ax_s_nbg = f_s_nbg.add_subplot(111)
    ax_s_nbg.imshow(np.zeros(bg.shape,dtype=np.uint8),vmin=0,vmax=1,cmap='gray',interpolation='nearest')

    i = 0
    for mask_ind in range(binary_mask_array.shape[0]):
        pt.plot_mask_borders(binary_mask_array[mask_ind], plotAxis=ax_c_bg, color=colors[i], borderWidth=1)
        pt.plot_mask_borders(binary_mask_array[mask_ind], plotAxis=ax_c_nbg, color=colors[i], borderWidth=1)
        pt.plot_mask_borders(binary_surround_array[mask_ind], plotAxis=ax_s_nbg, color=colors[i], borderWidth=1)
        i += 1

    # plt.show()

    print('saving figures ...')
    pt.save_figure_without_borders(f_c_bg, os.path.join(save_folder, '2P_ROIs_with_background.png'), dpi=300)
    pt.save_figure_without_borders(f_c_nbg, os.path.join(save_folder, '2P_ROIs_without_background.png'), dpi=300)
    pt.save_figure_without_borders(f_s_nbg, os.path.join(save_folder, '2P_ROI_surrounds_background.png'), dpi=300)
    f.savefig(os.path.join(save_folder, 'roi_area_distribution.pdf'), dpi=300)

if __name__ == '__main__':
    run()