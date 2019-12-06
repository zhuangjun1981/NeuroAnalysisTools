import sys
sys.path.extend([r"E:\data\github_packages\CaImAn"])

import caiman as cm
import numpy as np
import os
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
import tifffile as tf
import h5py
import warnings
from multiprocessing import Pool

base_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180605-M391355-2p\zstack\zstack_zoom2"

reference_chn = 'green'

n_processes = 5

def correct_single_movie(folder_path):

    #=======================================setup parameters==============================================
    # number of iterations for rigid motion correction
    niter_rig = 5

    # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    max_shifts = (30, 30)

    # for parallelization split the movies in  num_splits chuncks across time
    # if none all the splits are processed and the movie is saved
    splits_rig = 56

    # intervals at which patches are laid out for motion correction
    # num_splits_to_process_rig = None

    # create a new patch every x pixels for pw-rigid correction
    strides = (48, 48)

    # overlap between pathes (size of patch strides+overlaps)
    overlaps = (24, 24)

    # for parallelization split the movies in  num_splits chuncks across time
    splits_els = 56

    # num_splits_to_process_els = [28, None]

    # upsample factor to avoid smearing when merging patches
    upsample_factor_grid = 4

    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    # if True, apply shifts fast way (but smoothing results) by using opencv
    shifts_opencv = True

    # if True, make the SAVED movie and template mostly nonnegative by removing min_mov from movie
    nonneg_movie = False
    # =======================================setup parameters==============================================


    offset_mov = 0.

    file_path = [f for f in os.listdir(folder_path) if f[-4:] == '.tif']
    if len(file_path) == 0:
        raise LookupError('no tif file found in folder: {}'.format(folder_path))
    elif len(file_path) > 1:
        raise LookupError('more than one tif files found in folder: {}'.format(folder_path))
    else:
        file_path = os.path.join(folder_path, file_path[0])

    # create a motion correction object# creat
    mc = MotionCorrect(file_path, offset_mov,
                       dview=None, max_shifts=max_shifts, niter_rig=niter_rig,
                       splits_rig=splits_rig, strides=strides, overlaps=overlaps,
                       splits_els=splits_els, upsample_factor_grid=upsample_factor_grid,
                       max_deviation_rigid=max_deviation_rigid,
                       shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie)

    mc.motion_correct_rigid(save_movie=True)
    # load motion corrected movie
    m_rig = cm.load(mc.fname_tot_rig)
    m_rig = m_rig.astype(np.int16)
    save_name = os.path.splitext(file_path)[0] + '_corrected.tif'
    tf.imsave(os.path.join(folder_path, save_name), m_rig)
    tf.imsave(os.path.join(folder_path, 'corrected_mean_projection.tif'),
              np.mean(m_rig, axis=0).astype(np.float32))
    tf.imsave(os.path.join(folder_path, 'corrected_max_projection.tif'),
              np.max(m_rig, axis=0).astype(np.float32))

    offset_f = h5py.File(os.path.join(folder_path, 'correction_offsets.hdf5'))
    offsets = mc.shifts_rig
    offsets = np.array([np.array(o) for o in offsets]).astype(np.float32)
    offset_dset = offset_f.create_dataset(name='file_0000', data=offsets)
    offset_dset.attrs['format'] = 'height, width'
    offset_dset.attrs['path'] = file_path

    os.remove(mc.fname_tot_rig[0])


if __name__ == '__main__':
    data_folder = os.path.join(base_folder, reference_chn)
    chunk_p = Pool(n_processes)
    folder_list = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    folder_list.sort()
    print('\n'.join(folder_list))
    folder_list = [os.path.join(data_folder, f) for f in folder_list]
    chunk_p.map(correct_single_movie, folder_list)