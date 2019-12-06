"""
This is the module to do simple motion correction of a 2-photon data set. It use open cv phase coorelation function to
find parameters of x, y rigid transformation frame by frame iteratively. The input dataset should be a set of tif
files. This files should be small enough to be loaded and manipulated in your memory.

@Jun Zhuang May 27, 2016
"""

import tifffile as tf
import os
import numpy as np
import h5py
import scipy.ndimage as ni
import scipy.signal as sig
import matplotlib.pyplot as plt
import skimage.exposure as exp
import skimage.feature as fea
import time
import shutil
from multiprocessing import Pool

from .core import ImageAnalysis as ia

try:
    import cv2
except ImportError as e:
    print('MotionCorrection: cannot import opencv.')
    print(e)

def tukey_2d(shape, alpha=0.1, sym=True):
    """
    generate 2d tukey window

    :param shape: tuple of two positive integers
    :param alpha: alpha of tukey filter
    :param sym: symmetry of tukey filter
    :return: 2d array with shape as input shape of 2d tukey filter
    """

    if len(shape) != 2:
        raise ValueError('input shape should have length of 2.')

    win_h = np.array([sig.tukey(M=shape[1], alpha=alpha, sym=sym)])
    win_w = np.array([sig.tukey(M=shape[0], alpha=alpha, sym=sym)]).transpose()

    window = np.ones(shape)
    window = window * win_h
    window = window * win_w

    return window


def preprocessing(chunk, processing_type):
    """
    performing preprocessing before motion correction
    :param chunk: 3d array, frame * y * x
    :param processing_type: int, type of preprocessing
                            0: nothing, no preprocessing
                            1: histogram equlization, return uint8 bit chunk
                            2: rectifyinng with a certain threshold
                            3: gamma correct each frame by gamma=0.1
                            4: brightness contrast adjustment
                            5: spatial filtering
                            6: tukey window with alpha 0.1
    :return: preprocessed chunk
    """

    if processing_type == 0:
        return chunk
    elif processing_type == 1:
        chunk_eh = np.zeros(chunk.shape, dtype=np.uint8)
        for i in range(chunk.shape[0]):
            frame = chunk[i].astype(np.float32)
            frame = (frame - np.amin(frame)) / (np.amax(frame) - np.amin(frame))
            chunk_eh[i] = cv2.equalizeHist((frame * 255).astype(np.uint8))
        return chunk_eh
    elif processing_type == 2:
        low_thr = -200
        high_thr = 2000
        chunk[chunk < low_thr] = low_thr
        chunk[chunk > high_thr] = high_thr
        return chunk
    elif processing_type == 3:
        chunk_gamma = np.zeros(chunk.shape, dtype=np.float32)
        for i in range(chunk.shape[0]):
            frame = chunk[i].astype(np.float32)
            frame = (frame - np.amin(frame)) / (np.amax(frame) - np.amin(frame))
            chunk_gamma[i] = exp.adjust_gamma(frame, gamma=0.1)
        return chunk_gamma.astype(chunk.dtype)
    elif processing_type == 4:
        chunk_sigmoid = np.zeros(chunk.shape, dtype=np.float32)
        for i in range(chunk.shape[0]):
            frame = chunk[i].astype(np.float32)
            frame = (frame - np.amin(frame)) / (np.amax(frame) - np.amin(frame))
            frame_sigmoid = exp.adjust_sigmoid(frame, cutoff=0.1, gain=10, inv=False)
            # plt.imshow(frame_sigmoid)
            # plt.show()
            chunk_sigmoid[i] = frame_sigmoid
        return chunk_sigmoid.astype(chunk.dtype)
    elif processing_type == 5:
        chunk_filtered = np.zeros(chunk.shape, dtype=np.float32)
        for i in range(chunk.shape[0]):
            frame = chunk[i].astype(np.float32)
            frame_filtered = ni.gaussian_filter(frame, sigma=10.)
            chunk_filtered[i] = frame_filtered
        return chunk_filtered.astype(chunk.dtype)
    elif processing_type == 6:
        window = tukey_2d(shape=(chunk.shape[1], chunk.shape[2]), alpha=0.1, sym=True)
        window = np.array([window])
        return (chunk * window).astype(chunk.dtype)
    else:
        raise LookupError('Do not understand framewise preprocessing type.')


def phase_correlation(img_match, img_ref):
    """
    openCV phase correction wrapper, as one of the align_func to perform motion correction. Open CV phaseCorrelate
    function returns (x_offset, y_offset). This wrapper switches the order of result and returns (height_offset,
    width_offset) to be more consistent with numpy indexing convention.

    :param img_match: the matching image
    :param img_ref: the reference image
    :return: rigid_transform coordinates of alignment (height_offset, width_offset)
    """
    if cv2.__version__[0] == '2':
        x_offset, y_offset =  cv2.phaseCorrelate(img_match.astype(np.float32), img_ref.astype(np.float32))
    elif cv2.__version__[0] == '3' or cv2.__version__[0] == '4':
        (x_offset, y_offset), _ = cv2.phaseCorrelate(img_match.astype(np.float32), img_ref.astype(np.float32))
    else:
        raise EnvironmentError('Do not understand opencv version.')
    return [y_offset, x_offset]


def phase_correlation_scikit(img_match, img_ref, upsample_factor=10):
    shifts, error, phasediff = fea.register_translation(src_image=img_match,
                                                        target_image=img_ref,
                                                        upsample_factor=upsample_factor)
    return shifts


def align_single_chunk(chunk, img_ref, max_offset=(10., 10.), align_func=phase_correlation, fill_value=0.,
                       verbose=True):
    """
    align the frames in a single chunk of movie to the img_ref.

    all operations will be applied with np.float32 format

    If the translation is larger than max_offset, the translation of that particular frame will be set as zero,
    and it will not be counted during the calculation of average projection image.

    :param chunk: a movie chunk, should be small enough to be managed in memory, 3d numpy.array
    :param img_ref: reference image, 2d numpy.array
    :param max_offset: maximum offset, (height, width), if single value, will be applied to both height and width
    :param align_func: function to align two image frames, return rigid transform offset (height, width)
    :param fill_value: value to fill empty pixels
    :param verbose:
    :return: alignment offset list, aligned movie chunk (same data type of original chunk), updated mean projection
    image np.float32
    """

    data_type = chunk.dtype
    img_ref = img_ref.astype(np.float32)

    # handle max_offset
    try:
        max_offset_height = float(max_offset[0])
        max_offset_width = float(max_offset[1])
    except TypeError:
        max_offset_height = float(max_offset)
        max_offset_width = float(max_offset)

    offset_list = []
    aligned_chunk = np.empty(chunk.shape, dtype=np.float32)

    # sum of valid frames, will be used to calculate updated mean project
    sum_img = np.zeros((chunk.shape[1], chunk.shape[2]), dtype=np.float32)

    # number of valid frames, will be used to calculate updated mean project
    valid_frame_num = 0

    # total number of frames of the chunk
    total_frame_num = chunk.shape[0]

    for i in range(chunk.shape[0]):

        if verbose:
            if i % (total_frame_num // 10) == 0:
                print('Motion correction progress:', int(round(float(i) * 100 / total_frame_num)), '%')

        curr_frame = chunk[i, :, :].astype(np.float32)

        curr_offset = align_func(curr_frame, img_ref)

        if abs(curr_offset[0]) <= max_offset_height and abs(curr_offset[1]) <= max_offset_width:
            aligned_frame = ia.rigid_transform_cv2_2d(curr_frame, offset=curr_offset[::-1], fill_value=fill_value)
            aligned_chunk[i, :, :] = aligned_frame
            offset_list.append(curr_offset)
            sum_img = sum_img + aligned_frame
            valid_frame_num += 1
        else:
            # print('correction {} exceeds max_offset {}.'.format(curr_offset, max_offset))
            aligned_chunk[i, :, :] = curr_frame
            offset_list.append([0., 0.])

    if valid_frame_num == 0:
        new_mean_img = np.mean(aligned_chunk, axis=0)
    else:
        new_mean_img = sum_img / float(valid_frame_num)

    return offset_list, aligned_chunk.astype(data_type), new_mean_img

def align_single_chunk_iterate_anchor(chunk, anchor_frame_ind=0, iteration=2, max_offset=(10., 10.),
                                      align_func=phase_correlation, fill_value=0.,
                                      preprocessing_type=0, verbose=True):
    """
    align the frames in a single chunk of movie to its mean projection iteratively. for each iteration, the reference
    image (mean projection) will be updated based on the aligned chunk

    all operations will be applied with np.float32 format

    If the translation is larger than max_offset, the translation of that particular frame will be set as zero,
    and it will not be counted during the calculation of average projection image.

    the difference between align_single_chunk_iterate_anchor and align_single_chunk_iterate is the former added anchor
    frame correction. anchor frame correction happens before main correction. Different from main correction which uses
    mean projection of the whole chunk as reference image, the anchor frame correction only uses one predefined frame
    as reference image. The idea is for the movie with significant movement, the mean projection of the whole chunk can
    be very blurry, and adding iteration number may not be able to make the mean projection converge. So one iteration
    of anchor frame correction before main correction, may pre-align the chunk into a roughly motion-corrected chunk
    which helps the mean projection to converge.

    :param chunk: 3d numpy.array, a movie chunk, should be small enough to be managed in memory
    :anchor_frame_ind: non-negative int, the frame number of anchor frame correction
    :param iteration: positive int, number of iterations for mean correction
    :param max_offset: maximum offset, (height, width), if single value, will be applied to both height and width
    :param align_func: function to align two image frames, return rigid transform offset (height, width)
    :param fill_value: value to fill empty pixels
    :param preprocessing_type: int, type of preprocessing before motion correction,
                               refer to preprocessing() function of this module.
    :param verbose:
    :return: alignment offset list, aligned movie chunk (same data type of original chunk), updated mean projection
    image np.float32
    """

    if iteration < 1:
        raise ValueError('iteration should be an integer larger than 0.')

    print('performing framewise preprocessing')
    chunk = preprocessing(chunk, processing_type=preprocessing_type)

    if verbose:
        print('performing anchor frame correction ...')

    anchor_frame = chunk[anchor_frame_ind]
    offset_list, aligned_chunk, img_ref = align_single_chunk(chunk, anchor_frame, max_offset=max_offset,
                                                             align_func=align_func, fill_value=fill_value,
                                                             verbose=verbose)

    for i in range(iteration):

        if verbose:
            print("\nMotion Correction, iteration " + str(i))
        offset_list, aligned_chunk, img_ref = align_single_chunk(chunk, img_ref, max_offset=max_offset,
                                                                 align_func=align_func, fill_value=fill_value,
                                                                 verbose=verbose)

    # plt.imshow(img_ref)
    # plt.colorbar()
    # plt.show()

    return offset_list, aligned_chunk, img_ref


def align_single_chunk_iterate_anchor_for_multi_thread(params):
    """
    using stia.motion_correction.align_single_chunk_iterate_anchor() to correct single tiff file. Designed to run on a
    single thread when correcting multiple tiffs.

    chunk_path, anchor_frame_ind, iteration, max_offset, align_func, fill_value, output_folder = params

    modifications from stia.motion_correction.align_single_chunk_iterate_anchor() function
    1. input is a big tuple with all the inputs
    2. input chunk path instead of movie array to reduce the memory involvement between threads
    3. all output will be written onto the disk to coupe the cope the thread failure
    4. verbose = False
    5. last item in the input params is output_folder to save temporary correction results, if None, save in the same
       folder as chunk_path.
    """

    chunk_path, anchor_frame_ind, iteration, max_offset, align_func, fill_value, \
    preprocessing_type, output_folder = params

    chunk_real_path = os.path.abspath(chunk_path)
    chunk_name = os.path.splitext(os.path.split(chunk_real_path)[1])[0]
    chunk = tf.imread(chunk_real_path)

    t0 = time.time()
    print ('\nstart correcting {}'.format(chunk_name))

    offset_list, aligned_chunk, img_ref = align_single_chunk_iterate_anchor(chunk=chunk,
                                                                            anchor_frame_ind=anchor_frame_ind,
                                                                            iteration=iteration,
                                                                            max_offset=max_offset,
                                                                            align_func=align_func,
                                                                            fill_value=fill_value,
                                                                            preprocessing_type=preprocessing_type,
                                                                            verbose=False)

    chunk_folder, chunk_fn = os.path.split(os.path.abspath(chunk_path))
    chunk_fn_n = os.path.splitext(chunk_fn)[0]

    if output_folder is None:
        output_folder= chunk_folder

    result_f = h5py.File(os.path.join(output_folder, 'temp_offsets_' + chunk_fn_n + '.hdf5'), 'a')
    offset_dset = result_f.create_dataset('offsets', data=offset_list)
    offset_dset.attrs['data_format'] = '[row, col]'
    result_f['mean_projection'] = np.mean(aligned_chunk, axis=0, dtype=np.float32)
    result_f['max_projection'] = np.max(aligned_chunk, axis=0)
    result_f['file_path'] = chunk_real_path
    print ('\n\t{:09.2f} second; {}; correction finished.'.format(time.time() - t0, chunk_name))


def correct_movie(mov, offsets, fill_value=0., verbose=True):
    """
    correcting a movie with given offset list, whole process will be operating on np.float32 data format.

    :param mov: movie to be corrected, should be a 3-d np.array managable by the computer memory
    :param offsets: list of correction offsets for each frame of the movie
    :param fill_value: value to fill empty pixels
    :param verbose:
    :return: corrected movie, with same data type as original movie
    """

    if isinstance(offsets, np.ndarray):
        if len(offsets.shape) != 2:
            raise ValueError('offsets should be 2-dimensional.')
        elif offsets.shape[0] != mov.shape[0]:
            raise ValueError('offsets should have same length as the number of frames in the movie!')
        elif offsets.shape[1] != 2:
            raise ValueError('each item in offsets should contain 2 values, (offset_height, offset_width)!')
    elif isinstance(offsets, list):
        if len(offsets) != mov.shape[0]:
            raise ValueError('offsets should have same length as the number of frames in the movie!')
        else:
            for single_offset in offsets:
                if len(single_offset) != 2:
                    raise ValueError('each item in offsets should contain 2 values, (offset_height, offset_width)!')

    total_frame_num = mov.shape[0]
    corrected_mov = np.empty(mov.shape, dtype=np.float32)

    for i in range(mov.shape[0]):

        if verbose:
            if i % (total_frame_num // 10) == 0:
                print('Correction progress:', int(round(float(i) * 100 / total_frame_num)), '%')

        curr_frame = mov[i, :, :].astype(np.float32)
        curr_offset = offsets[i]
        corrected_frame = ia.rigid_transform_cv2_2d(curr_frame, offset=curr_offset[::-1], fill_value=fill_value)
        corrected_mov[i, :, :] = corrected_frame

    return corrected_mov.astype(mov.dtype)


def correct_movie_for_multi_thread(params):

    mov_path, offsets, fill_value, output_folder, down_sample_rate = params
    mov_real_path = os.path.abspath(mov_path)
    mov_folder, mov_fn = os.path.split(mov_real_path)
    mov_name, mov_ext = os.path.splitext(mov_fn)
    if mov_ext != '.tif':
        raise IOError('input movie file should be a .tif file.')
    save_fn = mov_name + '_corrected.tif'

    if output_folder is None:
        output_folder = mov_folder
    save_path = os.path.join(output_folder, save_fn)

    t0 = time.time()
    print('\napplying correction offsets to movie: {}.'.format(mov_name))
    mov = tf.imread(mov_real_path)
    mov_corr = correct_movie(mov=mov, offsets=offsets, fill_value=fill_value, verbose=False)
    tf.imsave(save_path, mov_corr)

    mean_projection_c = np.mean(mov_corr, axis=0, dtype=np.float32)
    max_projection_c =np.max(mov_corr, axis=0)

    if down_sample_rate is not None:
        mov_down = ia.z_downsample(mov_corr, down_sample_rate, is_verbose=False)
        print('\n\t{:09.2f} second; finished applying correction to movie: {}.'.format(time.time() - t0, mov_name))
        return mean_projection_c, max_projection_c, mov_down
    else:
        print('\n\t{:09.2f} second; finished applying correction to movie: {}.'.format(time.time() - t0, mov_name))
        return mean_projection_c, max_projection_c, None


def align_multiple_files_iterate_anchor_multi_thread(f_paths,
                                                     output_folder,
                                                     process_num=1,
                                                     anchor_frame_ind_chunk=0,
                                                     anchor_frame_ind_projection=0,
                                                     iteration_chunk=6,
                                                     iteration_projection=10,
                                                     max_offset_chunk=(10., 10.),
                                                     max_offset_projection=(10., 10.),
                                                     align_func=phase_correlation,
                                                     preprocessing_type=0,
                                                     fill_value=0.):
    """

    Motion correct a list of movie files (currently only support .tif format, designed for ScanImage output files.
    each files will be first aligned to its own mean projection iteratively. In the first iteration, it uses a single
    frame specified by anchor_frame_ind as reference image. In the following iterations, the reference image is
    calculated as the mean projection of the corrected movie in last iteration. This step will be applied with
    multi-thread processing. Temporary correction results will be saved in the output_folder.

    Notice: this function will not output motion_corrected movies. Instead it will only save motion correction offsets
    and mean/max_projections of each file after correction, and mean/max_projection of all files.

    all operations will be applied with np.float32 format

    :param f_paths: list of paths of movies to be corrected
    :param process_num: positive int, number of processes for multi-threading
    :param output_folder: str, the path of directory to save correction results, if None, it will be
                          'input_folder/corrected'
    :param anchor_frame_ind_chunk: non-negative int, frame index for first correction iteration for each file
    :param anchor_frame_ind_projection: non-negative int, frame index for first correction iteration for mean
                                        projections
    :param iteration_chunk: non-negative int, number of iterations to correct single file
    :param iteration_projection: non-negative int, number of iterations to correct mean projections
    :param max_offset: tuple of two positive floats, (row, col), if the absolute value of the correction offsets of a
                       single frame is larger than this value, it will be set to zero.
    :param align_func: function object, the function to align two frames
    :param fill_value: float, value to fill the correction margin
    :param preprocessing_type: int, type of preprocessing before motion correction,
                               refer to preprocessing() function of this module.
    :return: None
    """

    correction_temp_folder = os.path.join(output_folder, 'correction_temp')
    if os.path.isdir(correction_temp_folder):
        shutil.rmtree(correction_temp_folder, ignore_errors=False)
        time.sleep(1.)
    os.mkdir(correction_temp_folder)
    print ('\ncorrection output will be saved in {}.'.format(os.path.abspath(output_folder)))

    print ('\naligning single chunks:')
    if process_num == 1:
        for f in f_paths:
            curr_params = (f, anchor_frame_ind_chunk, iteration_chunk, max_offset_chunk, align_func, fill_value,
                           preprocessing_type, correction_temp_folder)
            align_single_chunk_iterate_anchor_for_multi_thread(curr_params)
    else:
        params_lst = [(f, anchor_frame_ind_chunk, iteration_chunk, max_offset_chunk, align_func, fill_value,
                       preprocessing_type, correction_temp_folder) for f in f_paths]
        # print '\n'.join([str(p) for p in params_lst])
        chunk_p = Pool(process_num)
        chunk_p.map(align_single_chunk_iterate_anchor_for_multi_thread, params_lst)

    print('\naligning among files ...')
    chunk_offset_fns = [f for f in os.listdir(correction_temp_folder) if f[0: 13] == 'temp_offsets_']
    chunk_offset_fns.sort()
    # print('\n'.join(chunk_offset_fns))
    mean_projections = []
    # max_projections = []
    file_paths = []
    chunk_offsets = []
    for chunk_offset_fn in chunk_offset_fns:
        chunk_offset_f = h5py.File(os.path.join(correction_temp_folder, chunk_offset_fn), 'r')
        mean_projections.append(chunk_offset_f['mean_projection'].value)
        # max_projections.append(chunk_offset_f['max_projection'].value)
        chunk_offsets.append(chunk_offset_f['offsets'].value)
        file_paths.append(chunk_offset_f['file_path'].value)

    _ = align_single_chunk_iterate_anchor(chunk=np.array(mean_projections),
                                          anchor_frame_ind=anchor_frame_ind_projection,
                                          iteration=iteration_projection,
                                          max_offset=max_offset_projection,
                                          align_func=align_func,
                                          fill_value=0.,
                                          preprocessing_type=preprocessing_type,
                                          verbose=False)
    offsets_chunk, mean_projections_c, mean_projection = _
    offsets_f = h5py.File(os.path.join(output_folder, "correction_offsets.hdf5"), 'a')
    for i, file_path in enumerate(file_paths):
        curr_chunk_offsets = chunk_offsets[i]
        curr_global_offset = offsets_chunk[i]
        offsets_dset = offsets_f.create_dataset('file_{:04d}'.format(i),
                                                data=curr_chunk_offsets + curr_global_offset)
        offsets_dset.attrs['format'] = ['height', 'width']
        offsets_dset.attrs['path'] = os.path.abspath(file_path)
    offsets_f.create_dataset('path_list', data=str(file_paths))
    offsets_f.close()
    print ('\nchunks aligned offsets and projection images saved.')


def align_single_file(f_path, output_folder, anchor_frame_ind=0, iteration=6, max_offset=(10, 10),
                      align_func=phase_correlation, fill_value=0., preprocessing_type=0, verbose=False):
    """

    :param f_path: str, path to the file to be corrected.
    :param output_folder: str, path to save correction results
    :param anchor_frame_ind: non-negative int, frame index for first correction iteration
    :param iteration: non-negative int, number of iterations to correct single file
    :param max_offset: tuple of two positive floats, (row, col), if the absolute value of the correction offsets of a
                       single frame is larger than this value, it will be set to zero.
    :param align_func: function object, the function to align two frames
    :param fill_value: float, value to fill the correction margin
    :param preprocessing_type: int, type of preprocessing before motion correction,
                               refer to preprocessing() function of this module.
    :return: None
    """

    print ('\ncorrection output will be saved in {}.'.format(os.path.abspath(output_folder)))

    print ('\naligning file: {} ...'.format(os.path.abspath(f_path)))
    mov = tf.imread(f_path)
    _ = align_single_chunk_iterate_anchor(mov, anchor_frame_ind=anchor_frame_ind, iteration=iteration,
                                          max_offset=max_offset, align_func=align_func, fill_value=fill_value,
                                          preprocessing_type=preprocessing_type, verbose=verbose)
    offset_list, aligned_chunk, img_ref = _
    # tf.imsave(os.path.join(output_folder, 'corrected_mean_projection.tif'), img_ref)
    # tf.imsave(os.path.join(output_folder, 'corrected_max_projection.tif'), np.max(aligned_chunk, axis=0))
    offsets_f = h5py.File(os.path.join(output_folder, "correction_offsets.hdf5"), 'a')
    offsets_dset = offsets_f.create_dataset('file_0000', data=offset_list)
    offsets_dset.attrs['format'] = ['height', 'width']
    offsets_dset.attrs['path'] = os.path.abspath(f_path)
    offsets_f.close()
    print ('\nmotion correction results saved.')


def motion_correction(input_folder,
                      input_path_identifier,
                      process_num=1,
                      output_folder=None,
                      anchor_frame_ind_chunk=0,
                      anchor_frame_ind_projection=0,
                      iteration_chunk=6,
                      iteration_projection=10,
                      max_offset_chunk=(10., 10.),
                      max_offset_projection=(10., 10.),
                      align_func=phase_correlation,
                      preprocessing_type=0,
                      fill_value=0.):
    """
    Motion correct a list of movie files (currently only support .tif format, designed for ScanImage output files.
    each files will be first aligned to its own mean projection iteratively. In the first iteration, it uses a single
    frame specified by anchor_frame_ind as reference image. In the following iterations, the reference image is
    calculated as the mean projection of the corrected movie in last iteration. This step will be applied with
    multi-thread processing. Temporary correction results will be saved in the output_folder.

    Notice: this function will not output motion_corrected movies. Instead it will only save motion correction offsets
    and mean/max_projections of each file after correction, and mean/max_projection of all files.

    all operations will be applied with np.float32 format

    :param input_folder: str, the path of directory containing raw .tif movies, all the movies to be corrected should
                         be in this directory.
    :param input_path_identifier: str, only .tif files in the input folder with name containing this string will be
                                  considered input files, if None, all .tif files will be corrected
    :param process_num: positive int, number of processes for multi-threading
    :param output_folder: str, the path of directory to save correction results, if None, it will be
                          'input_folder/corrected'
    :param anchor_frame_ind_chunk: non-negative int, frame index for first correction iteration for each file
    :param anchor_frame_ind_projection: non-negative int, frame index for first correction iteration for mean
                                        projections
    :param iteration_chunk: non-negative int, number of iterations to correct single file
    :param iteration_projection: non-negative int, number of iterations to correct mean projections
    :param max_offset: tuple of two positive floats, (row, col), if the absolute value of the correction offsets of a
                       single frame is larger than this value, it will be set to zero.
    :param align_func: function object, the function to align two frames
    :param fill_value: float, value to fill the correction margin
    :param preprocessing_type: int, type of preprocessing before motion correction,
                               refer to preprocessing() function of this module.
    :return f_paths: list of str, absolute paths of all files that are corrected
    :return output_folder: str, absolute path to the results folder
    """

    print ('\nfinding files to correct ...')
    f_paths = [f for f in os.listdir(input_folder) if f[-4:] == '.tif' and input_path_identifier in f]
    if len(f_paths) < 1:
        raise LookupError("Did not find any file to correct.")
    else:

        print ('\nChecking output folder structure ...')

        if output_folder is None:
            output_folder = os.path.join(input_folder, 'corrected')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        if os.path.isfile(os.path.join(output_folder, 'correction_offsets.hdf5')):
            raise IOError('"correction_offsets.hdf5" already exists in output folder.')

        if len(f_paths) == 1:

            f_paths = [os.path.join(input_folder, f) for f in f_paths]

            align_single_file(f_path=f_paths[0], output_folder=output_folder,
                              anchor_frame_ind=anchor_frame_ind_chunk, iteration=iteration_chunk,
                              max_offset=max_offset_chunk, align_func=align_func, fill_value=fill_value,
                              preprocessing_type=preprocessing_type, verbose=False)

        else:
            f_paths.sort()
            f_paths = [os.path.abspath(os.path.join(input_folder, f)) for f in f_paths]
            print ('files to be corrected:')
            print ('\n'.join(f_paths))

            align_multiple_files_iterate_anchor_multi_thread(f_paths=f_paths,
                                                             output_folder=output_folder,
                                                             process_num=process_num,
                                                             anchor_frame_ind_chunk=anchor_frame_ind_chunk,
                                                             anchor_frame_ind_projection=anchor_frame_ind_projection,
                                                             iteration_chunk=iteration_chunk,
                                                             iteration_projection=iteration_projection,
                                                             max_offset_chunk=max_offset_chunk,
                                                             max_offset_projection=max_offset_projection,
                                                             align_func=align_func,
                                                             fill_value=fill_value,
                                                             preprocessing_type=preprocessing_type)

    return [os.path.abspath(f) for f in f_paths], os.path.abspath(output_folder)


def apply_correction_offsets(offsets_path,
                             path_pairs,
                             output_folder=None,
                             process_num=1,
                             fill_value=0.,
                             avi_downsample_rate=20,
                             is_equalizing_histogram=False):
    """
    apply correction offsets to a set of uncorrected files

    :param offsets_path: str, path to the .hdf5 file contains the correction offsets. This file should contain a series
                         of datasets with names: file0000, file0001, file0002 .etc. Each dataset is a n x 2 array,
                         n is the frame number of the file, first column: global row correction; second column: global
                         column correction. Each dataset should also have an attribute 'path' with the source file path
                         string.
    :param path_pairs: list, each item is a tuple of two strings (source file path, target file path).
                       source file path should match the 'path' attribute in the correction offsets file. This is
                       designed to apply correction from one color channel to another color channel
    :param process_num: positive int, number of process
    :param output_folder: str, output path
    :param avi_downsample_rate: positive int, the down sample rate for the output .avi movie for visualizing the
                             correction results. if None, no .avi movie will be created.
    :return: None
    """

    print('\napplying correction ...')

    offsets_f = h5py.File(offsets_path, 'r')

    if output_folder is None:
        output_folder = os.path.dirname(os.path.abspath(offsets_path))

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if os.path.isfile(os.path.join(output_folder, 'corrected_mean_projections.tif')):
        raise IOError('"corrected_mean_projections.tif" already exists in output folder.')
    if os.path.isfile(os.path.join(output_folder, 'corrected_mean_projection.tif')):
        raise IOError('"corrected_mean_projection.tif" already exists in output folder.')
    if os.path.isfile(os.path.join(output_folder, 'corrected_max_projections.tif')):
        raise IOError('"corrected_max_projections.tif" already exists in output folder.')
    if os.path.isfile(os.path.join(output_folder, 'corrected_max_projection.tif')):
        raise IOError('"corrected_max_projection.tif" already exists in output folder.')

    params_list = []
    for path_pair in path_pairs:
        target_path = path_pair[1]
        source_path = path_pair[0]

        # print('target path: ' + target_path)
        # print('source path: ' + source_path)

        offset = None
        for offset_name, offset_dest in offsets_f.items():
            if offset_name != 'path_list' and offset_dest.attrs['path'] == source_path:
                offset = offset_dest.value
                break
        if offset is None:
            raise LookupError('can not find source file ({}) in the offsets file for target file: ({}).'
                              .format(source_path, target_path))
        params_list.append((target_path, offset, fill_value, output_folder, avi_downsample_rate))
    offsets_f.close()


    if process_num == 1:
        mov_downs = []
        mean_projections = []
        max_projections = []
        for params in params_list:
            _ = correct_movie_for_multi_thread(params)
            mean_projections.append(_[0])
            max_projections.append(_[1])
            mov_downs.append(_[2])
    elif process_num > 1:
        p_corr = Pool(process_num)
        _ = p_corr.map(correct_movie_for_multi_thread, params_list)
        mean_projections = [p[0] for p in _]
        max_projections = [p[1] for p in _]
        mov_downs = [p[2] for p in _]
    else:
        raise ValueError('process_num should be not less than one.')

    max_projection = np.max(max_projections, axis=0)
    mean_projection = np.mean(mean_projections, axis=0)

    tf.imsave(os.path.join(output_folder, 'corrected_mean_projections.tif'), np.array(mean_projections))
    tf.imsave(os.path.join(output_folder, 'corrected_mean_projection.tif'), mean_projection)
    tf.imsave(os.path.join(output_folder, 'corrected_max_projections.tif'), np.array(max_projections))
    tf.imsave(os.path.join(output_folder, 'corrected_max_projection.tif'), max_projection)

    if avi_downsample_rate is not None:
        print ('\ngenerating .avi downsampled movie after correction ...')

        # import skvideo.io
        # mov_name = 'corrected_movie.avi'
        # writer = skvideo.io.FFmpegWriter(os.path.join(output_folder, mov_name), outputdict={'-framerate': '30'})
        # writer.writeFrame(mov_down)
        # writer.close()

        if cv2.__version__[0:1] == '4':
            codex = 'XVID'
            mov_name = 'corrected_movie_' + codex + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*codex)
            out = cv2.VideoWriter(os.path.join(output_folder, mov_name), fourcc, 30,
                                  (mov_downs[0].shape[2], mov_downs[0].shape[1]), isColor=False)
        elif cv2.__version__[0:1] == '3.1':
            codex = 'XVID'
            mov_name = 'corrected_movie_' + codex + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*codex)
            out = cv2.VideoWriter(os.path.join(output_folder, mov_name), fourcc, 30,
                                  (mov_downs[0].shape[2], mov_downs[0].shape[1]), isColor=False)
        elif cv2.__version__[0:6] == '2.4.11':
            mov_name = 'corrected_movie.avi'
            out = cv2.VideoWriter(os.path.join(output_folder, mov_name), -1, 30,
                                  (mov_downs[0].shape[2], mov_downs[0].shape[1]), isColor=False)
        elif cv2.__version__[0:3] == '2.4':
            codex = 'XVID'
            mov_name = 'corrected_movie_' + codex + '.avi'
            fourcc = cv2.cv.CV_FOURCC(*codex)
            out = cv2.VideoWriter(os.path.join(output_folder, mov_name), fourcc, 30,
                                  (mov_downs[0].shape[2], mov_downs[0].shape[1]), isColor=False)
        else:
            raise EnvironmentError('Do not understand opencv cv2 version: {}.'.format(cv2.__version__))

        mov_down = np.concatenate(mov_downs, axis=0).astype(np.float32)
        mov_down = (mov_down - np.amin(mov_down)) / (np.amax(mov_down) - np.amin(mov_down))
        mov_down = (mov_down * 255).astype(np.uint8)

        for display_frame in mov_down:
            if is_equalizing_histogram:
                display_frame = cv2.equalizeHist(display_frame)
            out.write(display_frame)

        out.release()
        cv2.destroyAllWindows()
        print ('.avi moive generated.')


if __name__ == "__main__":

    win = tukey_2d((512, 512))
    plt.imshow(win, interpolation='nearest')
    plt.show()