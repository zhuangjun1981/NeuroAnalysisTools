"""
This is the module to do simple motion correction of a 2-photon data set. It use open cv phase coorelation function to
find parameters of x, y rigid transformation frame by frame iteratively. The input dataset should be a set of tif
files. This files should be small enough to be loaded and manipulated in your memory.

@Jun Zhuang May 27, 2016
"""

import tifffile as tf
import os
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
from .core import ImageAnalysis as ia
import time
import shutil
from multiprocessing import Pool


def add_suffix(path, suffix):
    """
    add a suffix to file name of a given path

    :param path: original path
    :param suffix: str
    :return: new path
    """

    folder, file_name_full = os.path.split(path)
    file_name, file_ext = os.path.splitext(file_name_full)
    file_name_full_new = file_name + suffix + file_ext
    return os.path.join(folder, file_name_full_new)


def phase_correlation(img_match, img_ref):
    """
    openCV phase correction wrapper, as one of the align_func to perform motion correction. Open CV phaseCorrelate
    function returns (x_offset, y_offset). This wrapper switches the order of result and returns (height_offset,
    width_offset) to be more consistent with numpy indexing convention.

    :param img_match: the matching image
    :param img_ref: the reference image
    :return: rigid_transform coordinates of alignment (height_offset, width_offset)
    """

    x_offset, y_offset =  cv2.phaseCorrelate(img_match.astype(np.float32), img_ref.astype(np.float32))
    return [y_offset, x_offset]


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
            aligned_chunk[i, :, :] = curr_frame
            offset_list.append([0., 0.])

    new_mean_img = sum_img / float(valid_frame_num)

    return offset_list, aligned_chunk.astype(data_type), new_mean_img


def align_single_chunk_iterate(chunk, iteration=2, max_offset=(10., 10.), align_func=phase_correlation, fill_value=0.,
                               verbose=True):
    """
    align the frames in a single chunk of movie to its mean projection iteratively. for each iteration, the reference
    image (mean projection) will be updated based on the aligned chunk

    all operations will be applied with np.float32 format

    If the translation is larger than max_offset, the translation of that particular frame will be set as zero,
    and it will not be counted during the calculation of average projection image.

    :param chunk: a movie chunk, should be small enough to be managed in memory, 3d numpy.array
    :param iteration: number of iterations, int
    :param max_offset: maximum offset, (height, width), if single value, will be applied to both height and width
    :param align_func: function to align two image frames, return rigid transform offset (height, width)
    :param fill_value: value to fill empty pixels
    :param verbose:
    :return: alignment offset list, aligned movie chunk (same data type of original chunk), updated mean projection
    image np.float32
    """

    if iteration < 1:
        raise ValueError('iteration should be an integer larger than 0.')

    img_ref = np.mean(chunk.astype(np.float32), axis=0)
    offset_list = None
    aligned_chunk = None

    for i in range(iteration):

        print("\nMotion Correction, iteration " + str(i))
        offset_list, aligned_chunk, img_ref = align_single_chunk(chunk, img_ref, max_offset=max_offset,
                                                                 align_func=align_func, fill_value=fill_value,
                                                                 verbose=verbose)

    return offset_list, aligned_chunk, img_ref


# def align_single_chunk_iterate2(chunk, iteration=2, max_offset=(10., 10.), align_func=phase_correlation, fill_value=0.,
#                                 verbose=True):
#     """
#     align the frames in a single chunk of movie to its mean projection iteratively. for each iteration, the reference
#     image (mean projection) will be updated based on the aligned chunk
#
#     all operations will be applied with np.float32 format
#
#     If the translation is larger than max_offset, the translation of that particular frame will be set as zero,
#     and it will not be counted during the calculation of average projection image.
#
#     the difference between align_single_chunk_iterate and align_single_chunk_iterate2 is during iteration, the first
#     method only updates img_ref and for each iteration it aligns the original movie to the updated img_ref. it returns
#     the offsets generated by the last iteration. But the second method updates both img_ref and the movie itself, and
#     for each iteration it aligns the corrected movie form last iteration to the updated img_ref, and return the
#     accumulated offsets of all iterations.
#
#     :param chunk: a movie chunk, should be small enough to be managed in memory, 3d numpy.array
#     :param iteration: number of iterations, int
#     :param max_offset: maximum offset, (height, width), if single value, will be applied to both height and width
#     :param align_func: function to align two image frames, return rigid transform offset (height, width)
#     :param fill_value: value to fill empty pixels
#     :param verbose:
#     :return: alignment offset list, aligned movie chunk (same data type of original chunk), updated mean projection
#     image np.float32
#     """
#
#     if iteration < 1:
#         raise ValueError('iteration should be an integer larger than 0.')
#
#     img_ref = np.mean(chunk.astype(np.float32), axis=0)
#     offset_list = None
#     aligned_chunk = chunk
#
#     for i in range(iteration):
#
#         print "\nMotion Correction, iteration " + str(i)
#         curr_offset_list, aligned_chunk, img_ref = align_single_chunk(aligned_chunk, img_ref, max_offset=max_offset,
#                                                                       align_func=align_func, fill_value=fill_value,
#                                                                       verbose=verbose)
#
#         if offset_list is None:
#             offset_list = curr_offset_list
#         else:
#             for i in range(len(offset_list)):
#                 offset_list[i] = [offset_list[i][0] + curr_offset_list[i][0],
#                                   offset_list[i][1] + curr_offset_list[i][1]]
#
#     return offset_list, aligned_chunk, img_ref


def align_single_chunk_iterate_anchor(chunk, anchor_frame_ind=0, iteration=2, max_offset=(10., 10.),
                                      align_func=phase_correlation, fill_value=0., verbose=True):
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
    :param verbose:
    :return: alignment offset list, aligned movie chunk (same data type of original chunk), updated mean projection
    image np.float32
    """

    if iteration < 1:
        raise ValueError('iteration should be an integer larger than 0.')

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

    chunk_path, anchor_frame_ind, iteration, max_offset, align_func, fill_value, output_folder = params
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
                                                                            verbose=False)

    chunk_folder, chunk_fn = os.path.split(os.path.abspath(chunk_path))
    chunk_fn_n = os.path.splitext(chunk_fn)[0]

    if output_folder is None:
        output_folder= chunk_folder

    result_f = h5py.File(os.path.join(output_folder, 'temp_offsets_' + chunk_fn_n + '.hdf5'))
    offset_dset = result_f.create_dataset('offsets', data=offset_list)
    offset_dset.attrs['data_format'] = '[row, col]'
    result_f['mean_projection'] = img_ref
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

    if down_sample_rate is not None:
        mov_down = ia.z_downsample(mov_corr, down_sample_rate, is_verbose=False)
        print('\n\t{:09.2f} second; finished applying correction to movie: {}.'.format(time.time() - t0, mov_name))
        return mov_down
    else:
        print('\n\t{:09.2f} second; finished applying correction to movie: {}.'.format(time.time() - t0, mov_name))
        return


def align_multiple_files_iterate(paths, output_folder=None, is_output_mov=True, iteration=2, max_offset=(10., 10.),
                                 align_func=phase_correlation, fill_value=0., verbose=True, offset_file_name=None,
                                 mean_projections_file_name=None, mean_projection_file_name=None):

    """
    Motion correct a list of movie files (currently only support .tif format, designed for ScanImage output files.
    each files will be first aligned to its own mean projection iteratively. for each iteration, the reference
    image (mean projection) will be updated based on the aligned result. Then all files will be aligned based on their
    final mean projection images. Designed for movies recorded from ScanImage with dtype int16

    all operations will be applied with np.float32 format

    :param paths: list of paths of data files (currently only support .tif format), they should have same height and
                  width dimensions.
    :param output_folder: folder to save output, if None, a subfolder named "motion_correction" will be created in the
                          folder of the first paths in paths
    :param is_output_mov: bool, if True, aligned movie will be saved, if False, only correction offsets and final mean
                          projection image of all files will be saved
    :param iteration: int, number of iterations to correct each file
    :param max_offset: If the correction is larger than max_offset, the correction of that particular frame will be
                       set as zero, and it will not be counted during the calculation of average projection image.
    :param align_func: function to align two image frames, return rigid transform offset (height, width)
    :param fill_value: value to fill empty pixels
    :param verbose:
    :param offset_file_name: str, the file name of the saved offsets hdf5 file (without extension), if None,
                             default will be 'correction_offsets.hdf5'
    :param mean_projections_file_name: str, the file name of the saved mean projection image stack (without extension),
                                       one image for each file, order is same as 'path_list' field in saved offsets
                                       hdf5 file. If None, default will be 'corrected_mean_projections'
    :param mean_projection_file_name: str, the file name of the saved mean projection image (without extension), if
                                      None, default will be 'corrected_mean_projection'
    :return: offsets, dictionary of correction offsets. Key: path of file; value: list of tuple with correction offsets,
             (height, width)
    """

    if output_folder is None:
        main_folder, _ = os.path.split(paths[0])
        output_folder = os.path.join(main_folder, 'motion_correction')

    if not os.path.isdir(output_folder):
        print("\n\nOutput folder: " + str(output_folder) + "does not exist. Create new folder.")
        os.mkdir(output_folder)
    else:
        print("\n\nOutput folder: " + str(output_folder) + "already exists. Write into this folder.")
    os.chdir(output_folder)

    offsets = [] # list of local correction for each file
    mean_projections=[] # final mean projection of each file

    for path in paths:

        if verbose:
            print('\nCorrecting file: ' + str(path) + ' ...')

        curr_mov = tf.imread(path)
        offset, _, mean_projection = align_single_chunk_iterate(curr_mov, iteration=iteration, max_offset=max_offset,
                                                                align_func=align_func, fill_value=fill_value,
                                                                verbose=verbose)

        offsets.append(offset)
        mean_projections.append(mean_projection)

    mean_projections = np.array(mean_projections, dtype=np.float32)

    print('\n\nCorrected mean projection images of all files ...')
    mean_projection_offset, final_mean_projections, _= align_single_chunk_iterate(mean_projections, iteration=iteration,
                                                                                  max_offset=max_offset,
                                                                                  align_func=align_func,
                                                                                  fill_value=fill_value,
                                                                                  verbose=False)

    print('\nAdding global correction offset to local correction offsets and save.')
    if offset_file_name is None:
        h5_file = h5py.File(os.path.join(output_folder, 'correction_offsets.hdf5'))
    else:
        h5_file = h5py.File(os.path.join(output_folder, offset_file_name + '.hdf5'))
    h5_file.create_dataset('path_list', data=paths)
    offset_dict = {}
    for i in range(len(offsets)):
        curr_offset = offsets[i]
        curr_global_offset = mean_projection_offset[i]
        offsets[i] = [[offset[0] + curr_global_offset[0],
                       offset[1] + curr_global_offset[1]] for offset in curr_offset]
        curr_h5_dset = h5_file.create_dataset('file{:04d}'.format(i), data=offsets[i])
        curr_h5_dset.attrs['path'] = str(paths[i])
        curr_h5_dset.attrs['format'] = ['row', 'col']
        offset_dict.update({str(paths[i]):offsets[i]})
    h5_file.close()

    print('\nSaving final mean projection image.')
    if mean_projections_file_name is None:
        tf.imsave(os.path.join(output_folder, 'corrected_mean_projections.tif'),
                  final_mean_projections.astype(np.float32))
    else:
        tf.imsave(os.path.join(output_folder, mean_projections_file_name + '.tif'),
                  final_mean_projections.astype(np.float32))

    if mean_projection_file_name is None:
        tf.imsave(os.path.join(output_folder, 'corrected_mean_projection.tif'),
                  np.mean(final_mean_projections, axis=0).astype(np.float32))
    else:
        tf.imsave(os.path.join(output_folder, mean_projection_file_name + '.tif'),
                  np.mean(final_mean_projections, axis=0).astype(np.float32))

    if is_output_mov:
        for i, curr_path in enumerate(paths):
            print('\nFinal Correction of file: ' + curr_path)
            curr_mov = tf.imread(curr_path)
            curr_offset = offsets[i]
            curr_corrected_mov = correct_movie(curr_mov, curr_offset, fill_value=fill_value, verbose=verbose)

            try:  # try to write a movie of corrected images
                if i == 0:
                    codex = 'XVID'
                    fourcc = cv2.cv.CV_FOURCC(*codex)
                    out = cv2.VideoWriter('corrected_movie_' + codex + '.avi', fourcc, 30, (512, 512), isColor=False)
                    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

                # generate .avi movie of concatenated corrected movies
                for i, frame in enumerate(curr_corrected_mov):
                    if i == 0:
                        stack = [frame]

                    elif i % 20 == 0:

                        # convert to 8-bit gray scale
                        display_frame = np.mean(stack, axis=0).astype(np.float32)
                        display_frame = (display_frame - np.amin(display_frame)) / \
                                        (np.amax(display_frame) - np.amin(display_frame))
                        display_frame = (display_frame * 255).astype(np.uint8)

                        # # histogram normalization
                        # display_frame = clahe.apply(display_frame)
                        display_frame = cv2.equalizeHist(display_frame)

                        # write frame
                        out.write(display_frame)

                        # clear stack
                        stack = [frame]

                    else:
                        stack.append(frame)

            except Exception:
                pass

            _, curr_file_name = os.path.split(curr_path)
            curr_save_name = add_suffix(curr_file_name, '_corrected')
            print('Saving corrected file: ' + curr_save_name + ' ...')
            tf.imsave(os.path.join(output_folder, curr_save_name), curr_corrected_mov)

    cv2.destroyAllWindows()
    out.release()

    tf.imshow(final_mean_projections, cmap='gray', interpolation='nearest')
    plt.show()

    return offset_dict


def align_multiple_files_iterate_anchor(paths, output_folder=None, is_output_mov=True, anchor_frame_ind=0,
                                        iteration_chunk=6, iteration_projection=10,
                                        max_offset=(10., 10.), align_func=phase_correlation, fill_value=0.,
                                        verbose=True, offset_file_name=None, mean_projections_file_name=None,
                                        mean_projection_file_name=None):

    """
    Motion correct a list of movie files (currently only support .tif format, designed for ScanImage output files.
    each files will be first aligned to its own mean projection iteratively. for each iteration, the reference
    image (mean projection) will be updated based on the aligned result. Then all files will be aligned based on their
    final mean projection images. Designed for movies recorded from ScanImage with dtype int16

    this uses one iteration of anchor frame correction before the main correction.

    all operations will be applied with np.float32 format

    :param paths: list of paths of data files (currently only support .tif format), they should have same height and
                  width dimensions.
    :param output_folder: folder to save output, if None, a subfolder named "motion_correction" will be created in the
                          folder of the first paths in paths
    :param is_output_mov: bool, if True, aligned movie will be saved, if False, only correction offsets and final mean
                          projection image of all files will be saved
    :param anchor_frame_ind: non-negative int, frame number for anchor frame correction
    :param iteration_chunk: positive int, number of iterations to correct each file
    :param iteration_projection: positive int, number of iterations to correct mean projections
    :param max_offset: If the correction is larger than max_offset, the correction of that particular frame will be
                       set as zero, and it will not be counted during the calculation of average projection image.
    :param align_func: function to align two image frames, return rigid transform offset (height, width)
    :param fill_value: value to fill empty pixels
    :param verbose:
    :param offset_file_name: str, the file name of the saved offsets hdf5 file (without extension), if None,
                             default will be 'correction_offsets.hdf5'
    :param mean_projections_file_name: str, the file name of the saved mean projection image stack (without extension),
                                       one image for each file, order is same as 'path_list' field in saved offsets
                                       hdf5 file. If None, default will be 'corrected_mean_projections'
    :param mean_projection_file_name: str, the file name of the saved mean projection image (without extension), if
                                      None, default will be 'corrected_mean_projection'
    :return: offsets, dictionary of correction offsets. Key: path of file; value: list of tuple with correction offsets,
             (height, width)
    """

    if output_folder is None:
        main_folder, _ = os.path.split(paths[0])
        output_folder = os.path.join(main_folder, 'motion_correction')

    if not os.path.isdir(output_folder):
        print("\n\nOutput folder: " + str(output_folder) + " does not exist. Create this folder.")
        os.mkdir(output_folder)
    else:
        print("\n\nOutput folder: " + str(output_folder) + " already exists. Write into this folder.")
    os.chdir(output_folder)

    offsets = [] # list of local correction for each file
    mean_projections=[] # final mean projection of each file

    for path in paths:

        if verbose:
            print('\nCorrecting file: ' + str(path) + ' ...')

        curr_mov = tf.imread(path)
        offset, _, mean_projection = \
            align_single_chunk_iterate_anchor(curr_mov, anchor_frame_ind=anchor_frame_ind, iteration=iteration_chunk,
                                              max_offset=max_offset, align_func=align_func, fill_value=fill_value,
                                              verbose=verbose)

        offsets.append(offset)
        mean_projections.append(mean_projection)

    mean_projections = np.array(mean_projections, dtype=np.float32)

    print('\n\nCorrected mean projection images of all files ...')
    mean_projection_offset, final_mean_projections, _= \
        align_single_chunk_iterate_anchor(mean_projections, anchor_frame_ind=anchor_frame_ind,
                                          iteration=iteration_projection, max_offset=max_offset, align_func=align_func,
                                          fill_value=fill_value, verbose=False)

    print('\nAdding global correction offset to local correction offsets and save.')
    if offset_file_name is None:
        h5_file = h5py.File(os.path.join(output_folder, 'correction_offsets.hdf5'))
    else:
        h5_file = h5py.File(os.path.join(output_folder, offset_file_name + '.hdf5'))
    h5_file.create_dataset('path_list', data=paths)
    offset_dict = {}
    for i in range(len(offsets)):
        curr_offset = offsets[i]
        curr_global_offset = mean_projection_offset[i]
        offsets[i] = [[offset[0] + curr_global_offset[0],
                       offset[1] + curr_global_offset[1]] for offset in curr_offset]
        curr_h5_dset = h5_file.create_dataset('file{:04d}'.format(i), data=offsets[i])
        curr_h5_dset.attrs['path'] = str(paths[i])
        curr_h5_dset.attrs['format'] = ['row', 'col']
        offset_dict.update({str(paths[i]):offsets[i]})
    h5_file.close()

    print('\nSaving final mean projection image.')
    if mean_projections_file_name is None:
        tf.imsave(os.path.join(output_folder, 'corrected_mean_projections.tif'),
                  final_mean_projections.astype(np.float32))
    else:
        tf.imsave(os.path.join(output_folder, mean_projections_file_name + '.tif'),
                  final_mean_projections.astype(np.float32))

    if mean_projection_file_name is None:
        tf.imsave(os.path.join(output_folder, 'corrected_mean_projection.tif'),
                  np.mean(final_mean_projections, axis=0).astype(np.float32))
    else:
        tf.imsave(os.path.join(output_folder, mean_projection_file_name + '.tif'),
                  np.mean(final_mean_projections, axis=0).astype(np.float32))

    if is_output_mov:
        for i, curr_path in enumerate(paths):
            print('\nFinal Correction of file: ' + curr_path)
            curr_mov = tf.imread(curr_path)
            curr_offset = offsets[i]
            curr_corrected_mov = correct_movie(curr_mov, curr_offset, fill_value=fill_value, verbose=verbose)

            try:  # try to write a movie of corrected images
                if i == 0:
                    codex = 'XVID'
                    fourcc = cv2.cv.CV_FOURCC(*codex)
                    out = cv2.VideoWriter('corrected_movie_' + codex + '.avi', fourcc, 30, (512, 512), isColor=False)
                    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

                # generate .avi movie of concatenated corrected movies
                for i, frame in enumerate(curr_corrected_mov):
                    if i == 0:
                        stack = [frame]

                    elif i % 20 == 0:

                        # convert to 8-bit gray scale
                        display_frame = np.mean(stack, axis=0).astype(np.float32)
                        display_frame = (display_frame - np.amin(display_frame)) / \
                                        (np.amax(display_frame) - np.amin(display_frame))
                        display_frame = (display_frame * 255).astype(np.uint8)

                        # # histogram normalization
                        # display_frame = clahe.apply(display_frame)
                        display_frame = cv2.equalizeHist(display_frame)

                        # write frame
                        out.write(display_frame)

                        # clear stack
                        stack = [frame]

                    else:
                        stack.append(frame)

            except Exception:
                pass

            _, curr_file_name = os.path.split(curr_path)
            curr_save_name = add_suffix(curr_file_name, '_corrected')
            print('Saving corrected file: ' + curr_save_name + ' ...')
            tf.imsave(os.path.join(output_folder, curr_save_name), curr_corrected_mov)

    cv2.destroyAllWindows()
    out.release()

    tf.imshow(final_mean_projections, cmap='gray', interpolation='nearest')
    plt.show()

    return offset_dict


def align_multiple_files_iterate_anchor_multi_thread(f_paths,
                                                     output_folder,
                                                     process_num=1,
                                                     anchor_frame_ind_chunk=0,
                                                     anchor_frame_ind_projection=0,
                                                     iteration_chunk=6,
                                                     iteration_projection=10,
                                                     max_offset=(10., 10.),
                                                     align_func=phase_correlation,
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
    :return: None
    """

    correction_temp_folder = os.path.join(output_folder, 'correction_temp')
    if os.path.isdir(correction_temp_folder):
        shutil.rmtree(correction_temp_folder, ignore_errors=False)
        time.sleep(1.)
    os.mkdir(correction_temp_folder)
    print ('\ncorrection output will be saved in {}.'.format(os.path.abspath(output_folder)))

    print ('\naligning single chunks:')
    params_lst = [(f, anchor_frame_ind_chunk, iteration_chunk, max_offset, align_func, fill_value,
                   correction_temp_folder) for f in f_paths]
    # print '\n'.join([str(p) for p in params_lst])
    chunk_p = Pool(process_num)
    chunk_p.map(align_single_chunk_iterate_anchor_for_multi_thread, params_lst)

    print('\naligning among files ...')
    chunk_offset_fns = [f for f in os.listdir(correction_temp_folder) if f[0: 13] == 'temp_offsets_']
    chunk_offset_fns.sort()
    # print('\n'.join(chunk_offset_fns))
    mean_projections = []
    max_projections = []
    file_paths = []
    chunk_offsets = []
    for chunk_offset_fn in chunk_offset_fns:
        chunk_offset_f = h5py.File(os.path.join(correction_temp_folder, chunk_offset_fn))
        mean_projections.append(chunk_offset_f['mean_projection'].value)
        max_projections.append(chunk_offset_f['max_projection'].value)
        chunk_offsets.append(chunk_offset_f['offsets'].value)
        file_paths.append(chunk_offset_f['file_path'].value)

    _ = align_single_chunk_iterate_anchor(chunk=np.array(mean_projections),
                                          anchor_frame_ind=anchor_frame_ind_projection,
                                          iteration=iteration_projection,
                                          max_offset=max_offset,
                                          align_func=align_func,
                                          fill_value=0., verbose=False)
    offsets_chunk, mean_projections_c, mean_projection = _
    max_projections_c = correct_movie(mov=np.array(max_projections), offsets=offsets_chunk, fill_value=fill_value,
                                      verbose=False)
    max_projection = np.max(max_projections_c, axis=0)

    tf.imsave(os.path.join(output_folder, 'corrected_mean_projections.tif'), mean_projections_c)
    tf.imsave(os.path.join(output_folder, 'corrected_mean_projection.tif'), mean_projection)
    tf.imsave(os.path.join(output_folder, 'corrected_max_projections.tif'), max_projections_c)
    tf.imsave(os.path.join(output_folder, 'corrected_max_projection.tif'), max_projection)
    offsets_f = h5py.File(os.path.join(output_folder, "correction_offsets.hdf5"))
    for i, file_path in enumerate(file_paths):
        curr_chunk_offsets = chunk_offsets[i]
        curr_global_offset = offsets_chunk[i]
        offsets_dset = offsets_f.create_dataset('file_{:04d}'.format(i),
                                                data=curr_chunk_offsets + curr_global_offset)
        offsets_dset.attrs['format'] = ['height', 'width']
        offsets_dset.attrs['path'] = os.path.abspath(file_path)
    offsets_f.close()
    print ('\nchunks aligned offsets and projection images saved.')


def align_single_file(f_path, output_folder, anchor_frame_ind=0, iteration=6, max_offset=(10, 10),
                      align_func=phase_correlation, fill_value=0.):
    """

    :param f_path: str, path to the file to be corrected.
    :param output_folder: str, path to save correction results
    :param anchor_frame_ind: non-negative int, frame index for first correction iteration
    :param iteration: non-negative int, number of iterations to correct single file
    :param max_offset: tuple of two positive floats, (row, col), if the absolute value of the correction offsets of a
                       single frame is larger than this value, it will be set to zero.
    :param align_func: function object, the function to align two frames
    :param fill_value: float, value to fill the correction margin
    :return: None
    """

    print ('\ncorrection output will be saved in {}.'.format(os.path.abspath(output_folder)))

    print ('\naligning file: {} ...'.format(os.path.abspath(f_path)))
    mov = tf.imread(f_path)
    _ = align_single_chunk_iterate_anchor(mov, anchor_frame_ind=anchor_frame_ind, iteration=iteration,
                                          max_offset=max_offset, align_func=align_func, fill_value=fill_value,
                                          verbose=True)
    offset_list, aligned_chunk, img_ref = _
    tf.imsave(os.path.join(output_folder, 'corrected_mean_projection.tif'), img_ref)
    tf.imsave(os.path.join(output_folder, 'corrected_max_projection.tif'), np.max(aligned_chunk, axis=0))
    offsets_f = h5py.File(os.path.join(output_folder, "correction_offsets.hdf5"))
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
                      max_offset=(10., 10.),
                      align_func=phase_correlation,
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

        if os.path.isfile(os.path.join(output_folder, 'corrected_mean_projections.tif')):
            raise IOError('"corrected_mean_projections.tif" already exists in output folder.')
        if os.path.isfile(os.path.join(output_folder, 'corrected_mean_projection.tif')):
            raise IOError('"corrected_mean_projection.tif" already exists in output folder.')
        if os.path.isfile(os.path.join(output_folder, 'corrected_max_projections.tif')):
            raise IOError('"corrected_max_projections.tif" already exists in output folder.')
        if os.path.isfile(os.path.join(output_folder, 'corrected_max_projection.tif')):
            raise IOError('"corrected_max_projection.tif" already exists in output folder.')
        if os.path.isfile(os.path.join(output_folder, 'correction_offsets.hdf5')):
            raise IOError('"correction_offsets.hdf5" already exists in output folder.')

        if len(f_paths) == 1:

            align_single_file(f_path=f_paths[0], output_folder=output_folder, anchor_frame_ind=anchor_frame_ind_chunk,
                              iteration=iteration_chunk, max_offset=max_offset, align_func=align_func,
                              fill_value=fill_value)

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
                                                             max_offset=max_offset,
                                                             align_func=align_func,
                                                             fill_value=fill_value)
    return [os.path.abspath(f) for f in f_paths], os.path.abspath(output_folder)


def apply_correction_offsets(offsets_path,
                             path_pairs,
                             process_num=1,
                             fill_value=0.,
                             output_folder=None,
                             avi_downsample_rate=20):
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

    offsets_f = h5py.File(offsets_path)

    params_list = []
    for path_pair in path_pairs:
        target_path = path_pair[1]
        source_path = path_pair[0]
        offset = None
        for offset_dest in offsets_f.values():
            if offset_dest.attrs['path'] == source_path:
                offset = offset_dest.value
                break
        if offset is None:
            raise LookupError('can not find source file {} in the offsets file for target file: {}'
                              .format(source_path, target_path))
        params_list.append((target_path, offset, fill_value, output_folder, avi_downsample_rate))
    offsets_f.close()

    p_corr = Pool(process_num)
    mov_downs = p_corr.map(correct_movie_for_multi_thread, params_list)

    if avi_downsample_rate is not None:
        print ('\ngenerating .avi downsampled movie after correction ...')

        if cv2.__version__[0:3] == '3.1':
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

        for mov_down in mov_downs:
            for frame in mov_down:
                display_frame = frame.astype(np.float32)
                display_frame = (display_frame - np.amin(display_frame)) / \
                                (np.amax(display_frame) - np.amin(display_frame))
                display_frame = (display_frame * 255).astype(np.uint8)
                display_frame = cv2.equalizeHist(display_frame)
                out.write(display_frame)
        out.release()
        cv2.destroyAllWindows()
        print ('.avi moive generated.')
