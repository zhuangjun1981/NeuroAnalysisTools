import os
import shutil
import operator
import time
import io
from multiprocessing import Pool
import PIL
import h5py
import numpy as np
import tifffile as tf
import pandas as pd
import scipy.ndimage as ni
import matplotlib.pyplot as plt
import cv2

import NeuroAnalysisTools.core.FileTools as ft
import NeuroAnalysisTools.core.ImageAnalysis as ia
import NeuroAnalysisTools.core.PlottingTools as pt
import NeuroAnalysisTools.NwbTools as nt
import NeuroAnalysisTools.HighLevel as hl
import NeuroAnalysisTools.DeepLabCutTools as dlct
import NeuroAnalysisTools.MotionCorrection as mc
import NeuroAnalysisTools.DisplayLogTools as dt


def get_traces(params):

    """
    this is only a function used for multiprocessing in PlaneProcessor.get_raw_center_and_surround_traces()

    :param params:
    :return:
    """

    t0 = time.time()

    chunk_ind, chunk_start, chunk_end, nwb_path, data_path, curr_folder, center_array, surround_array = params

    nwb_f = h5py.File(nwb_path, 'r')
    print('\tstart analyzing chunk: {}'.format(chunk_ind))
    curr_mov = nwb_f[data_path][chunk_start: chunk_end]
    nwb_f.close()

    # print 'extracting traces'
    curr_traces_center = np.empty((center_array.shape[0], curr_mov.shape[0]), dtype=np.float32)
    curr_traces_surround = np.empty((center_array.shape[0], curr_mov.shape[0]), dtype=np.float32)
    for i in range(center_array.shape[0]):
        curr_center = ia.WeightedROI(center_array[i])
        curr_surround = ia.ROI(surround_array[i])
        curr_traces_center[i, :] = curr_center.get_weighted_trace_pixelwise(curr_mov)

        # scale surround trace to be similar as center trace
        mean_center_weight = curr_center.get_mean_weight()
        curr_traces_surround[i, :] = curr_surround.get_binary_trace_pixelwise(curr_mov) * mean_center_weight

    # print 'saveing chunk {} ...'.format(chunk_ind)
    chunk_folder = os.path.join(curr_folder, 'chunks')
    if not os.path.isdir(chunk_folder):
        os.mkdir(chunk_folder)
    chunk_f = h5py.File(os.path.join(chunk_folder, 'chunk_temp_' + ft.int2str(chunk_ind, 4) + '.hdf5'), 'x')
    chunk_f['traces_center'] = curr_traces_center
    chunk_f['traces_surround'] = curr_traces_surround
    chunk_f.close()

    print('\t\t{:06d} seconds: chunk: {}; demixing finished.'.format(int(time.time() - t0), chunk_ind))

    return None


def plot_traces_chunks(traces, labels, chunk_size, roi_ind):
    """
    for multiprocessing

    :param traces: np.array, shape=[trace_type, t_num]
    :param labels:
    :param chunk_size:
    :param figures_folder:
    :param roi_ind:
    :return:
    """

    t_num = traces.shape[1]
    chunk_num = t_num // chunk_size

    chunks = []
    for chunk_ind in range(chunk_num):
        chunks.append([chunk_ind * chunk_size, (chunk_ind + 1) * chunk_size])

    if t_num % chunk_size != 0:
        chunks.append([chunk_num * chunk_size, t_num])

    v_max = np.amax(traces)
    v_min = np.amin(traces)

    fig = plt.figure(figsize=(75, 20))
    fig.suptitle('neuropil subtraction for ROI: {}'.format(roi_ind))
    for chunk_ind, chunk in enumerate(chunks):
        curr_ax = fig.add_subplot(len(chunks), 1, chunk_ind + 1)
        for trace_ind in range(traces.shape[0]):
           curr_ax.plot(traces[trace_ind, chunk[0]: chunk[1]], label=labels[trace_ind])

        curr_ax.set_xlim([0, chunk_size])
        curr_ax.set_ylim([v_min, v_max * 1.2])
        curr_ax.legend()

    return fig


def plot_traces_for_multi_process(params):
    """
    for multiprocessing
    :param params:
    :return:
    """

    curr_traces, plot_chunk_size, roi_ind, figures_folder = params

    # print('roi_{:04d}'.format(roi_ind))

    curr_fig = plot_traces_chunks(traces=curr_traces,
                                  labels=['center', 'surround', 'subtracted'],
                                  chunk_size=plot_chunk_size,
                                  roi_ind=roi_ind)
    curr_fig.savefig(os.path.join(figures_folder, 'neuropil_subtraction_ROI_{:04d}.png'.format(roi_ind)))
    curr_fig.clear()
    plt.close(curr_fig)


def downsample_for_multiprocessing(params):
    nwb_path, dset_path, frame_start_i, frame_end_i, dr = params

    print('\t\tdownsampling frame {} - {}'.format(frame_start_i, frame_end_i))

    ff = h5py.File(nwb_path, 'r')
    chunk = ff[dset_path][frame_start_i:frame_end_i, :, :]
    ff.close()
    chunk_d = ia.z_downsample(chunk, downSampleRate=dr, is_verbose=False)
    return chunk_d


def downsample_mov(nwb_path, dset_path, dr, chunk_size, process_num):
    ff = h5py.File(nwb_path, 'r')
    frame_num = ff[dset_path].shape[0]
    print('\t\tshape of movie: {}'.format(ff[dset_path].shape))
    chunk_starts = np.array(range(0, frame_num, chunk_size))
    chunk_ends = chunk_starts + chunk_size
    chunk_ends[-1] = frame_num

    params = []
    for i, chunk_start in enumerate(chunk_starts):
        params.append((nwb_path, dset_path, chunk_start, chunk_ends[i], dr))

    p = Pool(process_num)
    mov_d = p.map(downsample_for_multiprocessing, params)

    return np.concatenate(mov_d, axis=0)


class Preprocessor(object):
    """
    pipeline to preprocess two-photon full session data

    this is very high-level, may break easily
    """

    def __init__(self):
        pass

    @staticmethod
    def copy_initial_files(source_folder, save_folder, date, mouse_id):

        print('\nCopying initial files to local folder')

        # copy notebook file
        if os.path.isfile(os.path.join(source_folder, 'notebook.txt')):
            print('\tCopying notebook.txt')
            shutil.copyfile(os.path.join(source_folder, 'notebook.txt'),
                            os.path.join(save_folder, 'notebook.txt'))
        else:
            print('\tCannot find notebook.txt. Skip.')

        # copy display log file
        display_log_fn = [fn for fn in os.listdir(source_folder) if date in fn
                          and mouse_id in fn and fn[-4:] == '.pkl']
        if len(display_log_fn) == 0:
            print('\tCannot find visual display log file. Skip.')
        elif len(display_log_fn) > 1:
            print('\tMore than one display log files found. Skip.')
        else:
            display_log_fn = display_log_fn[0]
            print('\tCopying visual display log file.')
            shutil.copy(os.path.join(source_folder, display_log_fn),
                        os.path.join(save_folder, display_log_fn))

        # copy sync file
        sync_fn = [fn for fn in os.listdir(source_folder) if date in fn
                   and mouse_id in fn and fn[-3:] == '.h5']
        if len(sync_fn) == 0:
            print('\tCannot find sync file. Skip.')
        elif len(sync_fn) > 1:
            print('\tMore than one sync files found. Skip.')
        else:
            sync_fn = sync_fn[0]
            print('\tCopying sync file.')
            shutil.copyfile(os.path.join(source_folder, sync_fn),
                            os.path.join(save_folder, sync_fn))

        # copy correction zstack
        if os.path.isdir(os.path.join(source_folder, 'correction_zstack')):
            print('\tCopying correction zstack files.')
            shutil.copytree(os.path.join(source_folder, 'correction_zstack'),
                            os.path.join(save_folder, 'correction_zstack'))
        else:
            print('\tCannot find correction_zstack folder. Skip.')

        # copy slm_patterns
        if os.path.isdir(os.path.join(source_folder, 'slm_patterns')):
            print('\tCopying slm pattern files.')
            shutil.copytree(os.path.join(source_folder, 'slm_patterns'),
                            os.path.join(save_folder, 'slm_patterns'))
        else:
            print('\tCannot find slm_patterns folder. Skip.')

        # copy videomon
        if os.path.isdir(os.path.join(source_folder, 'videomon')):
            print('\tCopying video monitoring files.')
            shutil.copytree(os.path.join(source_folder, 'videomon'),
                            os.path.join(save_folder, 'videomon'))
        else:
            print('\tCannot find videomon folder. Skip.')

    @staticmethod
    def get_vasmap_2p(data_folder, save_folder, scope, channels=('green', 'red'),
                      identifier='vasmap_2p', is_equalize=False,):

        print('\nReading vasmap_2p from folder:')
        print('\t{}'.format(data_folder))

        vasmaps = {}
        for chn in channels:
            vasmaps.update({chn: []})

        file_ns = [f for f in os.listdir(data_folder) if identifier in f]
        file_ns.sort()

        if len(file_ns) == 0:
            raise LookupError('Cannot find file.')
        else:
            [print('\t\t{}'.format(f)) for f in file_ns]

        for file_n in file_ns:
            curr_vasmap = tf.imread(os.path.join(data_folder, file_n))

            if len(curr_vasmap.shape) == 2:
                if len(channels) == 1:
                    vasmaps[channels[0]].append(np.array([curr_vasmap]))
                else:
                    raise ValueError(
                        'recorded file is 2d, cannot be deinterleved into {} channels.'.format(len(channels)))
            else:
                if len(curr_vasmap.shape) != 3:
                    raise ValueError('shape of recorded file: {}. should be either 2d or 3d.'.format(curr_vasmap.shape))

                for ch_i, ch_n in enumerate(channels):
                    curr_vasmap_ch = curr_vasmap[ch_i::len(channels)]
                    curr_vasmap_ch = ia.array_nor(np.mean(curr_vasmap_ch, axis=0))
                    if is_equalize:
                        curr_vasmap_ch = (curr_vasmap_ch * 255).astype(np.uint8)
                        curr_vasmap_ch = cv2.equalizeHist(curr_vasmap_ch).astype(np.float32)
                    vasmaps[ch_n].append(curr_vasmap_ch)


        vasmaps_final = {}
        for ch_n, ch_vasmap in vasmaps.items():
            save_vasmap = ia.array_nor(np.mean(ch_vasmap, axis=0)).astype(np.float32)

            if scope == 'scientifica':
                save_vasmap_r = save_vasmap[::-1, :].astype(np.float32)
                save_vasmap_r = ia.rigid_transform_cv2_2d(save_vasmap_r, rotation=135).astype(np.float32)
            elif scope == 'sutter':
                save_vasmap_r = save_vasmap.transpose()[::-1, :].astype(np.float32)
            elif scope == 'deepscope':
                save_vasmap_r = ia.rigid_transform_cv2(save_vasmap, rotation=140)[:, ::-1].astype(np.float32)
            else:
                raise LookupError("Do not understand scope type. Should be 'sutter' or 'deepscope' or 'scientifica'.")

            vasmaps_final['{}_original'.format(ch_n)] = save_vasmap
            vasmaps_final['{}_rotated'.format(ch_n)] = save_vasmap_r

            tf.imsave(os.path.join(save_folder, 'vasmap_2p_{}.tif'.format(ch_n)), save_vasmap)
            tf.imsave(os.path.join(save_folder, 'vasmap_2p_{}_rotated.tif'.format(ch_n)), save_vasmap_r)

        print('\tvasmap_2p saved.')

        return vasmaps_final

    @staticmethod
    def get_vasmap_wf(data_folder, save_folder, scope, identifier):
        """

        :param data_folder:
        :param save_folder:
        :param scope:
        :param identifier: str, 'JCam' for 'sutter' scope, 'vasmap_wf' for deepscope
        :return:
        """

        print('\nReading vasmap_2p from folder:')
        print('\t{}'.format(data_folder))

        vasmap_fns = [f for f in os.listdir(data_folder) if identifier in f]
        vasmap_fns.sort()
        if len(vasmap_fns) == 0:
            raise LookupError('Cannot find file.')
        else:
            [print('\t\t{}'.format(f)) for f in vasmap_fns]

        if scope == 'sutter':
            vasmaps = []
            for vasmap_fn in vasmap_fns:
                vasmap_focused, _, _ = ft.importRawJCamF(os.path.join(data_folder, vasmap_fn), column=1024, row=1024,
                                                         headerLength=116, tailerLength=452)  # try 452 if 218 does not work
                vasmap_focused = vasmap_focused[2:]
                vasmap_focused[vasmap_focused > 50000] = 400
                vasmap_focused = np.mean(vasmap_focused, axis=0)
                vasmaps.append(ia.array_nor(vasmap_focused))

            vasmap = ia.array_nor(np.mean(vasmaps, axis=0)).astype(np.float32)
            vasmap_r = vasmap[::-1, :].astype(np.float32)

            tf.imsave(os.path.join(save_folder, 'vasmap_wf.tif'), vasmap)
            tf.imsave(os.path.join(save_folder, 'vasmap_wf_rotated.tif'), vasmap_r)
            print('\tvasmap_2p saved.')

            return vasmap, vasmap_r

        elif scope == 'deepscope':
            vasmaps = []
            for vasmap_fn in vasmap_fns:
                curr_map = tf.imread(os.path.join(data_folder, vasmap_fn)).astype(np.float32)
                vasmaps.append(ia.array_nor(curr_map))

            vasmap = ia.array_nor(np.mean(vasmaps, axis=0)).astype(np.float32)
            vasmap_r = ia.array_nor(ia.rigid_transform_cv2(vasmap, rotation=140)[:, ::-1]).astype(np.float32)

            tf.imsave(os.path.join(save_folder, 'vasmap_wf.tif'), vasmap)
            tf.imsave(os.path.join(save_folder, 'vasmap_wf_rotated.tif'), vasmap_r)
            print('\tvasmap_wf saved.')

            return vasmap, vasmap_r

        else:
            raise LookupError('\tDo not understand scope type "{}", should be "sutter" or "deepscope"'.format(scope))

    @staticmethod
    def save_png_vasmap(tif_path, pre_fix, saturation_level=10.):
        """

        :param tif_path:
        :param pre_fix:
        :param saturation_level: float, percentile of pixel to be saturated
        :return:
        """

        save_folder, tif_fn = os.path.split(tif_path)
        fn = os.path.splitext(tif_fn)[0]

        vasmap = tf.imread(tif_path)
        f = plt.figure(figsize=(5, 5))
        ax = f.add_subplot(111)
        ax.imshow(vasmap, vmin=np.percentile(vasmap[:], saturation_level),
                  vmax=np.percentile(vasmap[:], 100-saturation_level), cmap='gray', interpolation='nearest')
        pt.save_figure_without_borders(f, savePath=os.path.join(save_folder,
                                                                '{}_{}.png'.format(pre_fix, fn)))
        plt.close(f)

    @staticmethod
    def check_deepscope_file_creation_time(data_folder, identifier):

        """
        plot operation system file creation time for each 2p file generated by deepscope
        you rarely need to use this.

        :param data_folder:
        :param identifier:
        :return:
        """

        fns = np.array([f for f in os.listdir(data_folder) if f[-4:] == '.tif' and identifier in f])
        f_nums = [int(os.path.splitext(fn)[0].split('_')[-2]) for fn in fns]
        fns = fns[np.argsort(f_nums)]
        print('total file number: {}'.format(len(fns)))

        ctimes = []

        for fn in fns:
            ctimes.append(os.path.getctime(os.path.join(data_folder, fn)))

        ctime_diff = np.diff(ctimes)
        max_ind = np.argmax(ctime_diff)
        print('maximum creation gap: {}'.format(ctime_diff[max_ind]))

        fis = np.arange(21, dtype=np.int) - 10 + max_ind

        for fi in fis:
            print('{}, ctime: {}s, duration: {}s'.format(fns[fi], ctimes[fi], ctime_diff[fi]))

        plt.plot(ctime_diff)
        plt.show()

    @staticmethod
    def check_deepscope_filename(data_folder, identifier):
        """
        check file name and file order consistency of the 2p files generated by deepscope
        you rarely need to use this.

        :param data_folder:
        :param identifier:
        :return:
        """

        fns = np.array([f for f in os.listdir(data_folder) if f[-4:] == '.tif' and identifier in f])
        f_nums = [int(os.path.splitext(fn)[0].split('_')[-2]) for fn in fns]
        fns = fns[np.argsort(f_nums)]
        print('total file number: {}'.format(len(fns)))

        for i in range(1, len(fns) + 1):

            if i < 100000:
                if fns[i - 1] != '{}_{:05d}_00001.tif'.format(identifier, i):
                    print('{}th file, name: {}, do not match!'.format(i, fns[i]))
                    break
            elif i < 1000000:
                if fns[i - 1] != '{}_{:06d}_00001.tif'.format(identifier, i):
                    print('{}th file, name: {}, do not match!'.format(i, fns[i]))
                    break
            elif i < 10000000:
                if fns[i - 1] != '{}_{:07d}_00001.tif'.format(identifier, i):
                    print('{}th file, name: {}, do not match!'.format(i, fns[i]))
                    break

    @staticmethod
    def reorganize_raw_2p_data(data_folder, save_folder, identifier, scope, plane_num,
                               channels, temporal_downsample_rate, frames_per_file, low_thr):
        """

        :param data_folder: str
        :param save_folder: str
        :param identifier: str
        :param scope: str
        :param plane_num: int, number of planes,
                          ignored if scope == 'sutter' (always just one plane)
                          can be more than 1 if scope == 'deepscope'
        :param channels: list of strings
        :param temporal_downsample_rate: int
        :param frames_per_file: int, frames per file after reorganization
        :param low_thr: float, threshold for minimum value of the movie
        :return:
        """

        if scope == 'sutter':

            print('\nReorganize raw two-photon data. Scope: {}.'.format(scope))

            file_list = [f for f in os.listdir(data_folder) if identifier in f and f[-4:] == '.tif']
            file_list.sort()
            [print('\t{}'.format(f)) for f in file_list]

            file_paths = [os.path.join(data_folder, f) for f in file_list]

            save_folders = []
            save_ids = [0 for ch in channels]
            total_movs = [None for ch in channels]
            for ch_n in channels:
                curr_save_folder = os.path.join(save_folder, identifier + '_reorged', 'plane0', ch_n)
                if not os.path.isdir(curr_save_folder):
                    os.makedirs(curr_save_folder)
                save_folders.append(curr_save_folder)

            for file_path in file_paths:
                print('\tprocessing {} ...'.format(os.path.split(file_path)[1]))

                curr_mov = tf.imread(file_path)
                curr_mov[curr_mov < low_thr] = low_thr

                if curr_mov.shape[0] % len(channels) != 0:
                    raise ValueError('\ttotal frame number of current movie ({}) cannot be divided by number of '
                                     'channels ({})!'.format(curr_mov.shape[0], len(channels)))

                # curr_mov = curr_mov.transpose((0, 2, 1))[:, ::-1, :]

                for ch_i, ch_n in enumerate(channels):
                    print('\t\tprocessing channel: {}'.format(ch_n))

                    curr_mov_ch = curr_mov[ch_i::len(channels)]

                    if total_movs[ch_i] is None:
                        total_movs[ch_i] = curr_mov_ch
                    else:
                        total_movs[ch_i] = np.concatenate((total_movs[ch_i], curr_mov_ch), axis=0)

                    while (total_movs[ch_i] is not None) and \
                            (total_movs[ch_i].shape[0] >= frames_per_file * temporal_downsample_rate):

                        num_file_to_save = int(total_movs[ch_i].shape[0] // (frames_per_file * temporal_downsample_rate))

                        for save_file_id in range(num_file_to_save):
                            save_chunk = total_movs[ch_i][save_file_id * (frames_per_file * temporal_downsample_rate):
                                                          (save_file_id + 1) * (frames_per_file * temporal_downsample_rate)]
                            save_path = os.path.join(save_folders[ch_i], '{}_{:05d}_reorged.tif'.format(identifier,
                                                                                                        save_ids[ch_i]))
                            if temporal_downsample_rate != 1:
                                print('\t\tdown sampling for {} ...'.format(os.path.split(save_path)[1]))
                                save_chunk = ia.z_downsample(save_chunk,
                                                             downSampleRate=temporal_downsample_rate,
                                                             is_verbose=False)

                            print('\t\tsaving {} ...'.format(os.path.split(save_path)[1]))
                            tf.imsave(save_path, save_chunk)
                            save_ids[ch_i] = save_ids[ch_i] + 1

                        if total_movs[ch_i].shape[0] % (frames_per_file * temporal_downsample_rate) == 0:
                            total_movs[ch_i] = None
                        else:
                            frame_num_left = total_movs[ch_i].shape[0] % (frames_per_file * temporal_downsample_rate)
                            total_movs[ch_i] = total_movs[ch_i][-frame_num_left:]

            print('\n\tprocessing residual frames ...')

            for ch_i, ch_n in enumerate(channels):

                if total_movs[ch_i] is not None:
                    print('\n\t\tprocessing channel: {}'.format(ch_n))

                    save_path = os.path.join(save_folders[ch_i],
                                             '{}_{:05d}_reorged.tif'.format(identifier, save_ids[ch_i]))

                    curr_mov_ch = total_movs[ch_i]

                    if temporal_downsample_rate != 1:
                        if curr_mov_ch.shape[0] % temporal_downsample_rate != 0:
                            warning_msg = '\t\tthe residual frame number ({}) cannot be divided ' \
                                          'by temporal down sample rate ({}).' \
                                          ' Drop last few frames.'.format(curr_mov_ch.shape[0],
                                                                          temporal_downsample_rate)
                            print(warning_msg)
                        print('\t\tdown sampling for {} ...'.format(os.path.split(save_path)[1]))
                        curr_mov_ch = ia.z_downsample(curr_mov_ch,
                                                      downSampleRate=temporal_downsample_rate,
                                                      is_verbose=False)

                    print('\t\tsaving {} ...'.format(os.path.split(save_path)[1]))
                    tf.imsave(save_path, curr_mov_ch)

            print('\n\tcopying non-tif files:')
            ntif_file_list = [f for f in os.listdir(data_folder) if identifier in f and f[-4:] != '.tif']
            ntif_save_folder = os.path.join(save_folder, identifier + '_reorged')
            if not os.path.isdir(ntif_save_folder):
                os.makedirs(ntif_save_folder)
            for ntif_f in ntif_file_list:
                shutil.copyfile(os.path.join(data_folder, ntif_f), os.path.join(ntif_save_folder, ntif_f))

            print('\nDone!')
        elif scope == 'deepscope':

            print('\nReorganize raw two-photon data...')
            print('\tScope: {}. '.format(scope))
            print('\tNumber of planes: {}'.format(plane_num))

            fns = np.array([f for f in os.listdir(data_folder) if f[-4:] == '.tif' and identifier in f])
            f_nums = [int(os.path.splitext(fn)[0].split('_')[-2]) for fn in fns]
            fns = fns[np.argsort(f_nums)]
            print('\ttotal file number: {}'.format(len(fns)))

            # print('\n'.join(fns))

            save_folders = []
            for i in range(plane_num):
                curr_save_folder = os.path.join(os.path.split(data_folder)[0], identifier + '_reorged',
                                                'plane{}'.format(i))
                if not os.path.isdir(curr_save_folder):
                    os.makedirs(curr_save_folder)
                save_folders.append(curr_save_folder)

            # frame_per_plane = len(fns) // plane_num
            for plane_ind in range(plane_num):
                print('\nprocessing plane: {}'.format(plane_ind))
                curr_fns = fns[plane_ind::plane_num]

                total_frames_down = len(curr_fns) // temporal_downsample_rate
                curr_fns = curr_fns[: total_frames_down * temporal_downsample_rate].reshape(
                    (total_frames_down, temporal_downsample_rate))

                print(curr_fns.shape)

                print('current file ind: 000')
                curr_file_ind = 0
                curr_frame_ind = 0
                curr_mov = {}
                for ch_n in channels:
                    curr_mov.update({ch_n: []})

                for fgs in curr_fns:

                    frame_grp = []

                    for fn in fgs:
                        cf = tf.imread(os.path.join(data_folder, fn))
                        # remove extreme negative pixels
                        cf[cf < low_thr] = low_thr
                        if len(cf.shape) == 2:
                            cf = np.array([cf])
                        frame_grp.append(cf)

                    curr_frame = {}

                    for ch_i, ch_n in enumerate(channels):
                        ch_frame_grp = np.array([f[ch_i::len(channels)][0] for f in frame_grp])
                        # print ch_frame_grp.shape
                        ch_frame = np.mean(ch_frame_grp, axis=0).astype(np.int16)
                        # ch_frame = ch_frame.transpose()[::-1, ::-1]
                        curr_frame.update({ch_n: ch_frame})

                    if curr_frame_ind < frames_per_file:

                        for ch_n in channels:
                            curr_mov[ch_n].append(curr_frame[ch_n])

                        curr_frame_ind = curr_frame_ind + 1

                    else:
                        for ch_n in channels:
                            curr_mov_ch = np.array(curr_mov[ch_n], dtype=np.int16)
                            save_name = '{}_{:05d}_reorged.tif'.format(identifier, curr_file_ind)
                            save_folder_ch = os.path.join(save_folders[plane_ind], ch_n)
                            if not os.path.isdir(save_folder_ch):
                                os.makedirs(save_folder_ch)
                            tf.imsave(os.path.join(save_folder_ch, save_name), curr_mov_ch)
                            curr_mov[ch_n] = [curr_frame[ch_n]]
                            print('\tcurrent file ind: {:05d}; channel: {}'.format(curr_file_ind, ch_n))
                        curr_file_ind += 1
                        curr_frame_ind = 1

                for ch_n in channels:
                    curr_mov_ch = np.array(curr_mov[ch_n], dtype=np.int16)
                    save_name = '{}_{:05d}_reorged.tif'.format(identifier, curr_file_ind)
                    save_folder_ch = os.path.join(save_folders[plane_ind], ch_n)
                    if not os.path.isdir(save_folder_ch):
                        os.makedirs(save_folder_ch)
                    tf.imsave(os.path.join(save_folder_ch, save_name), curr_mov_ch)
                    print('\tcurrent file ind: {:05d}; channel: {}'.format(curr_file_ind, ch_n))

            print('\n\tcopying non-tif files:')
            ntif_file_list = [f for f in os.listdir(data_folder) if identifier in f and f[-4:] != '.tif']
            ntif_save_folder = os.path.join(save_folder, identifier + '_reorged')
            if not os.path.isdir(ntif_save_folder):
                os.makedirs(ntif_save_folder)
            for ntif_f in ntif_file_list:
                shutil.copyfile(os.path.join(data_folder, ntif_f), os.path.join(ntif_save_folder, ntif_f))

            print('\tDone!')
        else:
            raise LookupError('Do not understand scope: "{}". Should be "sutter" or "deepscope".'.format(scope))

    @staticmethod
    def motion_correction(data_folder, reference_channel_name, apply_channel_names):
        """

        :param data_folder: str, path to the reorganized folder
        :param reference_channel_name: str, channel name for correction reference
        :param apply_channel_names: list of str, channel names for applying correction offsets
        """

        def correct(data_folder, ref_ch_n, apply_ch_ns):
            # ref_ch_n = 'green'
            # apply_ch_ns = ['green', 'red']

            curr_folder = os.path.dirname(os.path.realpath(__file__))
            os.chdir(curr_folder)

            ref_data_folder = os.path.join(data_folder, ref_ch_n)

            mc.motion_correction(input_folder=ref_data_folder,
                                 input_path_identifier='.tif',
                                 process_num=6,
                                 output_folder=os.path.join(ref_data_folder, 'corrected'),
                                 anchor_frame_ind_chunk=10,
                                 anchor_frame_ind_projection=0,
                                 iteration_chunk=10,
                                 iteration_projection=10,
                                 max_offset_chunk=(100., 100.),
                                 max_offset_projection=(100., 100.),
                                 align_func=mc.phase_correlation,
                                 preprocessing_type=6,
                                 fill_value=0.)

            offsets_path = os.path.join(ref_data_folder, 'corrected', 'correction_offsets.hdf5')
            ref_fns = [f for f in os.listdir(ref_data_folder) if f[-4:] == '.tif']
            ref_fns.sort()
            ref_paths = [os.path.join(ref_data_folder, f) for f in ref_fns]
            print('\nreference paths:')
            print('\n'.join(ref_paths))

            for apply_ch_i, apply_ch_n in enumerate(apply_ch_ns):
                apply_data_folder = os.path.join(data_folder, apply_ch_n)
                apply_fns = [f for f in os.listdir(apply_data_folder) if f[-4:] == '.tif']
                apply_fns.sort()
                apply_paths = [os.path.join(apply_data_folder, f) for f in apply_fns]
                print('\napply paths:')
                print('\n'.join(apply_paths))

                mc.apply_correction_offsets(offsets_path=offsets_path,
                                            path_pairs=zip(ref_paths, apply_paths),
                                            output_folder=os.path.join(apply_data_folder, 'corrected'),
                                            process_num=6,
                                            fill_value=0.,
                                            avi_downsample_rate=10,
                                            is_equalizing_histogram=False)

        plane_folders = [f for f in os.listdir(data_folder) if f[0:5] == 'plane' and
                         os.path.isdir(os.path.join(data_folder, f))]
        plane_folders.sort()
        print('folders to be corrected:')
        print('\n'.join(plane_folders))

        for plane_folder in plane_folders:
            correct(os.path.join(data_folder, plane_folder),
                    ref_ch_n=reference_channel_name,
                    apply_ch_ns=apply_channel_names)

    @staticmethod
    def reapply_motion_correction_multiplane(data_folder, reference_plane_name, reference_channel_name,
                                             apply_plane_names, apply_channel_names):
        """
        In case the motion correction failed on some planes but succeeded on one plane,
        or the universal correction to the whole column. This function allows you to apply
        the correction offsets from the good plane to other planes. Only applicable to
        multi-plane interleaved imaging.

        you rarely need to use this

        :param data_folder:
        :param reference_plane_name:
        :param reference_channle_name:
        :param apply_plane_names:
        :param apply_channel_names:
        :return:
        """

        ref_folder = os.path.join(data_folder, reference_plane_name, reference_channel_name)
        offsets_path = os.path.join(ref_folder, 'corrected', 'correction_offsets.hdf5')
        ref_paths = [f for f in os.listdir(ref_folder) if f[-4:] == '.tif']
        ref_paths.sort()
        ref_paths = [os.path.join(ref_folder, f) for f in ref_paths]
        print('\nreference paths:')
        print('\n'.join(ref_paths))

        for apply_plane_n in apply_plane_names:
            for apply_ch_n in apply_channel_names:
                print('\n\tapply to {}, channel: {}'.format(apply_plane_n, apply_ch_n))
                working_folder = os.path.join(data_folder, apply_plane_n, apply_ch_n)
                apply_paths = [f for f in os.listdir(working_folder) if f[-4:] == '.tif']
                apply_paths.sort()
                apply_paths = [os.path.join(working_folder, f) for f in apply_paths]
                print('\n\tapply paths:')
                print('\t' + '\n\t'.join(apply_paths))

                mc.apply_correction_offsets(offsets_path=offsets_path,
                                            path_pairs=zip(ref_paths, apply_paths),
                                            output_folder=os.path.join(working_folder, 'corrected'),
                                            process_num=6,
                                            fill_value=0.,
                                            avi_downsample_rate=10,
                                            is_equalizing_histogram=False)

    @staticmethod
    def downsample_corrected_files(data_folder, identifier, temporal_downsample_rate,
                                   frames_per_file=500):
        """
        temporally downsample motion corrected files.
        This is useful when motion correction is applied to the undownsampled the raw movie

        :param data_folder:
        :param identifier:
        :param temporal_downsample_rate:
        :param frames_per_file:
        :return:
        """

        print('Downsampling motion-corrected files in {} ...'.format(data_folder))

        if temporal_downsample_rate == 1:
            print('"post_correction_td_rate" is 1. No downsampling is needed. Skip.')
            return

        def downsample_folder(working_folder,
                              td_rate=temporal_downsample_rate,
                              file_identifier=identifier,
                              frames_per_file=frames_per_file):

            file_list = [f for f in os.listdir(working_folder) if file_identifier in f and f[-14:] == '_corrected.tif']
            file_list.sort()
            print('\t\tall files:')
            print('\n'.join(['\t\t' + f for f in file_list]))

            print('\n\t\tmoving files to "not_downsampled" folder:')
            file_paths = [os.path.join(working_folder, f) for f in file_list]
            print

            not_downsampled_folder = os.path.join(working_folder, 'not_downsampled')
            os.mkdir(not_downsampled_folder)
            for file_path in file_paths:
                fn = os.path.split(file_path)[1]
                shutil.move(file_path, os.path.join(not_downsampled_folder, fn))

            file_paths_original = [os.path.join(not_downsampled_folder, fn) for fn in file_list]
            file_paths_original.sort()

            save_id = 0
            total_mov = None
            for file_path_o in file_paths_original:
                print('\t\tprocessing {} ...'.format(os.path.split(file_path_o)[1]))
                curr_mov = tf.imread(file_path_o)

                if total_mov is None:
                    total_mov = curr_mov
                else:
                    total_mov = np.concatenate((total_mov, curr_mov), axis=0)

                while total_mov is not None and \
                        (total_mov.shape[0] >= frames_per_file * td_rate):

                    num_file_to_save = total_mov.shape[0] // (frames_per_file * td_rate)

                    for save_file_id in range(num_file_to_save):
                        save_chunk = total_mov[save_file_id * (frames_per_file * td_rate):
                                               (save_file_id + 1) * (frames_per_file * td_rate)]
                        save_path = os.path.join(working_folder,
                                                 '{}_{:05d}_corrected_downsampled.tif'.format(file_identifier,
                                                                                              save_id))
                        save_chunk = ia.z_downsample(save_chunk, downSampleRate=td_rate, is_verbose=False)

                        print('\t\t\tsaving {} ...'.format(os.path.split(save_path)[1]))
                        tf.imsave(save_path, save_chunk)
                        save_id = save_id + 1

                    if total_mov.shape[0] % (frames_per_file * td_rate) == 0:
                        total_mov = None
                    else:
                        frame_num_left = total_mov.shape[0] % (frames_per_file * td_rate)
                        total_mov = total_mov[-frame_num_left:]

            if total_mov is not None:
                save_path = os.path.join(working_folder,
                                         '{}_{:05d}_corrected_downsampled.tif'.format(file_identifier, save_id))
                save_chunk = ia.z_downsample(total_mov, downSampleRate=td_rate, is_verbose=False)
                print('\t\t\tsaving {} ...'.format(os.path.split(save_path)[1]))
                tf.imsave(save_path, save_chunk)

            return

        plane_ns = [f for f in os.listdir(data_folder) if f[0:5] == 'plane']
        plane_ns.sort()
        print('all planes:')
        _ = [print('\t{}'.format(pn)) for pn in plane_ns]

        for plane_n in plane_ns:
            print('\tcurrent plane: {}'.format(plane_n))
            plane_folder = os.path.join(data_folder, plane_n)
            ch_ns = [f for f in os.listdir(plane_folder)]
            ch_ns.sort()
            print('\t\tall channels: {}'.format(ch_ns))

            for ch_n in ch_ns:
                print('\n\t\tcurrent channel: {}'.format(ch_n))
                ch_folder = os.path.join(plane_folder, ch_n)

                downsample_folder(working_folder=os.path.join(ch_folder, 'corrected'),
                                  td_rate=temporal_downsample_rate,
                                  file_identifier=identifier,
                                  frames_per_file=frames_per_file)

        print('\nDone!')

    @staticmethod
    def copy_correction_results(data_folder, save_folder, apply_channel_name='green',
                                reference_channel_name='red'):

        print('\nCopying correction results.')

        plane_ns = [f for f in os.listdir(data_folder) if
                    os.path.isdir(os.path.join(data_folder, f)) and f[:5] == 'plane']
        plane_ns.sort()
        print('\tplanes:')
        _ = [print('\t\t{}'.format(p)) for p in plane_ns]

        for plane_n in plane_ns:
            plane_save_folder = os.path.join(save_folder, plane_n)
            if not os.path.isdir(plane_save_folder):
                os.makedirs(plane_save_folder)

            plane_folder = os.path.join(data_folder, plane_n, apply_channel_name, 'corrected')

            correction_files = [fn for fn in os.listdir(plane_folder) if fn[0:10] == 'corrected_']
            correction_files.sort()
            print('\t\tcopying:')
            _ = [print('\t\t\t{}'.format(fn)) for fn in correction_files]

            for cf in correction_files:
                shutil.copyfile(os.path.join(plane_folder, cf), os.path.join(plane_save_folder, cf))

            print('\t\tcopying correction offsets.')
            shutil.copyfile(os.path.join(data_folder, plane_n, reference_channel_name,
                                         'corrected', 'correction_offsets.hdf5'),
                            os.path.join(plane_save_folder, 'correction_offsets.hdf5'))

    @staticmethod
    def get_downsampled_small_movies(data_folder, save_folder, identifier, xy_downsample_rate=2,
                                     t_downsample_rate=10, channel_names=('green', 'red')):

        """
        mostly this will be used to heavily downsample motion corrected movies
        for visual inspection.

        :param data_folder:
        :param save_folder:
        :param identifier:
        :param xy_downsample_rate:
        :param t_downsample_rate:
        :param channel_names:
        :return:
        """

        print('generating downsampled small movies ...')

        plane_ns = [f for f in os.listdir(data_folder) if
                    os.path.isdir(os.path.join(data_folder, f)) and f[:5] == 'plane']
        plane_ns.sort()
        print('planes:')
        print('\n'.join(plane_ns))

        for plane_n in plane_ns:
            print('\nprocessing plane: {}'.format(plane_n))

            plane_save_folder = os.path.join(save_folder, plane_n)
            if not os.path.isdir(plane_save_folder):
                os.makedirs(plane_save_folder)

            for ch_n in channel_names:
                print('\n\tprocessing channel: {}'.format(ch_n))
                plane_folder = os.path.join(data_folder, plane_n, ch_n, 'corrected')

                # f_ns = [f for f in os.listdir(plane_folder) if f[-14:] == '_corrected.tif']
                f_ns = [f for f in os.listdir(plane_folder) if f[-4:] == '.tif' and identifier in f]
                f_ns.sort()
                print('\t\t' + '\n\t\t'.join(f_ns) + '\n')

                mov_d = []

                for f_n in f_ns:
                    print('\t\tprocessing {} ...'.format(f_n))
                    curr_mov = tf.imread(os.path.join(plane_folder, f_n))
                    curr_mov_d = ia.rigid_transform_cv2(img=curr_mov, zoom=(1. / xy_downsample_rate))
                    curr_mov_d = ia.z_downsample(curr_mov_d, downSampleRate=t_downsample_rate, is_verbose=False)
                    mov_d.append(curr_mov_d)

                mov_d = np.concatenate(mov_d, axis=0)
                save_n = '{}_{}_{}_downsampled.tif'.format(os.path.split(data_folder)[1], plane_n, ch_n)
                tf.imsave(os.path.join(plane_save_folder, save_n), mov_d)

        print('Done.')

    @staticmethod
    def get_2p_data_file_for_nwb(data_folder, save_folder, identifier, file_prefix, channel='green'):

        # file_prefix = '{}_{}_{}'.format(date_recorded, mouse_id, sess_id)

        print('Getting 2p data as an hdf5 file ready to be linked to nwb file ...')

        plane_fns = [f for f in os.listdir(data_folder) if f[:5] == 'plane']
        plane_fns.sort()
        print('\n'.join(plane_fns))

        data_f = h5py.File(os.path.join(save_folder, file_prefix + '_2p_movies.hdf5'))

        for plane_fn in plane_fns:
            print('\nprocessing {} ...'.format(plane_fn))
            plane_folder = os.path.join(data_folder, plane_fn, channel, 'corrected')
            # mov_fns = [f for f in os.listdir(plane_folder) if f[-14:] == '_corrected.tif']
            mov_fns = [f for f in os.listdir(plane_folder) if f[-4:] == '.tif' and identifier in f]
            mov_fns.sort()
            print('\n'.join(mov_fns))

            # get shape of concatenated movie
            z1, y, x = tf.imread(os.path.join(plane_folder, mov_fns[0])).shape
            z0, _, _ = tf.imread(os.path.join(plane_folder, mov_fns[-1])).shape
            z = z0 + z1 * (len(mov_fns) - 1)

            # for mov_fn in mov_fns:
            #     print('reading {} ...'.format(mov_fn))
            #     curr_z, curr_y, curr_x = tf.imread(os.path.join(plane_folder, mov_fn)).shape
            #
            #     if y is None:
            #         y = curr_y
            #     else:
            #         if y != curr_y:
            #             raise ValueError('y dimension ({}) of file "{}" does not agree with previous file(s) ({}).'
            #                              .format(curr_y, mov_fn, y))
            #
            #     if x is None:
            #         x = curr_x
            #     else:
            #         if x != curr_x:
            #             raise ValueError('x dimension ({}) of file "{}" does not agree with previous file(s) ({}).'
            #                              .format(curr_x, mov_fn, x))
            #
            #     z = z + curr_z

            # print((z, y, x))
            dset = data_f.create_dataset(plane_fn, (z, y, x), dtype=np.int16, compression='lzf')

            start_frame = 0
            end_frame = 0
            for mov_fn in mov_fns:
                print('reading {} ...'.format(mov_fn))
                curr_mov = tf.imread(os.path.join(plane_folder, mov_fn))
                end_frame = start_frame + curr_mov.shape[0]
                dset[start_frame: end_frame] = curr_mov
                start_frame = end_frame

            dset.attrs['conversion'] = 1.
            dset.attrs['resolution'] = 1.
            dset.attrs['unit'] = 'arbiturary_unit'

        data_f.close()

        print('Done.')

    @staticmethod
    def get_hdf5_files_for_caiman(data_folder, save_name_prefix, identifier, temporal_downsample_rate=3,
                                  channel_name='green'):

        print('Getting .hdf5 files for segmentation by caiman')

        plane_ns = [p for p in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, p))]
        plane_ns.sort()
        print('planes:')
        print('\n'.join(plane_ns))

        for plane_n in plane_ns:
            print('\nprocessing {} ...'.format(plane_n))

            plane_folder = os.path.join(data_folder, plane_n, channel_name, 'corrected')
            os.chdir(plane_folder)

            # f_ns = [f for f in os.listdir(plane_folder) if f[-14:] == '_corrected.tif']
            f_ns = [f for f in os.listdir(plane_folder) if f[-4:] == '.tif' and identifier in f]
            f_ns.sort()
            print('\n'.join(f_ns))

            mov_join = []
            for f_n in f_ns:
                print('processing plane: {}; file: {} ...'.format(plane_n, f_n))

                curr_mov = tf.imread(os.path.join(plane_folder, f_n))

                if curr_mov.shape[0] % temporal_downsample_rate != 0:
                    print('the frame number of {} ({}) is not divisible by t_downsample_rate ({}).'
                          .format(f_n, curr_mov.shape[0], temporal_downsample_rate))

                curr_mov_d = ia.z_downsample(curr_mov, downSampleRate=temporal_downsample_rate, is_verbose=False)
                mov_join.append(curr_mov_d)

            mov_join = np.concatenate(mov_join, axis=0)

            save_name = '{}_{}_downsampled_for_caiman.hdf5'.format(save_name_prefix, plane_n)
            save_f = h5py.File(os.path.join(plane_folder, save_name), 'w')
            save_f.create_dataset('mov', data=mov_join)
            save_f.close()

        print('done!')

    @staticmethod
    def get_mmap_files_for_caiman_bouton(data_folder, save_name_base, identifier, temporal_downsample_rate=5,
                                         channel_name='green'):

        print('Getting .mmap files for bouton segmentation by caiman v1.0')

        plane_ns = [p for p in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, p))]
        plane_ns.sort()
        print('planes:')
        print('\n'.join(plane_ns))

        for plane_n in plane_ns:
            print('\nprocessing {} ...'.format(plane_n))

            plane_folder = os.path.join(data_folder, plane_n, channel_name, 'corrected')
            os.chdir(plane_folder)

            # f_ns = [f for f in os.listdir(plane_folder) if f[-14:] == '_corrected.tif']
            f_ns = [f for f in os.listdir(plane_folder) if f[-4:] == '.tif' and identifier in f]
            f_ns.sort()
            print('\n'.join(f_ns))

            mov_join = []
            for f_n in f_ns:
                print('processing plane: {}; file: {} ...'.format(plane_n, f_n))

                curr_mov = tf.imread(os.path.join(plane_folder, f_n))

                if curr_mov.shape[0] % temporal_downsample_rate != 0:
                    print('the frame number of {} ({}) is not divisible by t_downsample_rate ({}).'
                          .format(f_n, curr_mov.shape[0], temporal_downsample_rate))

                curr_mov_d = ia.z_downsample(curr_mov, downSampleRate=temporal_downsample_rate,
                                             is_verbose=False)
                mov_join.append(curr_mov_d)

            mov_join = np.concatenate(mov_join, axis=0)
            add_to_mov = 10 - np.amin(mov_join)

            save_name = '{}_d1_{}_d2_{}_d3_1_order_C_frames_{}_.mmap' \
                .format(save_name_base, mov_join.shape[2], mov_join.shape[1], mov_join.shape[0])

            mov_join = mov_join.reshape((mov_join.shape[0], mov_join.shape[1] * mov_join.shape[2]),
                                        order='F').transpose()
            mov_join_mmap = np.memmap(os.path.join(plane_folder, save_name), shape=mov_join.shape, order='C',
                                      dtype=np.float32, mode='w+')
            mov_join_mmap[:] = mov_join + add_to_mov
            mov_join_mmap.flush()
            del mov_join_mmap

            save_file = h5py.File(os.path.join(plane_folder, 'caiman_segmentation_results.hdf5'), 'w')
            save_file['bias_added_to_movie'] = add_to_mov
            save_file.close()

        print('done!')

    @staticmethod
    def generate_nwb_file(save_folder, date, mouse_id, session_id='unknown', experimenter='unknown', genotype='unknown',
                          sex='unknown', age='unknown', indicator='unknown', imaging_rate='unknown',
                          imaging_depth='unknown', imaging_location='unknown', imaging_device='unknown',
                          imaging_excitation_lambda='unknown'):

        print('\nGenerating .nwb file.')

        notebook_path = os.path.join(save_folder, 'notebook.txt')
        if os.path.isfile(notebook_path):
            with open(notebook_path, 'r') as ff:
                notes = ff.read()
        else:
            print('Cannot find notebook.txt in {}. Skip.'.format(os.path.realpath(notebook_path)))

        general = nt.DEFAULT_GENERAL
        general['experimenter'] = experimenter
        general['subject']['subject_id'] = mouse_id
        general['subject']['genotype'] = genotype
        general['subject']['sex'] = sex
        general['subject']['age'] = age
        general['optophysiology'].update({'imaging_plane_0': {}})
        general['optophysiology']['imaging_plane_0'].update({'indicator': indicator})
        general['optophysiology']['imaging_plane_0'].update({'imaging_rate': imaging_rate})
        general['optophysiology']['imaging_plane_0'].update({'imaging_depth': imaging_depth})
        general['optophysiology']['imaging_plane_0'].update({'location': imaging_location})
        general['optophysiology']['imaging_plane_0'].update({'device': imaging_device})
        general['optophysiology']['imaging_plane_0'].update({'excitation_lambda': imaging_excitation_lambda})
        general['notes'] = notes

        file_name = '{}_M{}_{}.nwb'.format(date, mouse_id, session_id)

        rf = nt.RecordedFile(os.path.join(save_folder, file_name), identifier=file_name[:-4], description='')
        rf.add_general(general=general)
        rf.close()

        print('Done.')

    @staticmethod
    def get_nwb_path(nwb_folder):
        nwb_fns = [fn for fn in os.listdir(nwb_folder) if fn[-4:] == '.nwb']
        if len(nwb_fns) == 0:
            print('Cannot find .nwb files in {}!'.format(os.path.realpath(nwb_folder)))
            return None
        elif len(nwb_fns) > 1:
            print('More than one .nwb files found in {}!'.format(os.path.realpath(nwb_folder)))
            return None
        elif len(nwb_fns) == 1:
            return os.path.realpath(os.path.join(nwb_folder, nwb_fns[0]))

    @staticmethod
    def get_display_log(display_log_folder,
                        # visual_stim_software='WarpedVisualStim'
                        ):

        stim_pkl_fns = [f for f in os.listdir(display_log_folder) if f[-4:] == '.pkl']
        if len(stim_pkl_fns) == 0:
            print('\t\tCannot find visal stim log .pkl file. Skip.')
            return None
        elif len(stim_pkl_fns) > 1:
            print('\t\tFound more than one visual stim log .pkl files. Skip.')
            return None
        else:
            try:
                display_log = dt.DisplayLogAnalyzer(stim_pkl_fns[0])
            except Exception as e:
                print('\t\tCannot read visual display log.')
                print('\t\t{}'.format(e))
                display_log = None

            return display_log

    def add_vasmap_to_nwb(self, nwb_folder, is_plot=False):

        print('\nAdding vasculature images to .nwb file.')
        vasmap_dict = {
            'vasmap_wf': 'wide field surface vasculature map through cranial window original',
            'vasmap_wf_rotated': 'wide field surface vasculature map through cranial window rotated',
            'vasmap_2p_green': '2p surface vasculature map through cranial window green original, zoom1',
            'vasmap_2p_green_rotated': '2p surface vasculature map through cranial window green rotated, zoom1',
            'vasmap_2p_red': '2p surface vasculature map through cranial window red original, zoom1',
            'vasmap_2p_red_rotated': '2p surface vasculature map through cranial window red rotated, zoom1'
        }

        nwb_path = self.get_nwb_path(nwb_folder)
        nwb_f = nt.RecordedFile(nwb_path)

        for mn, des in vasmap_dict.items():
            try:
                curr_m = ia.array_nor(tf.imread(mn + '.tif'))

                if is_plot:
                    f = plt.figure(figsize=(10, 10))
                    ax = f.add_subplot(111)
                    ax.imshow(curr_m, vmin=0., vmax=1., cmap='gray', interpolation='nearest')
                    ax.set_axis_off()
                    ax.set_title(mn)
                    plt.show()

                print('\tadding {} to nwb file.'.format(mn))
                nwb_f.add_acquisition_image(mn, curr_m, description=des)

            except Exception as e:
                print(e)

        nwb_f.close()
        print('\tDone.')

    def add_sync_data_to_nwb(self, nwb_folder, sync_identifier):

        print('\nAdding sync data to .nwb file.')
        nwb_path = self.get_nwb_path(nwb_folder)

        sync_fn = [f for f in os.listdir(nwb_folder) if f[-3:] == '.h5' and sync_identifier in f]
        if len(sync_fn) == 0:
            raise LookupError('Did not find sync .h5 file ...')
        elif len(sync_fn) > 1:
            raise LookupError('More than one sync .h5 files found.')
        else:
            sync_fn = sync_fn[0]

        nwb_f = nt.RecordedFile(nwb_path)
        nwb_f.add_sync_data(sync_fn)
        nwb_f.close()

        print('\tDone.')

    def get_photodiode_onset_in_nwb(self, nwb_folder, digitize_thr, filter_size, segment_thr,
                                    smallest_interval, is_interact):
        """
        using the analog photodiode signal to extract the timing of indicator onset
        please check NeuroAnalysisTools.HighLevel.segmentPhotodiodeSignal function

        :param nwb_folder: str
        :param digitize_thr: float, initial digitize threshold of the analog signal
                              sutter scope, try -0.15
                              deepscope, try 0.15
        :param filter_size: float, seconds, gaussian filter sigma to filter digitized signal
        :param segment_thr: float, threshold to segment filtered signal
        :param smallest_interval: float, seconds, smallest allowed the intervals between two onsets
        :param is_interact: bool, interact through keyboard or not
        :return:
        """
        print('\nGenerating photodiode onset timestamps.')

        nwb_path = self.get_nwb_path(nwb_folder)

        nwb_f = nt.RecordedFile(nwb_path)
        pd, pd_t = nwb_f.get_analog_data(ch_n='analog_photodiode')
        fs = 1. / np.mean(np.diff(pd_t))
        # print fs

        pd_onsets = hl.segmentPhotodiodeSignal(pd, digitizeThr=digitize_thr, filterSize=filter_size,
                                               segmentThr=segment_thr, Fs=fs, smallestInterval=smallest_interval)

        if is_interact:
            input('press enter to continue ...')

        pdo_ts = nwb_f.create_timeseries('TimeSeries', 'digital_photodiode_rise', modality='other')
        pdo_ts.set_time(pd_onsets)
        pdo_ts.set_data([], unit='', conversion=np.nan, resolution=np.nan)
        pdo_ts.set_value('digitize_threshold', digitize_thr)
        pdo_ts.set_value('filter_size', filter_size)
        pdo_ts.set_value('segment_threshold', segment_thr)
        pdo_ts.set_value('smallest_interval', smallest_interval)
        pdo_ts.set_description('Real Timestamps (master acquisition clock) of photodiode onset. '
                               'Extracted from analog photodiode signal by the function:'
                               'corticalmapping.HighLevel.segmentPhotodiodeSignal() using parameters saved in the'
                               'current timeseries.')
        pdo_ts.set_path('/analysis')
        pdo_ts.finalize()

        nwb_f.close()

    def add_2p_image_to_nwb(self, nwb_folder, image_identifier, zoom, scope, plane_ns,
                            plane_depths, temporal_downsample_rate):
        """

        :param nwb_folder: str, folder containing .nwb file and the packaged 2p imaging .hdf5 file
        :param image_identifier: str, for finding the the packaged 2p imaging .hdf5 file
        :param zoom: float, zoom of 2p recording
        :param scope: str, 'sutter' or 'deepscope'
        :param plane_ns: list of str, ['plane0', 'plane1', 'plane2', ...]
        :param plane_depths: list of numbers, depth below pia for each imaging plane in microns
                             the length of this list should match the number of imaging planes
        :param temporal_downsample_rate: int, total temporal downsample rate,
                                         td_rate_before_correction * tdrate_after_correction
        :return:
        """

        print('\nAdding 2p imaging data to .nwb file ...')

        if len(plane_ns) != len(plane_depths):
            raise ValueError('Length of "plane_ns" ({}) should match length of '
                             '"plane_depths" ({}).'.format(len(plane_ns), len(plane_depths)))

        nwb_path = self.get_nwb_path(nwb_folder)

        description = '2-photon imaging data'

        if scope == 'deepscope':
            pixel_size = 0.0000009765 / zoom  # meter
        elif scope == 'sutter':
            pixel_size = 0.0000014 / zoom  # meter
        else:
            raise LookupError('do not understand scope type')

        nwb_f = nt.RecordedFile(nwb_path)

        ts_2p_tot = nwb_f.file_pointer['/acquisition/timeseries/digital_vsync_2p_rise/timestamps'][()]

        # if scope == 'sutter':
        #     ts_2p_tot = nwb_f.file_pointer['/acquisition/timeseries/digital_vsync_2p_rise/timestamps'].value
        # elif scope == 'DeepScope':
        #     ts_2p_tot = nwb_f.file_pointer['/acquisition/timeseries/digital_2p_vsync_rise/timestamps'].value
        # else:
        #     raise LookupError('do not understand scope type')
        # print('total 2p timestamps count: {}'.format(len(ts_2p_tot)))

        mov_fn = os.path.join(nwb_folder, '{}_2p_movies.hdf5'.format(image_identifier))
        mov_f = h5py.File(mov_fn, 'r')

        for mov_i, mov_dn in enumerate(plane_ns):

            if mov_dn is not None:

                curr_dset = mov_f[mov_dn]
                if mov_dn is not None:
                    mov_ts = ts_2p_tot[mov_i::len(plane_ns)]
                    print('\n{}: total 2p timestamps count: {}'.format(mov_dn, len(mov_ts)))

                    mov_ts_d = mov_ts[::temporal_downsample_rate]
                    print('{}: downsampled 2p timestamps count: {}'.format(mov_dn, len(mov_ts_d)))
                    print('{}: downsampled 2p movie frame num: {}'.format(mov_dn, curr_dset.shape[0]))

                    # if len(mov_ts_d) == curr_dset.shape[0]:
                    #     pass
                    # elif len(mov_ts_d) == curr_dset.shape[0] + 1:
                    #     mov_ts_d = mov_ts_d[0: -1]
                    # else:
                    #     raise ValueError('the timestamp count of {} movie ({}) does not equal (or is not greater by one) '
                    #                      'the frame cound in the movie ({})'.format(mov_dn, len(mov_ts_d), curr_dset.shape[0]))
                    mov_ts_d = mov_ts_d[:curr_dset.shape[0]]

                    curr_description = '{}. Imaging depth: {} micron.'.format(description, plane_depths[mov_i])
                    nwb_f.add_acquired_image_series_as_remote_link('2p_movie_' + mov_dn, image_file_path=mov_fn,
                                                                   dataset_path=mov_dn, timestamps=mov_ts_d,
                                                                   description=curr_description, comments='',
                                                                   data_format='zyx',
                                                                   pixel_size=[pixel_size, pixel_size],
                                                                   pixel_size_unit='meter')

        mov_f.close()
        nwb_f.close()

    def add_motion_correction_module_to_nwb(self, nwb_folder, movie_fn, plane_num, post_correction_td_rate):

        """

        :param nwb_folder:
        :param movie_fn: str, this movie has to be in the "nwb_folder" and is most likely generated by
                         self.get_2p_data_file_for_nwb method
        :param plane_num:
        :param post_correction_td_rate: int, downsample rate after motion correction
        :return:
        """

        print('\nAdding motion correction module.')
        nwb_path = self.get_nwb_path(nwb_folder)

        input_parameters = []

        for i in range(plane_num):

            plane_n = 'plane{}'.format(i)

            offsets_path = os.path.join(nwb_folder, plane_n, 'correction_offsets.hdf5')
            offsets_f = h5py.File(offsets_path, 'r')
            offsets_keys = list(offsets_f.keys())
            if 'path_list' in offsets_keys:
                offsets_keys.remove('path_list')

            offsets_keys.sort()
            offsets = []
            for offsets_key in offsets_keys:
                offsets.append(offsets_f[offsets_key][()])
            offsets = np.concatenate(offsets, axis=0)
            offsets = np.array([offsets[:, 1], offsets[:, 0]]).transpose()
            offsets_f.close()

            mean_projection = tf.imread(os.path.join(nwb_folder, plane_n, 'corrected_mean_projection.tif'))
            max_projection = tf.imread(os.path.join(nwb_folder, plane_n, 'corrected_max_projections.tif'))
            max_projection = ia.array_nor(np.max(max_projection, axis=0))

            input_dict = {'field_name': plane_n,
                          'original_timeseries_path': '/acquisition/timeseries/2p_movie_plane' + str(i),
                          'corrected_file_path': movie_fn,
                          'corrected_dataset_path': plane_n,
                          'xy_translation_offsets': offsets,
                          'mean_projection': mean_projection,
                          'max_projection': max_projection,
                          'description': '',
                          'comments': '',
                          'source': ''}

            input_parameters.append(input_dict)

        nwb_f = nt.RecordedFile(nwb_path)

        nwb_f.add_muliple_dataset_to_motion_correction_module(input_parameters=input_parameters,
                                                              module_name='motion_correction',
                                                              temporal_downsample_rate=post_correction_td_rate)
        nwb_f.close()
        print('\tDone.')

    def add_visual_stimuli_to_nwb(self, nwb_folder):
        """

        :param visual_stim_software: str, 'retinotopic_mapping' (python 2.7) or 'WrapedVisualStim' (python 3)
        :return:
        """
        print('\nAdding visual stimulation log to nwb file.')

        nwb_path = self.get_nwb_path(nwb_folder=nwb_folder)

        display_log = self.get_display_log(display_log_folder=nwb_folder)
        if display_log is not None:
            nwb_f = nt.RecordedFile(nwb_path)
            nwb_f.add_visual_display_log_retinotopic_mapping(stim_log=display_log)
            nwb_f.close()

        print('\tDone.')

    def analyze_visual_display_in_nwb(self, nwb_folder, vsync_frame_path,
                                      photodiode_ts_path='analysis/digital_photodiode_rise',
                                      photodiode_thr=-0.5,
                                      ccg_t_range=(0., 0.1),
                                      ccg_bins=100,
                                      is_plot=True):
        """
        calculating display lag and get the right visual stim timing

        :param nwb_folder:
        :param vsync_frame_path:
        :param visual_stim_software:
        :param photodiode_ts_path:
        :param photodiode_thr:
        :param ccg_t_range:
        :param ccg_bins:
        :param is_plot:
        :return:
        """

        print('\nAnalyzing visual stimulation log and photodiode onsets in the nwb file.')

        nwb_path = self.get_nwb_path(nwb_folder=nwb_folder)

        display_log = self.get_display_log(display_log_folder=nwb_folder)

        if display_log is not None:
            nwb_f = nt.RecordedFile(nwb_path)

            # get display lag
            display_delay = nwb_f.get_display_delay_retinotopic_mapping(stim_log=display_log,
                                                                        indicator_color_thr=photodiode_thr,
                                                                        ccg_t_range=ccg_t_range, ccg_bins=ccg_bins,
                                                                        is_plot=is_plot,
                                                                        pd_onset_ts_path=photodiode_ts_path,
                                                                        vsync_frame_ts_path=vsync_frame_path)

            # analyze photodiode onset
            stim_dict = display_log.get_stim_dict()
            pd_onsets_seq = display_log.analyze_photodiode_onsets_sequential(stim_dict=stim_dict,
                                                                             pd_thr=photodiode_thr)
            pd_onsets_com = display_log.analyze_photodiode_onsets_combined(pd_onsets_seq=pd_onsets_seq,
                                                                        is_dgc_blocked=True)
            nwb_f.add_photodiode_onsets_combined_retinotopic_mapping(pd_onsets_com=pd_onsets_com,
                                                                     display_delay=display_delay,
                                                                     vsync_frame_path=vsync_frame_path)
            nwb_f.close()

        print('\tDone.')

    def fit_ellipse_deeplabcut(self, eyetracking_folder, confidence_thr, point_num_thr, ellipse_fit_function):
        """

        :param eyetracking_folder:
        :param confidence_thr:
        :param point_num_thr:
        :param ellipse_fit_function:
        :return:
        """

        dlc_result_fns = [fn for fn in os.listdir(eyetracking_folder) if fn[-3:] == '.h5' and
                          'DLC_resnet50_universal_eye_tracking' in fn]
        if len(dlc_result_fns) == 0:
            print('\tNo deeplabcut result file found. Skip.')
            return
        elif len(dlc_result_fns) > 1:
            print('\tMore than one deeplabcut resut files found. Skip.')
            return
        else:
            dlc_result_fn = dlc_result_fns[0]

            df_pts = dlct.read_data_file(os.path.join(eyetracking_folder, dlc_result_fn))

            df_ell = dlct.get_all_ellipse(df_pts=df_pts, lev_thr=confidence_thr, num_thr=point_num_thr,
                                          fit_func=ellipse_fit_function)

            save_path = os.path.join(eyetracking_folder, os.path.splitext(dlc_result_fn)[0] + '_ellipse.hdf5')

            if os.path.isfile(save_path):
                raise IOError('Ellipse fit results already exists. Path: \n{}'.format(os.path.realpath(save_path)))

            save_f = h5py.File(save_path, 'x')
            dset = save_f.create_dataset('ellipse', data=np.array(df_ell))
            dset.attrs['columns'] = list(df_ell.columns)
            save_f.close()

            return save_path

    def add_eyetracking_to_nwb_deeplabcut(self, nwb_folder, eyetracking_folder, confidence_thr,
                                          point_num_thr, ellipse_fit_function,
                                          is_generate_labeled_movie, side,
                                          nasal_dir, diagonal_length,
                                          eyetracking_ts_name):
        """

        :param nwb_folder: str, folder of nwb file
        :param eyetracking_folder: str, folder of deeplabcut eyetracking folder
        :param confidence_thr: float, [0., 1.], threshold of confidence for including points for fitting,
                               check NeuroAnalysisTools.DeepLabCutTools.fit_ellipse() function
        :param point_num_thr: int, [5, 12], threhold of number of points for each ellipse fitting
                              check NeuroAnalysisTools.DeepLabCutTools.fit_ellipse() function
        :param ellipse_fit_function: cv2 function object, opencv function for ellipse fitting
                                     check NeuroAnalysisTools.DeepLabCutTools.fit_ellipse() function
        :param side: str, the side of the eye, 'left' or 'right', mostly 'right'
        :param nasal_dir: str, the side of nasal direction in the movie, 'left' or 'right'
        :param diagonal_length: float, mm, the length of diagonal line of eyetracking field of view
        :return:
        """

        print('\nAdding deeplabcut eye tracking results to nwb file.')

        nwb_path = self.get_nwb_path(nwb_folder=nwb_folder)
        if nwb_path is None:
            print('No nwb file found. Skip.')
            return

        fit_result_path = self.fit_ellipse_deeplabcut(eyetracking_folder=eyetracking_folder,
                                                      confidence_thr=confidence_thr,
                                                      point_num_thr=point_num_thr,
                                                      ellipse_fit_function=ellipse_fit_function,)

        if fit_result_path is not None:

            ell_f = h5py.File(fit_result_path, 'r')
            ell_data = ell_f['ellipse'][()]

            movie_fns = [fn for fn in os.listdir(eyetracking_folder) if fn[-4:] == '.avi'
                         and 'labeled' not in fn]
            if len(movie_fns) == 0:
                raise LookupError('\t\tNo raw movie found. Abort.')
            elif len(movie_fns) > 1:
                raise LookupError('\t\tMore than one raw movie files found. Abort.')

            movie_path = os.path.join(eyetracking_folder, movie_fns[0])

            mov = cv2.VideoCapture(movie_path)
            mov_shape = (int(mov.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                         int(mov.get(cv2.CAP_PROP_FRAME_WIDTH)))
            mov.release()

            description = '''This is a PupilTracking timeseries. The eyetracking movie is recorded by 
            the pipeline stage and feature points were extracted by DeepLabCut using pipeline eyetracking
            model. For each movie frame, each circumference of corneal reflection, eye lid and pupil were 
            labeled by 12 points using DeepLabCut and the model. Then an ellipse was fit to each of the object. 
            Each ellipse is represent by 5 numbers (see below). The tracking data is saved in the "data" 
            field. Data should be an array with shape (n, 15). n is the eyetracking movie frame number. 
            data[:, 0:5] is the fitted ellipses for the corneal reflection, data[:, 5:10] is the fitted ellipse 
            for the eye, and data[:, 10:15] is the fitted ellipse for the pupil. Each five number of an ellipse 
            represent:
            center_elevation: in mm, small means ventral large means dorsal, this is relative to the movie FOV
            center_azimuth: in mm, small means nasal, large means temporal, this is realtive to the movie FOV
            long_axis_length: in mm
            short_axis_lengt: in mm
            angle_long_axis: in degrees, counterclockwise, 0 is the temporal.
            '''

            pixel_size = diagonal_length / np.sqrt(mov_shape[0] ** 2 + mov_shape[1] ** 2)
            if nasal_dir == 'right':
                pass
            elif nasal_dir == 'left':
                ell_data[:, 1] = mov_shape[1] - ell_data[:, 1]
                ell_data[:, 6] = mov_shape[1] - ell_data[:, 6]
                ell_data[:, 11] = mov_shape[1] - ell_data[:, 11]
                ell_data[:, 4] = (180 - ell_data[:, 4]) % 360
                ell_data[:, 9] = (180 - ell_data[:, 9]) % 360
                ell_data[:, 14] = (180 - ell_data[:, 14]) % 360
            else:
                raise ValueError('\tDo not understand "nasal_dir" ({}). '
                                 'Should be "left" or "right"'.format(nasal_dir))

            ell_data[0:4] = ell_data[0:4] * pixel_size
            ell_data[5:9] = ell_data[5:9] * pixel_size
            ell_data[10:14] = ell_data[10:14] * pixel_size

            nwb_f = nt.RecordedFile(nwb_path)
            nwb_f.add_eyetracking_general(ts_path=eyetracking_ts_name, data=ell_data,
                                          module_name='eye_tracking', side=side, comments='',
                                          description=description, source='')

            if is_generate_labeled_movie:
                ell_col = ell_f['ellipse'].attrs['columns']
                ell_df = pd.DataFrame(data=ell_f['ellipse'][()], columns=ell_col)

                et_ts = nwb_f.file_pointer['acquisition/timeseries'][eyetracking_ts_name]['timestamps'][()]
                fps = 1. / np.mean(np.diff(et_ts))

                print('\tGenerating ellipse overlay movie.')
                dlct.generate_labeled_movie(mov_path_raw=movie_path,
                                            mov_path_lab=os.path.splitext(movie_path)[0] + '_ellipse.avi',
                                            df_ell=ell_df,
                                            fourcc='XVID',
                                            is_verbose=True,
                                            fps=fps)
            nwb_f.close()

    def add_rois_and_traces_to_nwb_caiman(self, nwb_folder, plane_ns, plane_depths):

        print('\nAdding rois and traces.')

        def add_rois_and_traces(plane_folder, nwb_f, plane_n, imaging_depth,
                                mov_path='/processing/motion_correction/MotionCorrection'):

            mov_grp = nwb_f.file_pointer[mov_path + '/' + plane_n + '/corrected']

            data_f = h5py.File(os.path.join(plane_folder, 'rois_and_traces.hdf5'), 'r')
            mask_arr_c = data_f['masks_center'][()]
            mask_arr_s = data_f['masks_surround'][()]
            traces_center_raw = data_f['traces_center_raw'][()]
            # traces_center_demixed = data_f['traces_center_demixed'][()]
            traces_center_subtracted = data_f['traces_center_subtracted'][()]
            # traces_center_dff = data_f['traces_center_dff'][()]
            traces_surround_raw = data_f['traces_surround_raw'][()]
            neuropil_r = data_f['neuropil_r'][()]
            neuropil_err = data_f['neuropil_err'][()]
            data_f.close()

            if traces_center_raw.shape[1] != mov_grp['num_samples'][()]:
                raise ValueError('number of trace time points ({}) does not match frame number of '
                                 'corresponding movie ({}).'.format(traces_center_raw.shape[1],
                                                                    mov_grp['num_samples'][()]))

            # traces_center_raw = traces_center_raw[:, :mov_grp['num_samples'][()]]
            # traces_center_subtracted = traces_center_subtracted[:, :mov_grp['num_samples'][()]]
            # traces_surround_raw = traces_surround_raw[:, :mov_grp['num_samples'][()]]

            rf_img_max = tf.imread(os.path.join(plane_folder, 'corrected_max_projection.tif'))
            rf_img_mean = tf.imread(os.path.join(plane_folder, 'corrected_mean_projection.tif'))

            print('adding segmentation results ...')
            rt_mo = nwb_f.create_module('rois_and_traces_' + plane_n)
            rt_mo.set_value('imaging_depth_micron', imaging_depth)
            is_if = rt_mo.create_interface('ImageSegmentation')
            is_if.create_imaging_plane('imaging_plane', description='')
            is_if.add_reference_image('imaging_plane', 'max_projection', rf_img_max)
            is_if.add_reference_image('imaging_plane', 'mean_projection', rf_img_mean)

            for i in range(mask_arr_c.shape[0]):
                curr_cen = mask_arr_c[i]
                curr_cen_n = 'roi_' + ft.int2str(i, 4)
                curr_cen_roi = ia.WeightedROI(curr_cen)
                curr_cen_pixels_yx = curr_cen_roi.get_pixel_array()
                curr_cen_pixels_xy = np.array([curr_cen_pixels_yx[:, 1], curr_cen_pixels_yx[:, 0]]).transpose()
                is_if.add_roi_mask_pixels(image_plane='imaging_plane', roi_name=curr_cen_n, desc='',
                                          pixel_list=curr_cen_pixels_xy, weights=curr_cen_roi.weights, width=512,
                                          height=512)

                curr_sur = mask_arr_s[i]
                curr_sur_n = 'surround_' + ft.int2str(i, 4)
                curr_sur_roi = ia.ROI(curr_sur)
                curr_sur_pixels_yx = curr_sur_roi.get_pixel_array()
                curr_sur_pixels_xy = np.array([curr_sur_pixels_yx[:, 1], curr_sur_pixels_yx[:, 0]]).transpose()
                is_if.add_roi_mask_pixels(image_plane='imaging_plane', roi_name=curr_sur_n, desc='',
                                          pixel_list=curr_sur_pixels_xy, weights=None, width=512, height=512)
            is_if.finalize()

            trace_f_if = rt_mo.create_interface('Fluorescence')
            seg_if_path = '/processing/rois_and_traces_' + plane_n + '/ImageSegmentation/imaging_plane'
            # print seg_if_path
            ts_path = mov_path + '/' + plane_n + '/corrected'

            print('adding center fluorescence raw')
            trace_raw_ts = nwb_f.create_timeseries('RoiResponseSeries', 'f_center_raw')
            trace_raw_ts.set_data(traces_center_raw, unit='au', conversion=np.nan, resolution=np.nan)
            trace_raw_ts.set_value('data_format', 'roi (row) x time (column)')
            trace_raw_ts.set_value('data_range', '[-8192, 8191]')
            trace_raw_ts.set_description('fluorescence traces extracted from the center region of each roi')
            trace_raw_ts.set_time_as_link(ts_path)
            trace_raw_ts.set_value_as_link('segmentation_interface', seg_if_path)
            roi_names = ['roi_' + ft.int2str(ind, 4) for ind in range(traces_center_raw.shape[0])]
            trace_raw_ts.set_value('roi_names', roi_names)
            trace_raw_ts.set_value('num_samples', traces_center_raw.shape[1])
            trace_f_if.add_timeseries(trace_raw_ts)
            trace_raw_ts.finalize()

            print('adding neuropil fluorescence raw')
            trace_sur_ts = nwb_f.create_timeseries('RoiResponseSeries', 'f_surround_raw')
            trace_sur_ts.set_data(traces_surround_raw, unit='au', conversion=np.nan, resolution=np.nan)
            trace_sur_ts.set_value('data_format', 'roi (row) x time (column)')
            trace_sur_ts.set_value('data_range', '[-8192, 8191]')
            trace_sur_ts.set_description('neuropil traces extracted from the surroud region of each roi')
            trace_sur_ts.set_time_as_link(ts_path)
            trace_sur_ts.set_value_as_link('segmentation_interface', seg_if_path)
            sur_names = ['surround_' + ft.int2str(ind, 4) for ind in range(traces_center_raw.shape[0])]
            trace_sur_ts.set_value('roi_names', sur_names)
            trace_sur_ts.set_value('num_samples', traces_surround_raw.shape[1])
            trace_f_if.add_timeseries(trace_sur_ts)
            trace_sur_ts.finalize()

            roi_center_n_path = '/processing/rois_and_traces_' + plane_n + '/Fluorescence/f_center_raw/roi_names'
            # print 'adding center fluorescence demixed'
            # trace_demix_ts = nwb_f.create_timeseries('RoiResponseSeries', 'f_center_demixed')
            # trace_demix_ts.set_data(traces_center_demixed, unit='au', conversion=np.nan, resolution=np.nan)
            # trace_demix_ts.set_value('data_format', 'roi (row) x time (column)')
            # trace_demix_ts.set_description('center traces after overlapping demixing for each roi')
            # trace_demix_ts.set_time_as_link(mov_path + '/' + plane_n + '/corrected')
            # trace_demix_ts.set_value_as_link('segmentation_interface', seg_if_path)
            # trace_demix_ts.set_value('roi_names', roi_names)
            # trace_demix_ts.set_value('num_samples', traces_center_demixed.shape[1])
            # trace_f_if.add_timeseries(trace_demix_ts)
            # trace_demix_ts.finalize()

            print('adding center fluorescence after neuropil subtraction')
            trace_sub_ts = nwb_f.create_timeseries('RoiResponseSeries', 'f_center_subtracted')
            trace_sub_ts.set_data(traces_center_subtracted, unit='au', conversion=np.nan, resolution=np.nan)
            trace_sub_ts.set_value('data_format', 'roi (row) x time (column)')
            trace_sub_ts.set_description('center traces after overlap demixing and neuropil subtraction for each roi')
            trace_sub_ts.set_time_as_link(mov_path + '/' + plane_n + '/corrected')
            trace_sub_ts.set_value_as_link('segmentation_interface', seg_if_path)
            trace_sub_ts.set_value_as_link('roi_names', roi_center_n_path)
            trace_sub_ts.set_value('num_samples', traces_center_subtracted.shape[1])
            trace_sub_ts.set_value('r', neuropil_r, dtype='float32')
            trace_sub_ts.set_value('rmse', neuropil_err, dtype='float32')
            trace_sub_ts.set_comments('value "r": neuropil contribution ratio for each roi. '
                                      'value "rmse": RMS error of neuropil subtraction for each roi')
            trace_f_if.add_timeseries(trace_sub_ts)
            trace_sub_ts.finalize()

            trace_f_if.finalize()

            # print 'adding global dF/F traces for each roi'
            # trace_dff_if = rt_mo.create_interface('DfOverF')
            #
            # trace_dff_ts = nwb_f.create_timeseries('RoiResponseSeries', 'dff_center')
            # trace_dff_ts.set_data(traces_center_dff, unit='au', conversion=np.nan, resolution=np.nan)
            # trace_dff_ts.set_value('data_format', 'roi (row) x time (column)')
            # trace_dff_ts.set_description('global df/f traces for each roi center, input fluorescence is the trace after demixing'
            #                              ' and neuropil subtraction. global df/f is calculated by '
            #                              'allensdk.brain_observatory.dff.compute_dff() function.')
            # trace_dff_ts.set_time_as_link(ts_path)
            # trace_dff_ts.set_value_as_link('segmentation_interface', seg_if_path)
            # trace_dff_ts.set_value('roi_names', roi_names)
            # trace_dff_ts.set_value('num_samples', traces_center_dff.shape[1])
            # trace_dff_if.add_timeseries(trace_dff_ts)
            # trace_dff_ts.finalize()
            # trace_dff_if.finalize()

            rt_mo.finalize()

        nwb_path = self.get_nwb_path(nwb_folder=nwb_folder)
        nwb_f = nt.RecordedFile(nwb_path)

        for plane_i, plane_n in enumerate(plane_ns):
            print('\n\n' + plane_n)

            plane_folder = os.path.join(nwb_folder, plane_n)
            add_rois_and_traces(plane_folder=plane_folder, nwb_f=nwb_f, plane_n=plane_n,
                                imaging_depth=plane_depths[plane_i])

        nwb_f.close()
        print('\tDone.')

    def add_response_tables(self, nwb_folder, strf_response_window, dgc_response_window,
                            lsn_stim_name, dgc_stim_name):

        print('\nAdding response tables.')

        nwb_path = self.get_nwb_path(nwb_folder=nwb_folder)
        nwb_f = nt.RecordedFile(nwb_path)

        nwb_f.get_drifting_grating_response_table_retinotopic_mapping(stim_name=dgc_stim_name,
                                                                      time_window=dgc_response_window)
        nwb_f.get_spatial_temporal_receptive_field_retinotopic_mapping(stim_name=lsn_stim_name,
                                                                       time_window=strf_response_window)
        nwb_f.close()

        print('\tDone.')


class PlaneProcessor(object):
    """
    pipeline to preprocess two-photon data with in each imaging plane
    includes get roi info from caiman segmentation results,
    filter those rois, generate marked movie, generate rois for neuropil,
    extract calcium traces, perform neuropil subtraction, and
    generate result file ready to be added into nwb files.

    this is very high-level, may break easily
    """

    def __init__(self):
        pass

    @staticmethod
    def get_rois_from_caiman_results(plane_folder, filter_sigma, cut_thr, bg_fn):
        """
        get isolated rois from caiman segmentation results
        :param plane_folder:
        :param filter_sigma: float, pixel, gaussian filter of original caiman results
        :param cut_thr: float, [0., 1.], cutoff threshold to get isolated rois,
                        low for more rois, high for less rois
        :param bg_fn:
        :return:
        """

        print('\nGetting rois from caiman segmentation.')

        data_f = h5py.File(os.path.join(plane_folder,
                                        'caiman_segmentation_results.hdf5'), 'r')
        masks = data_f['masks'][()]
        data_f.close()

        if bg_fn[-4:] != '.tif':
            print('\tCannot find background .tif file.')
            bg = None
        else:
            bg = tf.imread(os.path.join(plane_folder, bg_fn))
            if len(bg.shape) == 3:
                bg = ia.array_nor(np.max(bg, axis=0))
            elif len(bg.shape) == 2:
                bg = ia.array_nor(bg)
            else:
                print('\tBackgound shape not right. Skip.')
                bg = None

        fig_folder = os.path.join(plane_folder, 'figures')
        if not os.path.isdir(fig_folder):
            os.makedirs(fig_folder)

        final_roi_dict = {}

        for i, mask in enumerate(masks):

            msk_std = np.abs(np.std(mask.flatten()))

            if msk_std > 0:
                if filter_sigma is not None:
                    mask_f = ni.filters.gaussian_filter(mask, filter_sigma)
                    mask_f_nor = ia.array_nor(mask_f)
                    mask_bin = np.zeros(mask_f_nor.shape, dtype=np.uint8)
                    mask_bin[mask_f_nor > cut_thr] = 1
                else:
                    mask_bin = np.zeros(mask.shape, dtype=np.uint8)
                    mask_bin[mask > 0] = 1
            else:
                continue

            if np.max(mask_bin) == 1:
                mask_labeled, mask_num = ni.label(mask_bin)
                curr_mask_dict = ia.get_masks(labeled=mask_labeled, keyPrefix='caiman_mask_{:03d}'.format(i),
                                              labelLength=5)
                for roi_key, roi_mask in curr_mask_dict.items():
                    final_roi_dict.update({roi_key: ia.WeightedROI(roi_mask * mask)})

        print('\tTotal number of ROIs:', len(final_roi_dict))

        f = plt.figure(figsize=(15, 8))
        ax1 = f.add_subplot(121)

        if bg is not None:
            ax1.imshow(bg, vmin=0, vmax=0.5, cmap='gray', interpolation='nearest')

        colors1 = pt.random_color(masks.shape[0])
        for i, mask in enumerate(masks):
            pt.plot_mask_borders(mask, plotAxis=ax1, color=colors1[i], borderWidth=1)
        ax1.set_title('original ROIs')
        ax1.set_axis_off()
        ax2 = f.add_subplot(122)

        if bg is not None:
            ax2.imshow(bg, vmin=0, vmax=0.5, cmap='gray', interpolation='nearest')

        colors2 = pt.random_color(len(final_roi_dict))
        i = 0
        for roi in final_roi_dict.values():
            pt.plot_mask_borders(roi.get_binary_mask(), plotAxis=ax2, color=colors2[i], borderWidth=1)
            i = i + 1
        ax2.set_title('filtered ROIs')
        ax2.set_axis_off()
        # plt.show()

        f.savefig(os.path.join(fig_folder, 'caiman_segmentation_filtering.pdf'), dpi=300)

        cell_file = h5py.File(os.path.join(plane_folder, 'rois.hdf5'), 'x')

        i = 0
        for key, value in sorted(final_roi_dict.items()):
            curr_grp = cell_file.create_group('cell{:04d}'.format(i))
            curr_grp.attrs['name'] = key
            value.to_h5_group(curr_grp)
            i += 1

        cell_file.close()

    @staticmethod
    def filter_rois(plane_folder, margin_pix_num, area_range, overlap_thr,
                    bg_fn):
        """
        filter the first pass rois

        :param plane_folder:
        :param margin_pix_num: int, pixel, margin to add from motion correction edge.
        :param area_range: list of two ints, pixel, min and max areas allowed for valid rois
        :param overlap_thr: float, [0., 1.],
                            the smaller rois with overlap ratio larger than this value will be discarded
        :param bg_fn:
        :return:
        """

        print('\nFilter rois.')

        offsets_f = h5py.File(os.path.join(plane_folder, 'correction_offsets.hdf5'), 'r')
        file_keys = list(offsets_f.keys())
        if 'path_list' in file_keys:
            file_keys.remove('path_list')
        offsets = []
        for file_key in file_keys:
            offsets.append(offsets_f[file_key][()])
        offsets = np.concatenate(offsets, axis=0)
        max_offsets0 = np.max(offsets, axis=0)
        max_offsets1 = np.abs(np.min(offsets, axis=0))

        center_margin = [int(round(max_offsets0[0]) + margin_pix_num),
                         int(round(max_offsets1[0]) + margin_pix_num),
                         int(round(max_offsets0[1]) + margin_pix_num),
                         int(round(max_offsets1[1]) + margin_pix_num)]

        print('\tcenter margin due to motion: {}.'.format(center_margin))

        if bg_fn[-4:] != '.tif':
            print('\tCannot find background .tif file.')
            bg = None
        else:
            bg = tf.imread(os.path.join(plane_folder, bg_fn))
            if len(bg.shape) == 3:
                bg = ia.array_nor(np.max(bg, axis=0))
            elif len(bg.shape) == 2:
                bg = ia.array_nor(bg)
            else:
                print('\tBackgound shape not right. Skip.')
                bg = None

        fig_folder = os.path.join(plane_folder, 'figures')
        if not os.path.isdir(fig_folder):
            os.makedirs(fig_folder)

        roi_path = os.path.join(plane_folder, 'rois.hdf5')
        roi_f = h5py.File(roi_path, 'r')
        rois = {}
        for roi_n in roi_f.keys():
            rois.update({roi_n: ia.WeightedROI.from_h5_group(roi_f[roi_n])})

        print('\ttotal number of cells:', len(rois))

        # get the names of rois which are on the edge
        edge_rois = []
        for roiname, roimask in rois.items():
            dimension = roimask.dimension
            center = roimask.get_center()
            if center[0] < center_margin[0] or \
                    center[0] > dimension[0] - center_margin[1] or \
                    center[1] < center_margin[2] or \
                    center[1] > dimension[1] - center_margin[3]:
                edge_rois.append(roiname)

        print('\tnumber of rois to be removed because they are on the edges: {}'.format(len(edge_rois)))
        # remove edge rois
        for edge_roi in edge_rois:
            _ = rois.pop(edge_roi)

        # get dictionary of roi areas
        roi_areas = {}
        for roiname, roimask in rois.items():
            roi_areas.update({roiname: roimask.get_binary_area()})

        # remove roinames that have area outside of the area_range
        invalid_roi_ns = []
        for roiname, roiarea in roi_areas.items():
            if roiarea < area_range[0] or roiarea > area_range[1]:
                invalid_roi_ns.append(roiname)
        print("\tnumber of rois to be removed because they do not meet area criterion: {}".format(len(invalid_roi_ns)))
        for invalid_roi_n in invalid_roi_ns:
            roi_areas.pop(invalid_roi_n)

        # sort rois with their binary area
        roi_areas_sorted = sorted(roi_areas.items(), key=operator.itemgetter(1))
        roi_areas_sorted.reverse()
        roi_names_sorted = [c[0] for c in roi_areas_sorted]
        # print '\n'.join([str(c) for c in roi_areas_sorted])

        # get the name of rois that needs to be removed because of overlapping
        retain_rois = []
        remove_rois = []
        for roi1_name in roi_names_sorted:
            roi1_mask = rois[roi1_name]
            is_remove = 0
            roi1_area = roi1_mask.get_binary_area()
            for roi2_name in retain_rois:
                roi2_mask = rois[roi2_name]
                roi2_area = roi2_mask.get_binary_area()
                curr_overlap = roi1_mask.binary_overlap(roi2_mask)

                if float(curr_overlap) / roi1_area > overlap_thr:
                    remove_rois.append(roi1_name)
                    is_remove = 1
                    # print('\t\t' + roi1_name, ':', roi1_mask.get_binary_area(), ': removed')
                    break

            if is_remove == 0:
                retain_rois.append(roi1_name)
                # print('\t\t' + roi1_name, ':', roi1_mask.get_binary_area(), ': retained')

        print('\tnumber of rois to be removed because of overlapping: {}'.format(len(remove_rois)))
        print('\ttotal number of reatined rois:', len(retain_rois))

        # plotting
        colors = pt.random_color(len(rois.keys()))

        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(111)

        if bg is not None:
            ax.imshow(bg, cmap='gray', vmin=0, vmax=0.5, interpolation='nearest')
        else:
            ax.imshow(np.zeros((512, 512), dtype=np.uint8), vmin=0, vmax=1, cmap='gray', interpolation='nearest')

        f2 = plt.figure(figsize=(10, 10))
        ax2 = f2.add_subplot(111)
        if bg is not None:
            ax2.imshow(np.zeros(bg.shape, dtype=np.uint8), vmin=0, vmax=1, cmap='gray', interpolation='nearest')
        else:
            ax2.imshow(np.zeros((512, 512), dtype=np.uint8), vmin=0, vmax=1, cmap='gray', interpolation='nearest')

        i = 0
        for retain_roi in retain_rois:
            rois[retain_roi].plot_binary_mask_border(plotAxis=ax, color=colors[i], borderWidth=1)
            rois[retain_roi].plot_binary_mask_border(plotAxis=ax2, color=colors[i], borderWidth=1)
            i += 1
        # plt.show()

        # save figures
        pt.save_figure_without_borders(f, os.path.join(fig_folder, '2P_refined_ROIs_with_background.png'), dpi=300)
        pt.save_figure_without_borders(f2, os.path.join(fig_folder, '2P_refined_ROIs_without_background.png'), dpi=300)

        # save h5 file
        save_file = h5py.File(os.path.join(plane_folder, 'rois_refined.hdf5'), 'x')
        i = 0
        for retain_roi in retain_rois:
            # print(retain_roi, ':', rois[retain_roi].get_binary_area())

            currGroup = save_file.create_group('roi' + ft.int2str(i, 4))
            currGroup.attrs['name'] = retain_roi
            roiGroup = currGroup.create_group('roi')
            rois[retain_roi].to_h5_group(roiGroup)
            i += 1

        for attr, value in roi_f.attrs.items():
            save_file.attrs[attr] = value

        save_file.close()
        roi_f.close()

    @staticmethod
    def get_rois_and_surrounds(plane_folder, surround_limit, bg_fn):
        """

        :param plane_folder:
        :param surround_limit: list of two ints, pixel, inner and outer edge of surround donut.
        :param bg_fn:
        :return:
        """

        print('\nGetting rois and surrounds.')

        if bg_fn[-4:] != '.tif':
            print('\tCannot find background .tif file.')
            bg = None
        else:
            bg = tf.imread(os.path.join(plane_folder, bg_fn))
            if len(bg.shape) == 3:
                bg = ia.array_nor(np.max(bg, axis=0))
            elif len(bg.shape) == 2:
                bg = ia.array_nor(bg)
            else:
                print('\tBackgound shape not right. Skip.')
                bg = None

        fig_folder = os.path.join(plane_folder, 'figures')
        if not os.path.isdir(fig_folder):
            os.makedirs(fig_folder)

        print('\treading cells file ...')
        data_f = h5py.File(os.path.join(plane_folder, 'rois_refined.hdf5'), 'r')

        roi_ns = list(data_f.keys())
        roi_ns.sort()

        binary_mask_array = []
        weight_mask_array = []

        for roi_n in roi_ns:
            curr_roi = ia.ROI.from_h5_group(data_f[roi_n]['roi'])
            binary_mask_array.append(curr_roi.get_binary_mask())
            weight_mask_array.append(curr_roi.get_weighted_mask())

        data_f.close()
        binary_mask_array = np.array(binary_mask_array)
        weight_mask_array = np.array(weight_mask_array)
        print('\tstarting mask_array shape:', weight_mask_array.shape)

        print('\tgetting total mask ...')
        total_mask = np.zeros((binary_mask_array.shape[1], binary_mask_array.shape[2]), dtype=np.uint8)
        for curr_mask in binary_mask_array:
            total_mask = np.logical_or(total_mask, curr_mask)
        total_mask = np.logical_not(total_mask)

        plt.imshow(total_mask, interpolation='nearest')
        plt.title('total_mask')
        # plt.show()

        print('\tgetting surround masks ...')
        binary_surround_array = []
        for binary_center in binary_mask_array:
            curr_surround = np.logical_xor(ni.binary_dilation(binary_center, iterations=surround_limit[1]),
                                           ni.binary_dilation(binary_center, iterations=surround_limit[0]))
            curr_surround = np.logical_and(curr_surround, total_mask).astype(np.uint8)
            binary_surround_array.append(curr_surround)
            # plt.imshow(curr_surround)
            # plt.show()
        binary_surround_array = np.array(binary_surround_array)

        print("\tsaving rois ...")
        center_areas = []
        surround_areas = []
        for mask_ind in range(binary_mask_array.shape[0]):
            center_areas.append(np.sum(binary_mask_array[mask_ind].flat))
            surround_areas.append(np.sum(binary_surround_array[mask_ind].flat))
        roi_f = h5py.File(os.path.join(plane_folder, 'rois_and_traces.hdf5'), 'x')
        roi_f.create_dataset('masks_center', data=weight_mask_array, compression='lzf')
        roi_f.create_dataset('masks_surround', data=binary_surround_array, compression='lzf')

        roi_f.close()
        print('\tminimum surround area:', min(surround_areas), 'pixels.')

        f = plt.figure(figsize=(10, 10))
        ax_center = f.add_subplot(211)
        ax_center.hist(center_areas, bins=30)
        ax_center.set_title('roi center area distribution')
        ax_surround = f.add_subplot(212)
        ax_surround.hist(surround_areas, bins=30)
        ax_surround.set_title('roi surround area distribution')
        # plt.show()

        print('\tplotting ...')
        colors = pt.random_color(weight_mask_array.shape[0])

        f_c_bg = plt.figure(figsize=(10, 10))

        if bg is None:
            bg = np.zeros((512, 512), dtype=np.float32)

        ax_c_bg = f_c_bg.add_subplot(111)
        ax_c_bg.imshow(bg, cmap='gray', vmin=0, vmax=0.5, interpolation='nearest')
        f_c_nbg = plt.figure(figsize=(10, 10))
        ax_c_nbg = f_c_nbg.add_subplot(111)
        ax_c_nbg.imshow(np.zeros(bg.shape, dtype=np.uint8), vmin=0, vmax=1, cmap='gray', interpolation='nearest')
        f_s_nbg = plt.figure(figsize=(10, 10))
        ax_s_nbg = f_s_nbg.add_subplot(111)
        ax_s_nbg.imshow(np.zeros(bg.shape, dtype=np.uint8), vmin=0, vmax=1, cmap='gray', interpolation='nearest')

        i = 0
        for mask_ind in range(binary_mask_array.shape[0]):
            pt.plot_mask_borders(binary_mask_array[mask_ind], plotAxis=ax_c_bg, color=colors[i], borderWidth=1)
            pt.plot_mask_borders(binary_mask_array[mask_ind], plotAxis=ax_c_nbg, color=colors[i], borderWidth=1)
            pt.plot_mask_borders(binary_surround_array[mask_ind], plotAxis=ax_s_nbg, color=colors[i], borderWidth=1)
            i += 1

        # plt.show()

        print('\tsaving figures ...')
        pt.save_figure_without_borders(f_c_bg, os.path.join(fig_folder, '2P_ROIs_with_background.png'), dpi=300)
        pt.save_figure_without_borders(f_c_nbg, os.path.join(fig_folder, '2P_ROIs_without_background.png'), dpi=300)
        pt.save_figure_without_borders(f_s_nbg, os.path.join(fig_folder, '2P_ROI_surrounds_background.png'), dpi=300)
        f.savefig(os.path.join(fig_folder, 'roi_area_distribution.pdf'), dpi=300)

    @staticmethod
    def get_nwb_path(plane_folder):

        nwb_folder = os.path.dirname(os.path.realpath(plane_folder))
        nwb_fns = [fn for fn in os.listdir(nwb_folder) if fn[-4:] == '.nwb']
        if len(nwb_fns) == 0:
            print('Cannot find .nwb files in {}!'.format(os.path.realpath(nwb_folder)))
            return None
        elif len(nwb_fns) > 1:
            print('More than one .nwb files found in {}!'.format(os.path.realpath(nwb_folder)))
            return None
        elif len(nwb_fns) == 1:
            return os.path.realpath(os.path.join(nwb_folder, nwb_fns[0]))

    def get_raw_center_and_surround_traces(self, plane_folder, chunk_size, process_num):
        """

        :param plane_folder:
        :param chunk_size: int, number of frames for multiprocessing
        :param process_num: int, number of processes
        :return:
        """

        def get_chunk_frames(frame_num, chunk_size):
            chunk_num = frame_num // chunk_size
            if frame_num % chunk_size > 0:
                chunk_num = chunk_num + 1

            print("\ttotal number of frames:", frame_num)
            print("\ttotal number of chunks:", chunk_num)

            chunk_ind = []
            chunk_starts = []
            chunk_ends = []

            for chunk_i in range(chunk_num):
                chunk_ind.append(chunk_i)
                chunk_starts.append(chunk_i * chunk_size)

                if chunk_i < chunk_num - 1:
                    chunk_ends.append((chunk_i + 1) * chunk_size)
                else:
                    chunk_ends.append(frame_num)

            return zip(chunk_ind, chunk_starts, chunk_ends)

        print('\nExtracting traces.')
        nwb_path = self.get_nwb_path(plane_folder=plane_folder)

        plane_n = os.path.split(os.path.realpath(plane_folder))[1]

        print('\tgetting masks ...')
        rois_f = h5py.File(os.path.join(plane_folder, 'rois_and_traces.hdf5'), 'a')
        center_array = rois_f['masks_center'][()]
        surround_array = rois_f['masks_surround'][()]

        print('\tanalyzing movie in chunks of size: {} frames.'.format(chunk_size))

        data_path = '/processing/motion_correction/MotionCorrection/' + plane_n + '/corrected/data'
        nwb_f = h5py.File(nwb_path, 'r')
        total_frame = nwb_f[data_path].shape[0]
        nwb_f.close()

        chunk_frames = get_chunk_frames(total_frame, chunk_size)
        chunk_params = [(cf[0], cf[1], cf[2], nwb_path, data_path,
                         plane_folder, center_array, surround_array) for cf in chunk_frames]

        p = Pool(process_num)
        p.map(get_traces, chunk_params)

        chunk_folder = os.path.join(plane_folder, 'chunks')
        chunk_fns = [f for f in os.listdir(chunk_folder) if f[0:11] == 'chunk_temp_']
        chunk_fns.sort()
        print('\treading chunks files ...')
        _ = [print('\t\t{}'.format(c)) for c in chunk_fns]

        traces_raw = []
        traces_surround = []

        for chunk_fn in chunk_fns:
            curr_chunk_f = h5py.File(os.path.join(chunk_folder, chunk_fn), 'r')
            traces_raw.append(curr_chunk_f['traces_center'][()])
            traces_surround.append(curr_chunk_f['traces_surround'][()])

        print("\tsaving ...")
        traces_raw = np.concatenate(traces_raw, axis=1)
        traces_surround = np.concatenate(traces_surround, axis=1)
        rois_f.create_dataset('traces_center_raw', data=traces_raw, compression='lzf')
        rois_f.create_dataset('traces_surround_raw', data=traces_surround, compression='lzf')
        print('\tdone.')

    @staticmethod
    def get_neuropil_subtracted_traces(plane_folder, lam, plot_chunk_size, plot_process_num):

        print('\nNeuropil subtraction.')

        data_f = h5py.File(os.path.join(plane_folder, 'rois_and_traces.hdf5'), 'a')
        traces_raw = data_f['traces_center_raw'][()]
        traces_srround = data_f['traces_surround_raw'][()]

        traces_subtracted = np.zeros(traces_raw.shape, np.float32)
        ratio = np.zeros(traces_raw.shape[0], np.float32)
        err = np.zeros(traces_raw.shape[0], np.float32)

        for i in range(traces_raw.shape[0]):
            curr_trace_c = traces_raw[i]
            curr_trace_s = traces_srround[i]
            curr_r, curr_err, curr_trace_sub = hl.neural_pil_subtraction(curr_trace_c, curr_trace_s, lam=lam)
            print("\t\troi_{:05d}\tr = {:.4f}; error = {:.4f}.".format(i, curr_r, curr_err))
            traces_subtracted[i] = curr_trace_sub
            ratio[i] = curr_r
            err[i] = curr_err

        print('\tplotting neuropil subtraction results ...')
        fig_folder = os.path.join(plane_folder, 'figures')
        if not os.path.isdir(fig_folder):
            os.makedirs(fig_folder)
        save_folder = os.path.join(fig_folder, 'neuropil_subtraction_lam_{}'.format(lam))
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        params = []
        for roi_ind in range(traces_raw.shape[0]):
            curr_traces = np.array([traces_raw[roi_ind], traces_srround[roi_ind], traces_subtracted[roi_ind]])

            params.append((curr_traces, plot_chunk_size, roi_ind, save_folder))

        p = Pool(plot_process_num)
        p.map(plot_traces_for_multi_process, params)

        data_f['traces_center_subtracted'] = traces_subtracted
        data_f['neuropil_r'] = ratio
        data_f['neuropil_err'] = err

        data_f.close()
        print('\tDone.')

    def generate_labeled_movie(self, plane_folder, downsample_rate, process_num, chunk_size, frame_size):
        """
        generate downsampled .avi movie overlayed with roi contour

        :param plane_folder:
        :param downsample_rate: int, downsample rate
        :param process_num: int
        :param chunk_size: int
        :param frame_size: float, movie frame size, inch
        :return:
        """

        print('\nGenerating labeled movie ...')

        print('\tgetting total mask ...')
        roi_f = h5py.File(os.path.join(plane_folder, 'rois_refined.hdf5'), 'r')
        h, w = roi_f['roi0000']['roi'].attrs['dimension']
        total_mask = np.zeros((h, w), dtype=np.uint8)
        for roi_n, roi_grp in roi_f.items():
            curr_roi = ia.WeightedROI.from_h5_group(roi_grp['roi'])
            curr_mask = curr_roi.get_binary_mask()
            total_mask = np.logical_or(total_mask, curr_mask)
        roi_f.close()
        total_mask = ni.binary_dilation(total_mask, iterations=1)


        nwb_path = self.get_nwb_path(plane_folder=plane_folder)

        plane_n = os.path.split(os.path.realpath(plane_folder))[1]
        dset_path = 'processing/motion_correction/MotionCorrection/{}/corrected/data'.format(plane_n)

        print('\tdownsampling movie ...')
        print('\t\tnwb_path: {}'.format(nwb_path))
        print('\t\tdset_path: {}'.format(dset_path))

        nwb_f = h5py.File(nwb_path, 'r')
        dset = nwb_f[dset_path]
        print('\t\ttotal shape: {}'.format(dset.shape))
        nwb_f.close()

        mov_d = downsample_mov(nwb_path=nwb_path, dset_path=dset_path, dr=downsample_rate,
                               chunk_size=chunk_size, process_num=process_num)
        v_min = np.amin(mov_d)
        v_max = np.amax(mov_d)
        print('\t\tshape of downsampled movie: {}'.format(mov_d.shape))

        print('\t\tgenerating avi ...')

        if cv2.__version__[0:3] == '3.1' or cv2.__version__[0] == '4':
            codex = 'XVID'
            fourcc = cv2.VideoWriter_fourcc(*codex)
            out = cv2.VideoWriter('marked_mov.avi', fourcc, 30, (frame_size * 100, frame_size * 100), isColor=True)
        elif cv2.__version__[0:6] == '2.4.11':
            out = cv2.VideoWriter('marked_mov.avi', -1, 30, (frame_size * 100, frame_size * 100), isColor=True)
        elif cv2.__version__[0:3] == '2.4':
            codex = 'XVID'
            fourcc = cv2.cv.CV_FOURCC(*codex)
            out = cv2.VideoWriter('marked_mov.avi', fourcc, 30, (frame_size * 100, frame_size * 100), isColor=True)
        else:
            raise EnvironmentError('Do not understand opencv cv2 version: {}.'.format(cv2.__version__))

        f = plt.figure(figsize=(frame_size, frame_size))
        for frame_i, frame in enumerate(mov_d):

            if frame_i % 100 == 0:
                print('\t\tframe: {} / {}'.format(frame_i, mov_d.shape[0]))

            f.clear()
            ax = f.add_subplot(111)
            ax.imshow(frame, vmin=v_min, vmax=v_max * 0.5, cmap='gray', interpolation='nearest')
            pt.plot_mask_borders(total_mask, plotAxis=ax, color='#ff0000', zoom=1, borderWidth=1)
            ax.set_aspect('equal')
            # plt.show()

            buffer_ = io.BytesIO()
            pt.save_figure_without_borders(f, buffer_, dpi=100)
            buffer_.seek(0)
            image = PIL.Image.open(buffer_)
            curr_frame = np.asarray(image)
            r, g, b, a = np.rollaxis(curr_frame, axis=-1)
            curr_frame = (np.dstack((b, g, r)))

            out.write(curr_frame)

        out.release()
        cv2.destroyAllWindows()
        print('\t\tDone.')





