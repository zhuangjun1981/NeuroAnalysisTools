import os
import shutil
import h5py
import numpy as np
import tifffile as tf
import NeuroAnalysisTools.core.FileTools as ft
import NeuroAnalysisTools.core.ImageAnalysis as ia
import NeuroAnalysisTools.core.PlottingTools as pt
import NeuroAnalysisTools.NwbTools as nt
import matplotlib.pyplot as plt
import cv2
import NeuroAnalysisTools.MotionCorrection as mc


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

    def add_2p_image_to_nwb(self, nwb_folder, image_identifier, zoom, scope, plane_ns,
                            image_depths, temporal_downsample_rate):
        """

        :param nwb_folder: str, folder containing .nwb file and the packaged 2p imaging .hdf5 file
        :param image_identifier: str, for finding the the packaged 2p imaging .hdf5 file
        :param zoom: float, zoom of 2p recording
        :param scope: str, 'sutter' or 'deepscope'
        :param plane_ns: list of str, ['plane0', 'plane1', 'plane2', ...]
        :param image_depths: list of numbers, depth below pia for each imaging plane in microns
                             the length of this list should match the number of imaging planes
        :param temporal_downsample_rate: int, total temporal downsample rate,
                                         td_rate_before_correction * tdrate_after_correction
        :return:
        """

        print('\nAdding 2p imaging data to .nwb file ...')

        if len(plane_ns) != len(image_depths):
            raise ValueError('Length of "plane_ns" ({}) should match length of '
                             '"image_depths" ({}).'.format(len(plane_ns), len(image_depths)))

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

                    curr_description = '{}. Imaging depth: {} micron.'.format(description, image_depths[mov_i])
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



