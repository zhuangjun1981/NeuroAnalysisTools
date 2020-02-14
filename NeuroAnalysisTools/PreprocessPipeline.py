import os
import numpy as np
import tifffile as tf
import NeuroAnalysisTools.core.FileTools as ft
import NeuroAnalysisTools.core.ImageAnalysis as ia
import NeuroAnalysisTools.core.PlottingTools as pt
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
    def get_vasmap_2p(data_folder,
                      save_folder,
                      scope, # 'sutter', 'deepscope' or 'scientifica'
                      channels = ('green', 'red'),
                      identifier='vasmap_2p',
                      is_equalize=False,):

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
    def get_vasmap_wf(data_folder,
                      save_folder,
                      scope,
                      identifier, # 'JCam' for 'sutter' scope, 'vasmap_wf' for deepscope
                      ):

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
    def save_png_vasmap(tif_path,
                        pre_fix,
                        saturation_level=10, # percentile of pixel to be saturated
                        ):

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
    def check_deepscope_file_creation_time(data_folder,
                                           identifier):

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
    def check_deepscope_filename(data_folder,
                                 identifier):
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
    def reorganize_raw_2p_data(data_folder,
                               save_folder,
                               identifier,
                               scope,
                               plane_num,
                               channels,
                               temporal_downsample_rate,
                               frames_per_file,
                               low_thr
                               ):
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

            print('\n\tDone!')
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

            print('\n\tDone!')
        else:
            raise LookupError('Do not understand scope: "{}". Should be "sutter" or "deepscope".'.format(scope))

    @staticmethod
    def motion_correction(data_folder,
                          reference_channel_name,
                          apply_channel_names):
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
    def reapply_motion_correction_multiplane(data_folder,
                                             reference_plane_name,
                                             reference_channel_name,
                                             apply_plane_names,
                                             apply_channel_names):
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

