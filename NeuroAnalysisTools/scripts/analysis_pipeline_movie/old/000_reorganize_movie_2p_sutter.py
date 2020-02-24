import os
import numpy as np
import NeuroAnalysisTools.core.ImageAnalysis as ia
import tifffile as tf

data_folder = r'\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\190503-M439939-2p\movie'
file_identifier = '110_LSNDGC'
ch_ns = ['green', 'red']
frames_per_file = 500
td_rate = 1 # temporal downsample rate

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

file_list = [f for f in os.listdir(data_folder) if file_identifier in f and f[-4:] == '.tif']
file_list.sort()
print('\n'.join(file_list))

file_paths = [os.path.join(data_folder, f) for f in file_list]

save_folders = []
save_ids = [0 for ch in ch_ns]
total_movs = [None for ch in ch_ns]
for ch_n in ch_ns:
    curr_save_folder = os.path.join(data_folder, file_identifier + '_reorged', 'plane0', ch_n)
    if not os.path.isdir(curr_save_folder):
        os.makedirs(curr_save_folder)
    save_folders.append(curr_save_folder)

for file_path in file_paths:
    print('\nprocessing {} ...'.format(os.path.split(file_path)[1]))

    curr_mov = tf.imread(file_path)

    if curr_mov.shape[0] % len(ch_ns) != 0:
        raise ValueError('\ttotal frame number of current movie ({}) cannot be divided by number of '
                         'channels ({})!'.format(curr_mov.shape[0], len(ch_ns)))

    # curr_mov = curr_mov.transpose((0, 2, 1))[:, ::-1, :]

    for ch_i, ch_n in enumerate(ch_ns):
        print('\n\tprocessing channel: {}'.format(ch_n))

        curr_mov_ch = curr_mov[ch_i::len(ch_ns)]

        if total_movs[ch_i] is None:
            total_movs[ch_i] = curr_mov_ch
        else:
            total_movs[ch_i] = np.concatenate((total_movs[ch_i], curr_mov_ch), axis=0)

        while (total_movs[ch_i] is not None) and \
                (total_movs[ch_i].shape[0] >= frames_per_file * td_rate):

            num_file_to_save = total_movs[ch_i].shape[0] // (frames_per_file * td_rate)

            for save_file_id in range(num_file_to_save):
                save_chunk = total_movs[ch_i][save_file_id * (frames_per_file * td_rate) :
                                              (save_file_id + 1) * (frames_per_file * td_rate)]
                save_path = os.path.join(save_folders[ch_i], '{}_{:05d}_reorged.tif'.format(file_identifier,
                                                                                            save_ids[ch_i]))
                if td_rate != 1:
                    print('\tdown sampling for {} ...'.format(os.path.split(save_path)[1]))
                    save_chunk = ia.z_downsample(save_chunk, downSampleRate=td_rate, is_verbose=False)

                print('\tsaving {} ...'.format(os.path.split(save_path)[1]))
                tf.imsave(save_path, save_chunk)
                save_ids[ch_i] = save_ids[ch_i] + 1

            if total_movs[ch_i].shape[0] % (frames_per_file * td_rate) == 0:
                total_movs[ch_i] = None
            else:
                frame_num_left = total_movs[ch_i].shape[0] % (frames_per_file * td_rate)
                total_movs[ch_i] = total_movs[ch_i][-frame_num_left:]

print('\nprocessing residual frames ...')

for ch_i, ch_n in enumerate(ch_ns):

    if total_movs[ch_i] is not None:
        print('\n\tprocessing channel: {}'.format(ch_n))

        save_path = os.path.join(save_folders[ch_i], '{}_{:05d}_reorged.tif'.format(file_identifier, save_ids[ch_i]))

        curr_mov_ch = total_movs[ch_i]

        if td_rate != 1:
            if curr_mov_ch.shape[0] % td_rate !=0:
                warning_msg = '\tthe residual frame number ({}) cannot be divided by temporal down sample rate ({}).' \
                              ' Drop last few frames.'.format(curr_mov_ch.shape[0], td_rate)
                print(warning_msg)
            print('\tdown sampling for {} ...'.format(os.path.split(save_path)[1]))
            curr_mov_ch = ia.z_downsample(curr_mov_ch, downSampleRate=td_rate, is_verbose=False)

        print('\tsaving {} ...'.format(os.path.split(save_path)[1]))
        tf.imsave(save_path, curr_mov_ch)

print('\nDone!')