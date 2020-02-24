import os
import numpy as np
import tifffile as tf

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\190921-M488586-deepscope\movie"
identifier = '110_LSNDGCUC'
channels = ['green']
plane_num = 3
temporal_downsample_rate = 2
frame_each_file = 500
low_thr = -500

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

fns = np.array([f for f in os.listdir(data_folder) if f[-4:] == '.tif' and identifier in f])
f_nums = [int(os.path.splitext(fn)[0].split('_')[-2]) for fn in fns]
fns = fns[np.argsort(f_nums)]
print('total file number: {}'.format(len(fns)))

# print('\n'.join(fns))

save_folders = []
for i in range(plane_num):
    curr_save_folder = os.path.join(os.path.split(data_folder)[0], identifier + '_reorged', 'plane{}'.format(i))
    if not os.path.isdir(curr_save_folder):
        os.makedirs(curr_save_folder)
    save_folders.append(curr_save_folder)

# frame_per_plane = len(fns) // plane_num
for plane_ind in range(plane_num):
    print('\nprocessing plane: {}'.format(plane_ind))
    curr_fns = fns[plane_ind::plane_num]

    total_frames_down = len(curr_fns) // temporal_downsample_rate
    curr_fns = curr_fns[: total_frames_down * temporal_downsample_rate].reshape((total_frames_down, temporal_downsample_rate))

    print(curr_fns.shape)

    print('current file ind: 000')
    curr_file_ind = 0
    curr_frame_ind = 0
    curr_mov = {}
    for ch_n in channels:
        curr_mov.update({ch_n : []})

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

        if curr_frame_ind < frame_each_file:

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
                print('current file ind: {:05d}; channel: {}'.format(curr_file_ind, ch_n))
            curr_file_ind += 1
            curr_frame_ind = 1

    for ch_n in channels:
        curr_mov_ch = np.array(curr_mov[ch_n], dtype=np.int16)
        save_name = '{}_{:05d}_reorged.tif'.format(identifier, curr_file_ind)
        save_folder_ch = os.path.join(save_folders[plane_ind], ch_n)
        if not os.path.isdir(save_folder_ch):
            os.makedirs(save_folder_ch)
        tf.imsave(os.path.join(save_folder_ch, save_name), curr_mov_ch)
        print('current file ind: {:05d}; channel: {}'.format(curr_file_ind, ch_n))

