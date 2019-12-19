import os
import numpy as np
import tifffile as tf
import NeuroAnalysisTools.core.ImageAnalysis as ia
import h5py

date_recorded = '190503'
mouse_id = 'M439939'
sess_id = '110'
t_downsample_rate = 5
channel = 'green'
data_folder_n = '110_LSNDGC_reorged'
imaging_mode = '2p' # '2p' or 'deepscope'
identifier = '110_LSNDGC'

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\{}-{}-{}" \
              r"\{}".format(date_recorded, mouse_id, imaging_mode, data_folder_n)
base_name = '{}_{}_{}'.format(date_recorded, mouse_id, sess_id)

plane_ns = [p for p in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, p))]
plane_ns.sort()
print('planes:')
print('\n'.join(plane_ns))

for plane_n in plane_ns:
    print('\nprocessing {} ...'.format(plane_n))

    plane_folder = os.path.join(data_folder, plane_n, channel, 'corrected')
    os.chdir(plane_folder)

    # f_ns = [f for f in os.listdir(plane_folder) if f[-14:] == '_corrected.tif']
    f_ns = [f for f in os.listdir(plane_folder) if f[-4:] == '.tif' and identifier in f]
    f_ns.sort()
    print('\n'.join(f_ns))

    mov_join = []
    for f_n in f_ns:
        print('processing plane: {}; file: {} ...'.format(plane_n, f_n))

        curr_mov = tf.imread(os.path.join(plane_folder, f_n))

        if curr_mov.shape[0] % t_downsample_rate != 0:
            print('the frame number of {} ({}) is not divisible by t_downsample_rate ({}).'
                             .format(f_n, curr_mov.shape[0], t_downsample_rate))

        curr_mov_d = ia.z_downsample(curr_mov, downSampleRate=t_downsample_rate, is_verbose=False)
        mov_join.append(curr_mov_d)

    mov_join = np.concatenate(mov_join, axis=0)
    add_to_mov = 10 - np.amin(mov_join)

    save_name = '{}_d1_{}_d2_{}_d3_1_order_C_frames_{}_.mmap'\
        .format(base_name, mov_join.shape[2], mov_join.shape[1], mov_join.shape[0])

    mov_join = mov_join.reshape((mov_join.shape[0], mov_join.shape[1] * mov_join.shape[2]), order='F').transpose()
    mov_join_mmap = np.memmap(os.path.join(plane_folder, save_name), shape=mov_join.shape, order='C',
                              dtype=np.float32, mode='w+')
    mov_join_mmap[:] = mov_join + add_to_mov
    mov_join_mmap.flush()
    del mov_join_mmap

    save_file = h5py.File(os.path.join(plane_folder, 'caiman_segmentation_results.hdf5'))
    save_file['bias_added_to_movie'] = add_to_mov
    save_file.close()

print('done!')