import os
import numpy as np
import NeuroAnalysisTools.core.ImageAnalysis as ia
import tifffile as tf
import shutil

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\190104-M417949-2p\110_LSNDGC_reorged"
td_rate = 5
file_identifier = '110_LSNDGC'
frames_per_file = 500

def downsample_folder(working_folder,
                      td_rate,
                      file_identifier,
                      frames_per_file=500):

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
                save_chunk = total_mov[save_file_id * (frames_per_file * td_rate) :
                                       (save_file_id + 1) * (frames_per_file * td_rate)]
                save_path = os.path.join(working_folder, '{}_{:05d}_corrected_downsampled.tif'.format(file_identifier,
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
        save_path = os.path.join(working_folder, '{}_{:05d}_corrected_downsampled.tif'.format(file_identifier, save_id))
        save_chunk = ia.z_downsample(total_mov, downSampleRate=td_rate, is_verbose=False)
        print('\t\t\tsaving {} ...'.format(os.path.split(save_path)[1]))
        tf.imsave(save_path, save_chunk)

    return

if td_rate == 1:
    raise ValueError('Downsample rate shold not be 1!')

plane_ns = [f for f in os.listdir(data_folder) if f[0:5]=='plane']
plane_ns.sort()
print('all planes:')
print('\n'.join(plane_ns))
print

for plane_n in plane_ns:
    print('current plane: {}'.format(plane_n))
    plane_folder = os.path.join(data_folder, plane_n)
    ch_ns = [f for f in os.listdir(plane_folder)]
    ch_ns.sort()
    print('\tall channels: {}'.format(ch_ns))

    for ch_n in ch_ns:
        print
        print('\tcurrent channel: {}'.format(ch_n))
        ch_folder = os.path.join(plane_folder, ch_n)

        downsample_folder(working_folder=os.path.join(ch_folder, 'corrected'),
                          td_rate=td_rate,
                          file_identifier=file_identifier,
                          frames_per_file=frames_per_file)

print('\nDone!')