import os
import numpy as np
import tifffile as tf
import NeuroAnalysisTools.core.ImageAnalysis as ia

date_recorded = '190503'
mouse_id = 'M439939'
xy_downsample_rate = 2
t_downsample_rate = 10
ch_ns = ['green', 'red']
data_folder_n = '110_LSNDGC_reorged'
imaging_mode = '2p' # '2p' or 'deepscope'
identifier = '110_LSNDGC'

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\{}-{}-{}" \
              r"\{}".format(date_recorded, mouse_id, imaging_mode, data_folder_n)

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

plane_ns = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f)) and f[:5] == 'plane']
plane_ns.sort()
print('planes:')
print('\n'.join(plane_ns))

for plane_n in plane_ns:
    print('\nprocessing plane: {}'.format(plane_n))

    save_folder = os.path.join(curr_folder, plane_n)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    for ch_n in ch_ns:
        print('\n\tprocessing channel: {}'.format(ch_n))
        plane_folder = os.path.join(data_folder, plane_n, ch_n, 'corrected')

        # f_ns = [f for f in os.listdir(plane_folder) if f[-14:] == '_corrected.tif']
        f_ns= [f for f in os.listdir(plane_folder) if f[-4:] == '.tif' and identifier in f]
        f_ns.sort()
        print('\t\t'+'\n\t\t'.join(f_ns) + '\n')

        mov_d = []

        for f_n in f_ns:
            print('\t\tprocessing {} ...'.format(f_n))
            curr_mov = tf.imread(os.path.join(plane_folder, f_n))
            curr_mov_d = ia.rigid_transform_cv2(img=curr_mov, zoom=(1. / xy_downsample_rate))
            curr_mov_d = ia.z_downsample(curr_mov_d, downSampleRate=t_downsample_rate, is_verbose=False)
            mov_d.append(curr_mov_d)

        mov_d = np.concatenate(mov_d, axis=0)
        save_n = '{}_{}_{}_downsampled.tif'.format(os.path.split(data_folder)[1], plane_n, ch_n)
        tf.imsave(os.path.join(save_folder, save_n), mov_d)