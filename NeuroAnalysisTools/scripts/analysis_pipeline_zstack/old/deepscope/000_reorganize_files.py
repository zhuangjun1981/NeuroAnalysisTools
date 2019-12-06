import os
import numpy as np
import tifffile as tf
import corticalmapping.core.FileTools as ft
import corticalmapping.core.ImageAnalysis as ia


data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\181102-M412052-deepscope\zstack2"
file_identifier = 'zstack2'
ch_ns = ['red']
frames_per_step = 200
is_rotate = True

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

file_ns = [f for f in os.listdir(data_folder) if f[-4:] == '.tif' and file_identifier in f]
file_ns.sort()
print('\n'.join(file_ns))

save_folders = []
for ch_n in ch_ns:
    curr_save_folder = os.path.join(data_folder, file_identifier, ch_n)
    if not os.path.isdir(curr_save_folder):
        os.makedirs(curr_save_folder)
    save_folders.append(curr_save_folder)

curr_step = 0

for file_n in file_ns:
    curr_mov = tf.imread(os.path.join(data_folder, file_n))

    # reorient movie
    if is_rotate:
        h_new = int(curr_mov.shape[1] * np.sqrt(2))
        w_new = int(curr_mov.shape[2] * np.sqrt(2))
        curr_mov = ia.rigid_transform_cv2(curr_mov, rotation=140, outputShape=(h_new, w_new))[:, :, ::-1]

    curr_frame_num = curr_mov.shape[0] / len(ch_ns)

    if curr_frame_num % frames_per_step != 0:
        raise ValueError('{}: total frame number is not divisible by frames per step.'.format(file_n))

    curr_mov_chs = []
    for ch_i in range(len(ch_ns)):
        curr_mov_chs.append(curr_mov[ch_i::len(ch_ns)])

    steps = curr_frame_num // frames_per_step
    for step_ind in range(steps):

        print ('current step: {}'.format(curr_step))

        for ch_i in range(len(ch_ns)):
            curr_step_mov_ch = curr_mov_chs[ch_i][step_ind * frames_per_step:(step_ind + 1) * frames_per_step,:,:]
            curr_step_n = 'step_' + ft.int2str(curr_step, 4)
            curr_step_folder = os.path.join(save_folders[ch_i], curr_step_n)
            os.mkdir(curr_step_folder)
            tf.imsave(os.path.join(curr_step_folder, curr_step_n + '.tif'), curr_step_mov_ch)

        curr_step += 1
