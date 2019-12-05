import os
import h5py
import numpy as np
import tifffile as tf
import stia.motion_correction as mc
import stia.utility.image_analysis as ia

identifier = 'zstack1'
ch_ref = 'red'
ch_app = ['green', 'red']

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

stack_ref = tf.imread('{}_{}.tif'.format(identifier, ch_ref))

step_offsets = [[0., 0.]]  # offsets between adjacent steps

print('calculating step offsets ...')
for step_i in range(1, stack_ref.shape[0]):
    curr_offset = mc.phase_correlation(stack_ref[step_i], stack_ref[step_i - 1])
    step_offsets.append(curr_offset)
step_offsets = np.array([np.array(so) for so in step_offsets], dtype=np.float32)
print('\nsetp offsets:')
print(step_offsets)

print('\ncalculating final offsets ...')
final_offsets_y = np.cumsum(step_offsets[:, 0])
final_offsets_x = np.cumsum(step_offsets[:, 1])
final_offsets = np.array([final_offsets_x, final_offsets_y], dtype=np.float32).transpose()

middle_frame_ind = stack_ref.shape[0] // 2
middle_offsets = final_offsets[middle_frame_ind: middle_frame_ind + 1]
final_offsets = final_offsets - middle_offsets
print('\nfinal offsets:')
print(final_offsets)

print('applying final offsets ...')

for ch in ch_app:

    stack_app = tf.imread('{}_{}.tif'.format(identifier, ch))
    stack_aligned = []

    for step_i in range(stack_app.shape[0]):
        curr_offset = final_offsets[step_i]
        frame = stack_app[step_i]
        frame_aligned = ia.rigid_transform_cv2_2d(frame, offset=curr_offset, fill_value=0).astype(np.float32)
        stack_aligned.append(frame_aligned)

    stack_aligned = np.array(stack_aligned, dtype=np.float32)

    tf.imsave('{}_{}_aligned.tif'.format(identifier, ch), stack_aligned)
    # tf.imsave('{}_{}_max_projection.tif'.format(identifier, ch), np.max(stack_aligned, axis=0))



