import os
import h5py
import numpy as np
import tifffile as tf
import stia.motion_correction as mc
import stia.utility.image_analysis as ia

zstack_fn = 'FOV2_projection_site_zstack_red.tif'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

zstack = tf.imread(zstack_fn)

step_offsets = [[0., 0.]]  # offsets between adjacent steps

print('calculating step offsets ...')
for step_i in range(1, zstack.shape[0]):
    curr_offset = mc.phase_correlation(zstack[step_i], zstack[step_i - 1])
    step_offsets.append(curr_offset)
step_offsets = np.array([np.array(so) for so in step_offsets], dtype=np.float32)
print('\nsetp offsets:')
print(step_offsets)

print('\ncalculating final offsets ...')
final_offsets_y = np.cumsum(step_offsets[:, 0])
final_offsets_x = np.cumsum(step_offsets[:, 1])
final_offsets = np.array([final_offsets_x, final_offsets_y], dtype=np.float32).transpose()
print('\nfinal offsets:')
print(final_offsets)

print('applying final offsets ...')

zstack_f = []  # fine zstack

for step_i in range(zstack.shape[0]):

    curr_offset = final_offsets[step_i]

    frame = zstack[step_i]
    frame_f = ia.rigid_transform_cv2_2d(frame, offset=curr_offset, fill_value=0.).astype(np.float32)
    zstack_f.append(frame_f)

zstack_f = np.array(zstack_f, dtype=np.float32)

tf.imsave(os.path.splitext(zstack_fn)[0] + '_aligned.tif', zstack_f)
tf.imsave(os.path.splitext(zstack_fn)[0] + '_max_projection.tif', np.max(zstack_f, axis=0))


