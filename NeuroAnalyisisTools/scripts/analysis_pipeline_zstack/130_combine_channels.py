import os
import numpy as np
import tifffile as tf
import cv2

fn_lst = ['FOV2_projection_site_zstack_green_aligned.tif',
          'FOV2_projection_site_zstack_red_aligned.tif']

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

mov = []

for fn in fn_lst:
    curr_mov = tf.imread(fn).astype(np.float32)

    curr_mov_adjust = []
    # histogram equalization
    for frame in curr_mov:
        # display_frame = (frame - np.amin(frame)) / (np.amax(frame) - np.amin(frame))
        # display_frame = (display_frame * 255).astype(np.uint8)
        # display_frame = cv2.equalizeHist(display_frame)

        display_frame = frame - np.mean(frame[:])
        curr_mov_adjust.append(display_frame)

    curr_mov_adjust = np.array(curr_mov_adjust)
    mov.append(curr_mov_adjust)

mov = np.concatenate(mov, axis=2)

tf.imsave('zstack_aligned_combined.tif', mov)