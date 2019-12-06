import os
import numpy as np
import tifffile as tf
import stia.motion_correction as mc
import stia.utility.image_analysis as ia

fn = 'stack1_final.tif'
scope = 'deepscope' # 'sutter' or 'deepscope' or 'scientifica'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

stack = tf.imread(fn)

if scope == 'sutter':
	stack_r = stack.transpose((0, 2, 1))[:, ::-1, :]

elif scope == 'deepscope':
	h_new = int(stack.shape[1] * np.sqrt(2))
	w_new = int(stack.shape[2] * np.sqrt(2))
	stack_r = ia.rigid_transform_cv2(stack, rotation=140, output_shape=(h_new, w_new))[:, :, ::-1]

elif scope == 'scientifica':
	h_new = int(stack.shape[1] * np.sqrt(2))
	w_new = int(stack.shape[2] * np.sqrt(2))
	stack_r = ia.rigid_transform_cv2(stack[:,::-1,:], rotation=135, output_shape=(h_new, w_new))

else:
	raise LookupError("Do not understand scope type. Should be 'sutter' or 'deepscope' or 'scientifica'.")

tf.imsave(os.path.splitext(fn)[0] + '_rotated.tif', stack_r.astype(stack.dtype))