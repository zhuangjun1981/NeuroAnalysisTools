import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import corticalmapping.core.ImageAnalysis as ia

data_fn = 'zstack_2p_zoom2_red_aligned.tif'
save_fn = '2018-08-16-M376019-depth-profile-red.png'
start_depth = 50 # micron
step_depth = 2 # micron
pix_size = 0.7 # sutter scope, zoom2, 512 x 512
resolution =  512

curr_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_folder)

data = tf.imread(data_fn)
dp = ia.array_nor(np.mean(data, axis=1))

depth_i = np.array(range(0, dp.shape[0], 50))
depth_l = depth_i * step_depth + start_depth

f = plt.figure(figsize=(8, 8))
ax = f.add_subplot(111)
ax.imshow(dp, vmin=0, vmax=1, cmap='magma', aspect=step_depth / pix_size)
ax.set_xticks([0, resolution-1])
ax.set_xticklabels(['0', '{:7.2f}'.format(resolution*pix_size)])
ax.set_yticks(depth_i)
ax.set_yticklabels(depth_l)
ax.set_xlabel('horizontal dis (um)')
ax.set_ylabel('depth (um)')

plt.show()

f.savefig(save_fn)
