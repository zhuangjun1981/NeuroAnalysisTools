import os
import numpy as np
import tifffile as tf
import skimage.io as io
import skimage.color as color
import matplotlib.pyplot as plt
import corticalmapping.core.ImageAnalysis as ia

vasmap_wf_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\181102-M412052-deepscope\vasmap_wf"

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

map_fns = [f for f in os.listdir(vasmap_wf_folder) if f[-4:]=='.tif']
map_fns.sort()
print('\n'.join(map_fns))

map_wf = []
for map_fn in map_fns:
    curr_map = tf.imread(os.path.join(vasmap_wf_folder, map_fn)).astype(np.float32)
    map_wf.append(ia.array_nor(curr_map))

map_wf = ia.array_nor(np.mean(map_wf, axis=0))
map_wf_r = ia.array_nor(ia.rigid_transform_cv2(map_wf, rotation=140)[:, ::-1])

f = plt.figure(figsize=(12, 6))
ax_wf = f.add_subplot(121)
ax_wf.imshow(map_wf, vmin=0., vmax=1., cmap='gray', interpolation='nearest')
ax_wf.set_axis_off()
ax_wf.set_title('vasmap widefield')

ax_wf_r = f.add_subplot(122)
ax_wf_r.imshow(map_wf_r, vmin=0., vmax=1., cmap='gray', interpolation='nearest')
ax_wf_r.set_axis_off()
ax_wf_r.set_title('vasmap widefield rotated')

plt.show()

tf.imsave('vasmap_wf.tif', map_wf.astype(np.float32))
tf.imsave('vasmap_wf_rotated.tif', map_wf_r.astype(np.float32))