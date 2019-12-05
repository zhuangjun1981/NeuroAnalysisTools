import os
import numpy as np
import corticalmapping.core.FileTools as ft
import corticalmapping.core.ImageAnalysis as ia
import tifffile as tf

save_name = 'vasmap_wf'
data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\190228-M426525-2p\vasmap_wf"

saveFolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(saveFolder)

vasmap_fns = [f for f in os.listdir(data_folder) if 'JCam' in f]
vasmap_fns.sort()
print('\n'.join(vasmap_fns))

vasmaps = []

for vasmap_fn in vasmap_fns:

    vasmap_focused, _, _ = ft.importRawJCamF(os.path.join(data_folder, vasmap_fn), column=1024, row=1024,
                                              headerLength=116, tailerLength=452) # try 452 if 218 does not work
    vasmap_focused = vasmap_focused[2:]
    vasmap_focused[vasmap_focused > 50000] = 400
    vasmap_focused = np.mean(vasmap_focused, axis=0)
    vasmaps.append(ia.array_nor(vasmap_focused))

vasmap = ia.array_nor(np.mean(vasmaps, axis=0))
vasmap_r = vasmap[::-1, :]

tf.imsave('{}.tif'.format(save_name), vasmap.astype(np.float32))
tf.imsave('{}_rotated.tif'.format(save_name), vasmap_r.astype(np.float32))