import os
import numpy as np
import tifffile as tf
import corticalmapping.core.ImageAnalysis as ia
import matplotlib.pyplot as plt
import cv2


data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\181213-M421211-2p"
scope = 'sutter' # 'sutter', 'deepscope' or 'scientifica'
identifier = "vasmap_2p"
channels = ['green', 'red']
is_equalize = False # equalize histogram

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

vasmaps = {}
for chn in channels:
    vasmaps.update({chn: []})

file_ns = [f for f in os.listdir(data_folder) if identifier in f]
file_ns.sort()
print('\n'.join(file_ns))

for file_n in file_ns:
    print(file_n)

    curr_vasmap = tf.imread(os.path.join(data_folder, file_n))

    if len(curr_vasmap.shape) == 2:
        if len(channels) == 1:
            vasmaps[channels[0]].append(np.array([curr_vasmap]))
        else:
            raise ValueError('recorded file is 2d, cannot be deinterleved into {} channels.'.format(len(channels)))
    else:
        if len(curr_vasmap.shape) != 3:
            raise ValueError('shape of recorded file: {}. should be either 2d or 3d.'.format(curr_vasmap.shape))

        for ch_i, ch_n in enumerate(channels):
            curr_vasmap_ch = curr_vasmap[ch_i::len(channels)]
            curr_vasmap_ch = ia.array_nor(np.mean(curr_vasmap_ch, axis=0))
            if is_equalize:
                curr_vasmap_ch = (curr_vasmap_ch * 255).astype(np.uint8)
                curr_vasmap_ch = cv2.equalizeHist(curr_vasmap_ch).astype(np.float32)
            vasmaps[ch_n].append(curr_vasmap_ch)

for ch_n, ch_vasmap in vasmaps.items():
    # save_vasmap = np.concatenate(ch_vasmap, axis=0)
    # print(save_vasmap.shape)
    # save_vasmap = ia.array_nor(np.mean(save_vasmap, axis=0))
    # print(save_vasmap.shape)

    save_vasmap = ia.array_nor(np.mean(ch_vasmap, axis=0))

    if scope == 'scientifica':
        save_vasmap_r = save_vasmap[::-1, :]
        save_vasmap_r = ia.rigid_transform_cv2_2d(save_vasmap_r, rotation=135)
    elif scope == 'sutter':
        save_vasmap_r = save_vasmap.transpose()[::-1, :]
    elif scope == 'deepscope':
        save_vasmap_r = ia.rigid_transform_cv2(save_vasmap, rotation=140)[:, ::-1]
    else:
        raise LookupError("Do not understand scope type. Should be 'sutter' or 'deepscope' or 'scientifica'.")

    tf.imsave('{}_{}.tif'.format(identifier, ch_n), save_vasmap.astype(np.float32))
    tf.imsave('{}_{}_rotated.tif'.format(identifier, ch_n), save_vasmap_r.astype(np.float32))
