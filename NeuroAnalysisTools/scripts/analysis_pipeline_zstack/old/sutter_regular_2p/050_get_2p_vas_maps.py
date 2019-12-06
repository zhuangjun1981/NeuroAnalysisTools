import os
import numpy as np
import tifffile as tf
import corticalmapping.core.ImageAnalysis as ia
import matplotlib.pyplot as plt


data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project\180816-M376019-zstack"
file_ns = ["vasmap_2p_zoom1_00001_00001.tif",
           "vasmap_2p_zoom1_00002_00001.tif",
           "vasmap_2p_zoom1_00003_00001.tif"]

save_name = 'vasmap_2p_zoom1'

channels = ['green', 'red']

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

vasmaps = {}
for chn in channels:
    vasmaps.update({chn: []})

for file_n in file_ns:
    print(file_n)

    curr_vasmap = tf.imread(os.path.join(data_folder, file_n))

    for ch_i, ch_n in enumerate(channels):
        curr_vasmap_ch = curr_vasmap[ch_i::len(channels)]
        vasmaps[ch_n].append(curr_vasmap_ch.transpose((0, 2, 1))[:, ::-1, :])
        # print(curr_vasmap_ch.shape)

for ch_n, ch_vasmap in vasmaps.items():
    save_vasmap = np.concatenate(ch_vasmap, axis=0)
    # print(save_vasmap.shape)
    save_vasmap = ia.array_nor(np.mean(save_vasmap, axis=0))
    # print(save_vasmap.shape)
    tf.imsave('{}_{}.tif'.format(save_name, ch_n), save_vasmap.astype(np.float32))
