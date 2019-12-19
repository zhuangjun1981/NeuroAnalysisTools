import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import NeuroAnalysisTools.core.ImageAnalysis as ia

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\190822-M471944-deepscope\movie"
identifier = '110_LSNDGCUC'
start_ind = 121228
frame_num = 3

fns = []

for ind in np.arange(frame_num, dtype=np.int) + start_ind:

    if ind < 100000:
        fns.append('{}_{:05d}_00001.tif'.format(identifier, ind))
    elif ind < 1000000:
        fns.append('{}_{:06d}_00001.tif'.format(identifier, ind))
    elif ind < 10000000:
        fns.append('{}_{:07d}_00001.tif'.format(identifier, ind))

f = plt.figure(figsize=(5, 12))
for frame_i in range(frame_num):
    ax = f.add_subplot(frame_num, 1, frame_i+1)
    ax.imshow(ia.array_nor(tf.imread(os.path.join(data_folder, fns[frame_i]))), cmap='gray',
              vmin=0, vmax=0.5, interpolation='nearest')
    ax.set_title(fns[frame_i])
    ax.set_axis_off()

plt.tight_layout()
plt.show()
