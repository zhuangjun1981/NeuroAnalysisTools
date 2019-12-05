import os
import numpy as np
import tifffile as tf

data_folder = r"\\sd2\SD2\jun_backup\raw_data_temp\181217-M421211-2p\zstack"

chns = ['green', 'red']

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

for chn in chns:
    print('processing channel: {} ...'.format(chn))
    ch_folder = os.path.join(data_folder, chn)
    steps = [f for f in os.listdir(ch_folder) if os.path.isdir(os.path.join(ch_folder, f)) and f[0:5] == 'step_']
    steps.sort()
    print('\ntotal number of steps: {}'.format(len(steps)))

    zstack = []
    for step in steps:
        print("\t{}".format(step))
        movie = tf.imread(os.path.join(ch_folder, step, step + '.tif'))
        zstack.append(np.mean(movie, axis=0))

    zstack = np.array(zstack, dtype=np.float32)
    save_n = os.path.split(data_folder)[1] + '_uncorrected_' + chn + '.tif'
    tf.imsave(save_n, zstack)
