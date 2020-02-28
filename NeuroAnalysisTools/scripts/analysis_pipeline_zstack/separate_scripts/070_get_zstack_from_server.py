import os
import numpy as np
import tifffile as tf

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180816-M376019-zstack\zstack_2p_zoom2"

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
        zstack.append(tf.imread(os.path.join(ch_folder, step, 'corrected_mean_projection.tif')))

    zstack = np.array(zstack)
    save_n = os.path.split(data_folder)[1] + '_' + chn + '.tif'
    tf.imsave(save_n, zstack)
