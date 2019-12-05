import os
import tifffile as tf

data_fn = 'zstack_zoom2_00001_00001.tif'
ch_ns = ['green', 'red']
save_prefix = 'zstack'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

ch_num = len(ch_ns)

stack = tf.imread(data_fn)

for ch_i, ch_n in enumerate(ch_ns):
    tf.imsave('{}_{}.tif'.format(save_prefix, ch_n), stack[ch_i::ch_num])

print('done.')