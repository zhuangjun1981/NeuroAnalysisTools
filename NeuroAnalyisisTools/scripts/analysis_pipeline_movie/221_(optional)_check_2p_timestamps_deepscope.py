import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

ts_2p_key = 'digital_vsync_2p_rise'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = h5py.File(nwb_fn, 'r')

ts_2p = nwb_f['acquisition/timeseries/{}/timestamps'.format(ts_2p_key)].value
print('number of 2p frame timestamps: {}'.format(len(ts_2p)))

dur_2p = np.diff(ts_2p)
max_ind = np.argmax(dur_2p)
print('maximum 2p frame duration: {}'.format(dur_2p[max_ind]))

# fis = np.arange(21, dtype=np.int) - 10 + max_ind
#
# for fi in fis:
#     print('{}, ctime: {}s, duration: {}s'.format(fns[fi], ctimes[fi], ctime_diff[fi]))

plt.plot(dur_2p)
plt.show()
