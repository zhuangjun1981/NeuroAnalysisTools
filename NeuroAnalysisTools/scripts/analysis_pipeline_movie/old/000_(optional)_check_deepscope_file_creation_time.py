import os
import numpy as np
import matplotlib.pyplot as plt

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\190822-M471944-deepscope\movie"
identifier = '110_LSNDGCUC'

fns = np.array([f for f in os.listdir(data_folder) if f[-4:] == '.tif' and identifier in f])
f_nums = [int(os.path.splitext(fn)[0].split('_')[-2]) for fn in fns]
fns = fns[np.argsort(f_nums)]
print('total file number: {}'.format(len(fns)))

ctimes = []

for fn in fns:
    ctimes.append(os.path.getctime(os.path.join(data_folder, fn)))

ctime_diff = np.diff(ctimes)
max_ind = np.argmax(ctime_diff)
print('maximum creation gap: {}'.format(ctime_diff[max_ind]))

fis = np.arange(21, dtype=np.int) - 10 + max_ind

for fi in fis:
    print('{}, ctime: {}s, duration: {}s'.format(fns[fi], ctimes[fi], ctime_diff[fi]))

plt.plot(ctime_diff)
plt.show()
