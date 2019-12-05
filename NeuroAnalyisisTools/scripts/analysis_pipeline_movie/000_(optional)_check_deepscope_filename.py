import os
import numpy as np

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\190822-M471944-deepscope\movie"
identifier = '110_LSNDGCUC'

fns = np.array([f for f in os.listdir(data_folder) if f[-4:] == '.tif' and identifier in f])
f_nums = [int(os.path.splitext(fn)[0].split('_')[-2]) for fn in fns]
fns = fns[np.argsort(f_nums)]
print('total file number: {}'.format(len(fns)))

for i in range(1, len(fns) + 1):

    if i < 100000:
       if fns[i-1] != '{}_{:05d}_00001.tif'.format(identifier, i):
           print('{}th file, name: {}, do not match!'.format(i, fns[i]))
           break
    elif i < 1000000:
        if fns[i - 1] != '{}_{:06d}_00001.tif'.format(identifier, i):
            print('{}th file, name: {}, do not match!'.format(i, fns[i]))
            break
    elif i < 10000000:
        if fns[i - 1] != '{}_{:07d}_00001.tif'.format(identifier, i):
            print('{}th file, name: {}, do not match!'.format(i, fns[i]))
            break
