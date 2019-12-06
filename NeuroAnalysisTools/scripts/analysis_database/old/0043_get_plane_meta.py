import os
import datetime
import pandas as pd
import h5py

df_folder = 'dataframes_190529210731'
save_fn = 'plane_table'
nwb_folder = 'nwbs/small_nwbs'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

fns = [fn for fn in os.listdir(df_folder) if fn[-5:] == '.xlsx']
print('\n'.join(fns))

df = pd.DataFrame(index=range(len(fns)), columns=['date', 'mouse_id', 'plane_n', 'volume_n',
                                                  'depth', 'has_lsn', 'has_dgc'])

for fn_i, fn in enumerate(fns):
    print(fn)

    date = fn.split('_')[0]
    mouse_id = fn.split('_')[1]
    plane_n = fn.split('_')[-1][0:-5]

    nwb_path = os.path.join(curr_folder, nwb_folder, '{}_{}_110_repacked.nwb'.format(date, mouse_id))
    nwb_f = h5py.File(nwb_path, 'r')
    depth = nwb_f['processing/rois_and_traces_{}/imaging_depth_micron'.format(plane_n)].value
    nwb_f.close()

    df.loc[fn_i] = [date, mouse_id, plane_n, '', depth, True, True]

df.sort_values(by=['mouse_id', 'date', 'plane_n'], inplace=True)
df.reset_index(inplace=True, drop=True)

print(df)

date_str = datetime.datetime.now().strftime('%y%m%d%H%M%S')
with pd.ExcelWriter('{}_{}.xlsx'.format(save_fn, date_str), mode='w') as writer:
    df.to_excel(writer, sheet_name='sheet1')