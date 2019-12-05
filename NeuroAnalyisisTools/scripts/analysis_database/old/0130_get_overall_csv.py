import os
import pandas as pd

df_folder = 'other_dataframes'
# df_fn = 'dataframes_190530171338'
# df_fn = 'dataframes_190530171338_axon_AllStimuli_DistanceThr_0.50'
# df_fn = 'dataframes_190530171338_axon_AllStimuli_DistanceThr_1.00'
df_fn = 'dataframes_190530171338_axon_AllStimuli_DistanceThr_1.30'
plane_df_fn = 'plane_table_190530170648.xlsx'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

csv_fns = [fn for fn in os.listdir(os.path.join(df_folder, df_fn)) if fn[-4:] == '.csv']
csv_fns.sort()

plane_df = pd.read_excel(os.path.join(df_folder, plane_df_fn), sheetname='sheet1')

df_all = []

for csv_fn in csv_fns:
    print('reading {} ...'.format(csv_fn))
    df_all.append(pd.read_csv(os.path.join(df_folder, df_fn, csv_fn)))

df_all = pd.concat(df_all, axis=0)

try:
    df_all.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
except KeyError:
    pass

print(df_all.columns)

df_all['vol_n'] = ''

for plane_i, plane_row in plane_df.iterrows():
    plane_ind = ((df_all['date'] == plane_row['date']) &
                 (df_all['mouse_id'] == plane_row['mouse_id']) &
                 (df_all['plane_n'] == plane_row['plane_n']))
    df_all.loc[plane_ind, 'vol_n'] = plane_row['volume_n']

print(df_all.vol_n.drop_duplicates())

df_all.sort_values(by=['vol_n', 'depth', 'roi_n'], inplace=True)
df_all.reset_index(inplace=True)
df_all.drop(['index'], axis=1, inplace=True)

print(df_all[['date', 'mouse_id', 'plane_n', 'depth', 'vol_n']].drop_duplicates())

df_all.to_csv(df_fn.replace('dataframes', 'dataframe') + '.csv')




