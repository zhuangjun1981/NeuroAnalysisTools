import os
import datetime
import pandas as pd

meta_fn = "plane_table_190530165648.xlsx"

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

meta_df = pd.read_excel(meta_fn, sheet_name='sheet1')

meta_df.sort_values(by=['mouse_id', 'volume_n', 'depth'], inplace=True)
meta_df.reset_index(inplace=True, drop=True)

print(meta_df)

date_str = datetime.datetime.now().strftime('%y%m%d%H%M%S')
with pd.ExcelWriter('plane_table_{}.xlsx'.format(date_str), mode='w') as writer:
    meta_df.to_excel(writer, sheet_name='sheet1')