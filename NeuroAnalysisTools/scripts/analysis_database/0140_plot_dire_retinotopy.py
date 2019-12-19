import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# df_path = r"G:\bulk_LGN_database\dataframe_190530171338.csv"
df_path = r"G:\bulk_LGN_database\dataframe_190530171338_axon_AllStimuli_DistanceThr_1.00.csv"

depths = [50, 100, 150, 200, 250, 300, 350, 400,]
mouse_ids = ['M360495', 'M376019', 'M386444', 'M426525', 'M439939', 'M439943']
# mouse_ids = ['M439939']
dire_type = 'peak_dire' # 'vs_dire' or 'peak_dire'
response_dir = 'pos'
response_type = 'dff'
post_process_type = 'ele' # 'raw', 'ele' or 'rec'
skew_thr = 0.6
dgc_peak_z_thr = 3.
dgc_p_anova_thr = 0.01
dsi_type = 'gdsi'
dsi_thr = 0.5

rf_z_thr = 1.6

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

if dire_type == 'peak_dire' and (post_process_type == 'ele' or post_process_type == 'rec'):
    dire_pp = 'raw'
else:
    dire_pp = post_process_type

print('loading csv file: {}'.format(df_path))
df = pd.read_csv(df_path)
print('csv file loaded.')

df = df[df['mouse_id'].isin(mouse_ids)]

df = df[(df['skew_fil'] >= skew_thr) &
        (df['dgc_{}_peak_z'.format(response_dir)] >= dgc_peak_z_thr) &
        (df['dgc_p_anova_{}'.format(response_type)] <= dgc_p_anova_thr) &
        (df['dgc_{}_{}_{}_{}'.format(response_dir, dsi_type, post_process_type, response_type)] >= dsi_thr)]

pdff = PdfPages(os.path.join('intermediate_results', 'preferred_dire_depth.pdf'))

f_all = plt.figure(figsize=(12, 8))
ax_all = f_all.add_subplot(111)
ax_all.set_xlim([0, 90])
ax_all.set_ylim([-30, 30])
ax_all.set_aspect('equal')
ax_all.set_title('all depths')

for depth_i, depth in enumerate(depths):

    depth_df = df[df['depth'] == depth]
    print(len(depth_df))

    f = plt.figure(figsize=(12, 8))
    ax = f.add_subplot(111)
    ax.set_xlim([0, 90])
    ax.set_ylim([-30, 30])
    ax.set_aspect('equal')
    ax.set_title('{} um'.format(depth))

    for roi_i, roi_row in depth_df.iterrows():

        if roi_row['rf_{}_on_peak_z'.format(response_dir)] >= rf_z_thr:
            alt = roi_row['rf_{}_on_center_alt'.format(response_dir)]
            azi = roi_row['rf_{}_on_center_azi'.format(response_dir)]
            dire = roi_row['dgc_{}_{}_{}_{}'.format(response_dir, dire_type, dire_pp, response_type)]
            # print('alt: {:6.2f}, azi: {:6.2f}, dire: {}'.format(alt, azi, dire))
            dire = dire * np.pi / 180.
            bazi = azi - np.cos(dire) * 1.
            dazi = np.cos(dire) * 2.
            balt = alt - np.sin(dire) * 1.
            dalt = np.sin(dire) * 2.

            ax.arrow(x=bazi, y=balt, dx=dazi, dy=dalt, length_includes_head=True,
                     head_width=0.5, head_length=1, ec='none', fc='r', alpha=0.5)
            ax_all.arrow(x=bazi, y=balt, dx=dazi, dy=dalt, length_includes_head=True,
                         head_width=0.5, head_length=1, ec='none', fc='r', alpha=0.5)

        if roi_row['rf_{}_off_peak_z'.format(response_dir)] >= rf_z_thr:
            alt = roi_row['rf_{}_off_center_alt'.format(response_dir)]
            azi = roi_row['rf_{}_off_center_azi'.format(response_dir)]
            dire = roi_row['dgc_{}_{}_{}_{}'.format(response_dir, dire_type, dire_pp, response_type)]
            # print('alt: {:6.2f}, azi: {:6.2f}, dire: {}'.format(alt, azi, dire))
            dire = dire * np.pi / 180.
            bazi = azi - np.sin(dire) * 1.
            dazi = np.sin(dire) * 2.
            balt = alt - np.cos(dire) * 1.
            dalt = np.cos(dire) * 2.

            ax.arrow(x=bazi, y=balt, dx=dazi, dy=dalt, length_includes_head=True,
                     head_width=0.5, head_length=1, ec='none', fc='b', alpha=0.5)
            ax_all.arrow(x=bazi, y=balt, dx=dazi, dy=dalt, length_includes_head=True,
                         head_width=0.5, head_length=1, ec='none', fc='b', alpha=0.5)


    pdff.savefig(f)
    f.clear()
    plt.close(f)

pdff.savefig(f_all)
pdff.close()