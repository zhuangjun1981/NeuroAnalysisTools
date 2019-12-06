import sys
sys.path.extend(['/home/junz/PycharmProjects/corticalmapping'])
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import corticalmapping.core.ImageAnalysis as ia
import corticalmapping.DatabaseTools as dt

table_name = 'big_roi_table_test.xlsx'
sheet_name = 'f_center_subtracted'

response_dir = 'pos'
skew_thr = 0.6
rf_peak_z_thr = 1.6

save_fn = 'plane_rf_centers.pdf'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

table_path = os.path.join(curr_folder, table_name)
df = pd.read_excel(table_path, sheetname=sheet_name)
subdf = df[df['skew_fil'] >= skew_thr]

planes = subdf[['date', 'mouse_id', 'plane_n', 'depth']].drop_duplicates().reset_index()
print(planes)

pdff = PdfPages(os.path.join('intermediate_figures', save_fn))

for plane_i, plane_row in planes.iterrows():

    print('plotting {}_{}_{}, {} / {}'.format(
        plane_row['date'],
        plane_row['mouse_id'],
        plane_row['plane_n'],
        plane_i + 1,
        len(planes)))

    planedf = subdf[(subdf['date'] == plane_row['date']) & \
        (subdf['mouse_id'] == plane_row['mouse_id']) & \
        (subdf['plane_n'] == plane_row['plane_n']) & \
        (subdf['depth'] == plane_row['depth'])]

    df_or = planedf[planedf['rf_{}_onoff_peak_z'.format(response_dir)] >= rf_peak_z_thr]
    df_and = planedf[(planedf['rf_{}_on_peak_z'.format(response_dir)] >= rf_peak_z_thr) & \
                     (planedf['rf_{}_off_peak_z'.format(response_dir)] >= rf_peak_z_thr)]
    df_on = planedf[planedf['rf_{}_on_peak_z'.format(response_dir)] >= rf_peak_z_thr].drop(df_and.index)
    df_off = planedf[planedf['rf_{}_off_peak_z'.format(response_dir)] >= rf_peak_z_thr].drop(df_and.index)

    df_or = df_or.reset_index()
    df_and = df_and.reset_index()
    df_on = df_on.reset_index()
    df_off = df_off.reset_index()

    if len(df_or) == 0:
        print('no any receptive fields. skip.')
    else:
        print('\tnumber of rois with significant rf: {}'.format(len(df_or)))
        print('\tnumber of rois with S1 ON: {}'.format(len(df_on)))
        print('\tnumber of rois with S1 OFF: {}'.format(len(df_off)))
        print('\tnumber of rois with S2 ON/OFF: {}'.format(len(df_and)))

        f = plt.figure(figsize=(11, 8.5))

        f.suptitle('{}_{}_{}; {} um'.format(plane_row['date'],
                                            plane_row['mouse_id'],
                                            plane_row['plane_n'],
                                            plane_row['depth']))

        #=============================RF center=============================================
        # ON/OFF
        alt_min = int(np.min(df_or['rf_{}_onoff_center_alt'.format(response_dir)]) - 5)
        alt_max = int(np.max(df_or['rf_{}_onoff_center_alt'.format(response_dir)]) + 5)
        azi_min = int(np.min(df_or['rf_{}_onoff_center_azi'.format(response_dir)]) - 5)
        azi_max = int(np.max(df_or['rf_{}_onoff_center_azi'.format(response_dir)]) + 5)
        ax_or_scatter = f.add_subplot(4, 5, 1)
        ax_or_scatter.plot(df_or['rf_{}_onoff_center_azi'.format(response_dir)],
                          df_or['rf_{}_onoff_center_alt'.format(response_dir)],
                              '.', color='#888888')
        ax_or_scatter.set_xlim([azi_min, azi_max])
        ax_or_scatter.set_ylim([alt_min, alt_max])
        ax_or_scatter.set_title('RF center')

        # ON
        ax_on_scatter = f.add_subplot(4, 5, 6)
        ax_on_scatter.plot(df_off['rf_{}_off_center_azi'.format(response_dir)],
                           df_off['rf_{}_off_center_alt'.format(response_dir)],
                           '.', color='#aaaaaa')
        ax_on_scatter.plot(df_on['rf_{}_on_center_azi'.format(response_dir)],
                           df_on['rf_{}_on_center_alt'.format(response_dir)],
                           '.', color='#ff0000')
        ax_on_scatter.set_xlim([azi_min, azi_max])
        ax_on_scatter.set_ylim([alt_min, alt_max])

        # OFF
        ax_off_scatter = f.add_subplot(4, 5, 11)
        ax_off_scatter.plot(df_on['rf_{}_on_center_azi'.format(response_dir)],
                            df_on['rf_{}_on_center_alt'.format(response_dir)],
                            '.', color='#aaaaaa')
        ax_off_scatter.plot(df_off['rf_{}_off_center_azi'.format(response_dir)],
                            df_off['rf_{}_off_center_alt'.format(response_dir)],
                            '.', color='#0000ff')
        ax_off_scatter.set_xlim([azi_min, azi_max])
        ax_off_scatter.set_ylim([alt_min, alt_max])

        # ON-OFF
        ax_and_scatter = f.add_subplot(4, 5, 16)
        ax_and_scatter.plot(df_and['rf_{}_on_center_azi'.format(response_dir)],
                            df_and['rf_{}_on_center_alt'.format(response_dir)],
                            '.', color='#ff0000')
        ax_and_scatter.plot(df_and['rf_{}_off_center_azi'.format(response_dir)],
                            df_and['rf_{}_off_center_alt'.format(response_dir)],
                            '.', color='#0000ff')
        ax_and_scatter.set_xlim([azi_min, azi_max])
        ax_and_scatter.set_ylim([alt_min, alt_max])

        # =============================pairwise distance=============================================
        dis_or = ia.pairwise_distance(df_or[['rf_{}_onoff_center_azi'.format(response_dir),
                                             'rf_{}_onoff_center_alt'.format(response_dir)]].values)
        ax_or_pd = f.add_subplot(4, 5, 2)
        if len(dis_or) > 0:
            ax_or_pd.hist(dis_or, range=[0, 80], bins=20, facecolor='#aaaaaa', edgecolor='none')
        ax_or_pd.get_yaxis().set_ticks([])
        ax_or_pd.set_title('pw RF dis') # pairwise receptive field center distance

        dis_on = ia.pairwise_distance(df_on[['rf_{}_on_center_azi'.format(response_dir),
                                             'rf_{}_on_center_alt'.format(response_dir)]].values)
        ax_on_pd = f.add_subplot(4, 5, 7)
        if len(dis_on) > 0:
            ax_on_pd.hist(dis_on, range=[0, 80], bins=20, facecolor='#ff0000', edgecolor='none')
        ax_on_pd.get_yaxis().set_ticks([])

        dis_off = ia.pairwise_distance(df_off[['rf_{}_off_center_azi'.format(response_dir),
                                               'rf_{}_off_center_alt'.format(response_dir)]].values)
        ax_off_pd = f.add_subplot(4, 5, 12)
        if len(dis_off) > 0:
            ax_off_pd.hist(dis_off, range=[0, 80], bins=20, facecolor='#0000ff', edgecolor='none')
        ax_off_pd.get_yaxis().set_ticks([])

        dis_and_on = ia.pairwise_distance(df_and[['rf_{}_on_center_azi'.format(response_dir),
                                                  'rf_{}_on_center_alt'.format(response_dir)]].values)
        dis_and_off = ia.pairwise_distance(df_and[['rf_{}_off_center_azi'.format(response_dir),
                                                   'rf_{}_off_center_alt'.format(response_dir)]].values)
        ax_and_pd = f.add_subplot(4, 5, 17)
        if len(dis_and_on) > 0:
            ax_and_pd.hist(dis_and_on, range=[0, 80], bins=20, facecolor='#ff0000', edgecolor='none', alpha=0.5)
            ax_and_pd.hist(dis_and_off, range=[0, 80], bins=20, facecolor='#0000ff', edgecolor='none', alpha=0.5)
        ax_and_pd.get_yaxis().set_ticks([])

        # =============================parewise magnification=============================================
        mag_or = ia.pairwise_magnification(df_or[['rf_{}_onoff_center_azi'.format(response_dir),
                                                  'rf_{}_onoff_center_alt'.format(response_dir)]].values,
                                           df_or[['roi_center_col', 'roi_center_row']].values)
        ax_or_pm = f.add_subplot(4, 5, 3)
        if len(mag_or) > 0:
            mag_or = 0.00035 / mag_or  # 0.35 um per pixel
            ax_or_pm.hist(mag_or, range=[0, 0.2], bins=20, facecolor='#aaaaaa', edgecolor='none')
        ax_or_pm.get_yaxis().set_ticks([])
        ax_or_pm.set_title('mm/deg') # pairwise magnification
        #
        mag_on = ia.pairwise_magnification(df_on[['rf_{}_on_center_azi'.format(response_dir),
                                                  'rf_{}_on_center_alt'.format(response_dir)]].values,
                                           df_on[['roi_center_col', 'roi_center_row']].values)
        ax_on_pm = f.add_subplot(4, 5, 8)
        if len(mag_on) > 0:
            mag_on = 0.00035 / mag_on  # 0.35 um per pixel
            ax_on_pm.hist(mag_on, range=[0, 0.2], bins=20, facecolor='#ff0000', edgecolor='none')
        ax_on_pm.get_yaxis().set_ticks([])

        mag_off = ia.pairwise_magnification(df_off[['rf_{}_off_center_azi'.format(response_dir),
                                                    'rf_{}_off_center_alt'.format(response_dir)]].values,
                                            df_off[['roi_center_col', 'roi_center_row']].values)
        ax_off_pm = f.add_subplot(4, 5, 13)
        if len(mag_off) > 0:
            mag_off = 0.00035 / mag_off  # 0.35 um per pixel
            ax_off_pm.hist(mag_off, range=[0, 0.2], bins=20, facecolor='#0000ff', edgecolor='none')
        ax_off_pm.get_yaxis().set_ticks([])

        mag_and_on = ia.pairwise_magnification(df_and[['rf_{}_on_center_azi'.format(response_dir),
                                                       'rf_{}_on_center_alt'.format(response_dir)]].values,
                                               df_and[['roi_center_col', 'roi_center_row']].values)

        mag_and_off = ia.pairwise_magnification(df_and[['rf_{}_off_center_azi'.format(response_dir),
                                                        'rf_{}_off_center_alt'.format(response_dir)]].values,
                                                df_and[['roi_center_col', 'roi_center_row']].values)

        ax_and_pm = f.add_subplot(4, 5, 18)
        if len(mag_and_on) > 0:
            mag_and_on = 0.00035 / mag_and_on  # 0.35 um per pixel
            mag_and_off = 0.00035 / mag_and_off  # 0.35 um per pixel
            ax_and_pm.hist(mag_and_on, range=[0, 0.2], bins=20, facecolor='#ff0000', edgecolor='none', alpha=0.5,)
            ax_and_pm.hist(mag_and_off, range=[0, 0.2], bins=20, facecolor='#0000ff', edgecolor='none', alpha=0.5,)
        ax_and_pm.get_yaxis().set_ticks([])

        # =============================azi alt spatial distribution=============================================
        ax_alt_or = f.add_subplot(4, 5, 4)
        ax_alt_or.set_title('altitude')
        ax_azi_or = f.add_subplot(4, 5, 5)
        ax_azi_or.set_title('azimuth')
        if len(df_or) > 0:
            dt.plot_roi_retinotopy(coords_rf=df_or[['rf_{}_onoff_center_alt'.format(response_dir),
                                                    'rf_{}_onoff_center_azi'.format(response_dir)]].values,
                                   coords_roi=df_or[['roi_center_row', 'roi_center_col']].values,
                                   ax_alt=ax_alt_or,
                                   ax_azi=ax_azi_or,
                                   cmap='viridis',
                                   canvas_shape=(512, 512),
                                   edgecolors='#000000',
                                   linewidth=0.5)
        else:
            ax_alt_or.set_xticks([])
            ax_alt_or.set_yticks([])
            ax_azi_or.set_xticks([])
            ax_azi_or.set_yticks([])

        ax_alt_on = f.add_subplot(4, 5, 9)
        ax_azi_on = f.add_subplot(4, 5, 10)
        if len(df_on) > 0:
            dt.plot_roi_retinotopy(coords_rf=df_on[['rf_{}_on_center_alt'.format(response_dir),
                                                    'rf_{}_on_center_azi'.format(response_dir)]].values,
                                   coords_roi=df_on[['roi_center_row', 'roi_center_col']].values,
                                   ax_alt=ax_alt_on,
                                   ax_azi=ax_azi_on,
                                   cmap='viridis',
                                   canvas_shape=(512, 512),
                                   edgecolors='#000000',
                                   linewidth=0.5)
        else:
            ax_alt_on.set_xticks([])
            ax_alt_on.set_yticks([])
            ax_azi_on.set_xticks([])
            ax_azi_on.set_yticks([])

        ax_alt_off = f.add_subplot(4, 5, 14)
        ax_azi_off = f.add_subplot(4, 5, 15)
        if len(df_off) > 0:
            dt.plot_roi_retinotopy(coords_rf=df_off[['rf_{}_off_center_alt'.format(response_dir),
                                                     'rf_{}_off_center_azi'.format(response_dir)]].values,
                                   coords_roi=df_off[['roi_center_row', 'roi_center_col']].values,
                                   ax_alt=ax_alt_off,
                                   ax_azi=ax_azi_off,
                                   cmap='viridis',
                                   canvas_shape=(512, 512),
                                   edgecolors='#000000',
                                   linewidth=0.5)
        else:
            ax_alt_off.set_xticks([])
            ax_alt_off.set_yticks([])
            ax_azi_off.set_xticks([])
            ax_azi_off.set_yticks([])

        # plt.tight_layout()
        # plt.show()
        pdff.savefig(f)
        f.clear()
        plt.close(f)

pdff.close()

print('for debug ...')
