import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import NeuroAnalysisTools.SingleCellAnalysis as sca
import scipy.stats as stats
import h5py

# df_path = r"G:\bulk_LGN_database\dataframe_190530171338.csv"
# rf_maps_folder = r"intermediate_results\rf_maps_dataframes_190529210731"

df_path = r"G:\bulk_LGN_database\dataframe_190530171338_axon_AllStimuli_DistanceThr_1.30.csv"
rf_maps_folder = r"G:\bulk_LGN_database\intermediate_results" \
                 r"\rf_maps_dataframe_190530171338_axon_AllStimuli_DistanceThr_1.30"

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
osi_type = 'gosi'
osi_thr = 1. / 3.

ellipse_aspect_thr = 1.0

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

if dire_type == 'peak_dire' and (post_process_type == 'ele' or post_process_type == 'rec'):
    dire_pp = 'raw'
else:
    dire_pp = post_process_type

print('loading csv file: {}'.format(df_path))
df = pd.read_csv(df_path)
print('csv file loaded.')

df = df[(df['mouse_id'].isin(mouse_ids)) & \
        (df['skew_fil'] >= skew_thr) & \
        (df['dgc_{}_peak_z'.format(response_dir)] >= dgc_peak_z_thr) & \
        (df['dgc_p_anova_{}'.format(response_type)] <= dgc_p_anova_thr) & \
        (np.isfinite(df['rf_{}_on_peak_z'.format(response_dir)]))]

osdf = df[(df['dgc_{}_{}_{}_{}'.format(response_dir, osi_type, post_process_type, response_type)] >= osi_thr) & \
          (df['dgc_{}_{}_{}_{}'.format(response_dir, dsi_type, post_process_type, response_type)] < dsi_thr)]

os_diff_onoff = []
os_diff_on = []
os_diff_off = []
for roi_i, roi_row in osdf.iterrows():
    date = int(roi_row['date'])
    mid = roi_row['mouse_id']
    plane_n = roi_row['plane_n']
    roi_n = roi_row['roi_n']

    map_fn = '{}_{}_{}_{}'.format(date, mid, plane_n, response_dir)
    map_f = h5py.File(os.path.join(rf_maps_folder, map_fn + '.hdf5'), 'r')

    on_grp = map_f['{}_ON'.format(map_fn)]
    off_grp = map_f['{}_OFF'.format(map_fn)]

    dire = roi_row['dgc_{}_{}_{}_{}'.format(response_dir, dire_type, dire_pp, response_type)]
    ori = sca.dire2ori(dire)

    if roi_n in on_grp.keys() and roi_n in off_grp.keys():
        rf_on = sca.SpatialReceptiveField.from_h5_group(on_grp[roi_n])
        rf_off = sca.SpatialReceptiveField.from_h5_group(off_grp[roi_n])
        c_alt_on, c_azi_on = rf_on.get_weighted_rf_center()
        c_alt_off, c_azi_off = rf_off.get_weighted_rf_center()

        onoff_ang = np.arctan((c_alt_on - c_alt_off) / (c_azi_on - c_azi_off))
        onoff_ang = onoff_ang * 180. / np.pi
        onoff_ang = sca.dire2ori(onoff_ang)

        curr_diff = abs(onoff_ang - ori)
        if curr_diff > 90.:
            curr_diff = 180 - curr_diff

        os_diff_onoff.append(curr_diff)

    elif roi_n in on_grp.keys():
        rf_on = sca.SpatialReceptiveField.from_h5_group(on_grp[roi_n])
        ell_on = rf_on.ellipse_fitting(is_plot=False)
        if ell_on is not None and ell_on.get_aspect_ratio() >= ellipse_aspect_thr:
            curr_diff = abs(ell_on.angle - ori)
            if curr_diff > 90.:
                curr_diff = 180 - curr_diff
            os_diff_on.append(curr_diff)

    elif roi_n in off_grp.keys():
        rf_off = sca.SpatialReceptiveField.from_h5_group(off_grp[roi_n])
        ell_off = rf_off.ellipse_fitting(is_plot=False)
        if ell_off is not None and ell_off.get_aspect_ratio() >= ellipse_aspect_thr:
            curr_diff = abs(ell_off.angle - ori)
            if curr_diff > 90.:
                curr_diff = 180 - curr_diff
            os_diff_off.append(curr_diff)

print('\nOrientation Selective ROIs:')
print('\tWith ONOFF receptive fields:')
print('\t\tn={}'.format(len(os_diff_onoff)))
print('\t\torie difference predicted vs. measured, mean={}'.format(np.mean(os_diff_onoff)))
print('\t\torie difference predicted vs. measured, std={}'.format(np.std(os_diff_onoff)))
chisq_os_onoff, p_os_onoff = stats.chisquare(np.histogram(os_diff_onoff, range=[0., 90.], bins=20)[0])
print('\t\tagainst uniform distribution: chi-squared={}, p={}'.format(chisq_os_onoff, p_os_onoff))

print('\tWith only ON receptive fields:')
print('\t\tn={}'.format(len(os_diff_on)))
print('\t\torie difference predicted vs. measured, mean={}'.format(np.mean(os_diff_on)))
print('\t\torie difference predicted vs. measured, std={}'.format(np.std(os_diff_on)))
chisq_os_on, p_os_on = stats.chisquare(np.histogram(os_diff_on, range=[0., 90.], bins=20)[0])
print('\t\tagainst uniform distribution: chi-squared={}, p={}'.format(chisq_os_on, p_os_on))

print('\tWith only OFF receptive fields:')
print('\t\tn={}'.format(len(os_diff_off)))
print('\t\torie difference predicted vs. measured, mean={}'.format(np.mean(os_diff_off)))
print('\t\torie difference predicted vs. measured, std={}'.format(np.std(os_diff_off)))
chisq_os_off, p_os_off = stats.chisquare(np.histogram(os_diff_off, range=[0., 90.], bins=20)[0])
print('\t\tagainst uniform distribution: chi-squared={}, p={}'.format(chisq_os_off, p_os_off))

os_diff_all = os_diff_onoff + os_diff_on + os_diff_off
print('\tWith all receptive fields:')
print('\t\tn={}'.format(len(os_diff_all)))
print('\t\torie difference predicted vs. measured, mean={}'.format(np.mean(os_diff_all)))
print('\t\torie difference predicted vs. measured, std={}'.format(np.std(os_diff_all)))
chisq_os_all, p_os_all = stats.chisquare(np.histogram(os_diff_all, range=[0., 90.], bins=20)[0])
print('\t\tagainst uniform distribution: chi-squared={}, p={}'.format(chisq_os_all, p_os_all))


plt.hist([os_diff_onoff, os_diff_on, os_diff_off], range=[0, 90], bins=20, stacked=True,
         color=['purple', 'r', 'b'], ec='none', alpha=0.5)
plt.show()