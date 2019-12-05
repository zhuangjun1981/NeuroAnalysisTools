import sys
sys.path.extend(['/home/junz/PycharmProjects/corticalmapping'])
import os
import numpy as np
import h5py
# import datetime
import pandas as pd
import corticalmapping.DatabaseTools as dt
import corticalmapping.core.ImageAnalysis as ia
import corticalmapping.SingleCellAnalysis as sca
from shutil import copyfile

df_fn = 'dataframe_190530171338_axon_AllStimuli_DistanceThr_1.30.csv'
clu_folder = r'intermediate_results\bouton_clustering\AllStimuli_DistanceThr_1.30'
nwb_folder = 'nwbs'
save_folder = "intermediate_results"

response_dir = 'pos'
skew_thr = 0.6
analysis_params = dt.ANALYSIS_PARAMS

notes = '''
   zscore receptive field maps of all significant rois. Spatial temporal receptive fields
   are first converted to df/f. Then 2-d zscore maps are generated. Then the zscore maps are
   2d filtered to smooth and interpolated in to high resolution. After preprocessing, if the
   peak value of zscore is larger than the threshold, the receptive field will be considered 
   as sigificant.
        '''

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

save_folder = os.path.join(save_folder, 'rf_maps_' + os.path.splitext(df_fn)[0])
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

copyfile(os.path.realpath(__file__),
         os.path.join(save_folder,
                      'script_log.py'))

df_axon = pd.read_csv(df_fn)
df_axon = df_axon[np.logical_not(df_axon['rf_{}_on_peak_z'.format(response_dir)].isnull())]
df_axon = df_axon[df_axon['skew_fil'] >= skew_thr]

df_axon = df_axon[(df_axon['rf_{}_on_peak_z'.format(response_dir)] >= analysis_params['rf_z_thr_abs']) |
                  (df_axon['rf_{}_off_peak_z'.format(response_dir)] >= analysis_params['rf_z_thr_abs'])]

plane_df = df_axon[['date', 'mouse_id', 'plane_n']].drop_duplicates().reset_index()
print('total number of planes with lsn data: {}'.format(len(plane_df)))

for plane_i, plane_row in plane_df.iterrows():
    date = int(plane_row['date'])
    mid = plane_row['mouse_id']
    plane_n = plane_row['plane_n']

    print('processing {}_{}_{}, {} / {}'.format(date, mid, plane_n, plane_i+1, len(plane_df)))

    subdf = df_axon[(df_axon['date'] == date) &
                    (df_axon['mouse_id'] == mid) &
                    (df_axon['plane_n'] == plane_n)]

    nwb_fn = '{}_{}_110_repacked.nwb'.format(date, mid)
    nwb_f = h5py.File(os.path.join(nwb_folder, nwb_fn), 'r')

    clu_fn = '{}_{}_{}_axon_grouping.hdf5'.format(date, mid, plane_n)
    clu_f = h5py.File(os.path.join(clu_folder, clu_fn), 'r')

    save_fn = '{}_{}_{}_{}.hdf5'.format(date, mid, plane_n, response_dir)
    save_f = h5py.File(os.path.join(save_folder, save_fn))

    # S2
    s2_df = subdf[(subdf['rf_{}_on_peak_z'.format(response_dir)] >= analysis_params['rf_z_thr_abs']) &
                  (subdf['rf_{}_off_peak_z'.format(response_dir)] >= analysis_params['rf_z_thr_abs'])].reset_index()

    if len(s2_df) > 0:
        s2_grp = save_f.create_group('{}_{}_{}_{}_ONOFF'.format(date, mid, plane_n, response_dir))
        s1_on_grp = save_f.create_group('{}_{}_{}_{}_ON'.format(date, mid, plane_n, response_dir))
        s1_off_grp = save_f.create_group('{}_{}_{}_{}_OFF'.format(date, mid, plane_n, response_dir))

        for roi_i, roi_row in s2_df.iterrows():

            print('\t s2 receptive fields, {}, {} / {} ...'.format(roi_row['roi_n'], roi_i + 1, len(s2_df)))

            if response_dir == 'pos':
                _, _, _, srf_on, srf_off, _, _, _, _, _, _, _, _, \
                _ = dt.get_everything_from_axon(nwb_f=nwb_f,
                                                clu_f=clu_f,
                                                plane_n=plane_n,
                                                axon_n=roi_row['roi_n'],
                                                params=analysis_params)

                _, rf_on_new = dt.get_rf_properties(srf=srf_on,
                                                    polarity='positive',
                                                    sigma=analysis_params['gaussian_filter_sigma_rf'],
                                                    interpolate_rate=analysis_params['interpolate_rate_rf'],
                                                    z_thr_abs=analysis_params['rf_z_thr_abs'],
                                                    z_thr_rel=analysis_params['rf_z_thr_rel'])

                _, rf_off_new = dt.get_rf_properties(srf=srf_off,
                                                     polarity='positive',
                                                     sigma=analysis_params['gaussian_filter_sigma_rf'],
                                                     interpolate_rate=analysis_params['interpolate_rate_rf'],
                                                     z_thr_abs=analysis_params['rf_z_thr_abs'],
                                                     z_thr_rel=analysis_params['rf_z_thr_rel'])

            elif response_dir == 'neg':
                _, _, _, _, _, srf_on, srf_off, _, _, _, _, _, _, \
                _ = dt.get_everything_from_axon(nwb_f=nwb_f,
                                                clu_f=clu_f,
                                                plane_n=plane_n,
                                                axon_n=roi_row['roi_n'],
                                                params=analysis_params)

                _, rf_on_new = dt.get_rf_properties(srf=srf_on,
                                                    polarity='negative',
                                                    sigma=analysis_params['gaussian_filter_sigma_rf'],
                                                    interpolate_rate=analysis_params['interpolate_rate_rf'],
                                                    z_thr_abs=analysis_params['rf_z_thr_abs'],
                                                    z_thr_rel=analysis_params['rf_z_thr_rel'])

                _, rf_off_new = dt.get_rf_properties(srf=srf_off,
                                                     polarity='negative',
                                                     sigma=analysis_params['gaussian_filter_sigma_rf'],
                                                     interpolate_rate=analysis_params['interpolate_rate_rf'],
                                                     z_thr_abs=analysis_params['rf_z_thr_abs'],
                                                     z_thr_rel=analysis_params['rf_z_thr_rel'])
            else:
                raise ValueError

            rf_on_mask = rf_on_new.get_weighted_mask()
            rf_off_mask = rf_off_new.get_weighted_mask()
            rf_onoff_new = sca.SpatialReceptiveField(mask=np.max([rf_on_mask, rf_off_mask], axis=0),
                                                     altPos=rf_on_new.altPos,
                                                     aziPos=rf_on_new.aziPos,
                                                     sign='ON_OFF',
                                                     thr=analysis_params['rf_z_thr_abs'])

            curr_s2_grp = s2_grp.create_group(roi_row['roi_n'])
            rf_onoff_new.to_h5_group(curr_s2_grp)

            curr_s1_on_grp = s1_on_grp.create_group(roi_row['roi_n'])
            rf_on_new.to_h5_group(curr_s1_on_grp)

            curr_s1_off_grp = s1_off_grp.create_group(roi_row['roi_n'])
            rf_off_new.to_h5_group(curr_s1_off_grp)

    # positive S1 ON
    s1_on_df = subdf[(subdf['rf_{}_on_peak_z'.format(response_dir)] >= analysis_params['rf_z_thr_abs']) &
                     (subdf['rf_{}_off_peak_z'.format(response_dir)] < analysis_params['rf_z_thr_abs'])].reset_index()

    if len(s1_on_df) > 0:

        s1_on_grp_n = '{}_{}_{}_{}_ON'.format(date, mid, plane_n, response_dir)

        if s1_on_grp_n in save_f.keys():
            s1_on_grp = save_f[s1_on_grp_n]
        else:
            s1_on_grp = save_f.create_group(s1_on_grp_n)

        for roi_i, roi_row in s1_on_df.iterrows():

            print('\t s1 ON receptive fields, {}, {} / {} ...'.format(roi_row['roi_n'], roi_i + 1, len(s1_on_df)))

            if response_dir == 'pos':
                _, _, _, srf_on, _, _, _, _, _, _, _, _, _, \
                _ = dt.get_everything_from_axon(nwb_f=nwb_f,
                                                clu_f=clu_f,
                                                plane_n=plane_n,
                                                axon_n=roi_row['roi_n'],
                                                params=analysis_params)

                _, rf_on_new = dt.get_rf_properties(srf=srf_on,
                                                    polarity='positive',
                                                    sigma=analysis_params['gaussian_filter_sigma_rf'],
                                                    interpolate_rate=analysis_params['interpolate_rate_rf'],
                                                    z_thr_abs=analysis_params['rf_z_thr_abs'],
                                                    z_thr_rel=analysis_params['rf_z_thr_rel'])
            elif response_dir == 'neg':
                _, _, _, _, _, srf_on, _, _, _, _, _, _, _, \
                _ = dt.get_everything_from_axon(nwb_f=nwb_f,
                                                clu_f=clu_f,
                                                plane_n=plane_n,
                                                axon_n=roi_row['roi_n'],
                                                params=analysis_params)

                _, rf_on_new = dt.get_rf_properties(srf=srf_on,
                                                    polarity='negative',
                                                    sigma=analysis_params['gaussian_filter_sigma_rf'],
                                                    interpolate_rate=analysis_params['interpolate_rate_rf'],
                                                    z_thr_abs=analysis_params['rf_z_thr_abs'],
                                                    z_thr_rel=analysis_params['rf_z_thr_rel'])
            else:
                print(response_dir)
                raise ValueError

            curr_s1_on_grp = s1_on_grp.create_group(roi_row['roi_n'])
            rf_on_new.to_h5_group(curr_s1_on_grp)

    # positive S1 OFF
    s1_off_df = subdf[(subdf['rf_{}_on_peak_z'.format(response_dir)] < analysis_params['rf_z_thr_abs']) &
                      (subdf['rf_{}_off_peak_z'.format(response_dir)] >= analysis_params['rf_z_thr_abs'])].reset_index()

    if len(s1_off_df) > 0:

        s1_off_grp_n = '{}_{}_{}_{}_OFF'.format(date, mid, plane_n, response_dir)

        if s1_off_grp_n in save_f.keys():
            s1_off_grp = save_f[s1_off_grp_n]
        else:
            s1_off_grp = save_f.create_group(s1_off_grp_n)

        for roi_i, roi_row in s1_off_df.iterrows():

            print('\t s1 OFF receptive fields, {}, {} / {} ...'.format(roi_row['roi_n'], roi_i + 1, len(s1_off_df)))

            if response_dir == 'pos':
                _, _, _, _, srf_off, _, _, _, _, _, _, _, _, \
                _ = dt.get_everything_from_axon(nwb_f=nwb_f,
                                                clu_f=clu_f,
                                                plane_n=plane_n,
                                                axon_n=roi_row['roi_n'],
                                                params=analysis_params)

                _, rf_off_new = dt.get_rf_properties(srf=srf_off,
                                                     polarity='positive',
                                                     sigma=analysis_params['gaussian_filter_sigma_rf'],
                                                     interpolate_rate=analysis_params['interpolate_rate_rf'],
                                                     z_thr_abs=analysis_params['rf_z_thr_abs'],
                                                     z_thr_rel=analysis_params['rf_z_thr_rel'])

            elif response_dir == 'neg':

                _, _, _, _, _, _, srf_off, _, _, _, _, _, _, \
                _ = dt.get_everthing_from_axon(nwb_f=nwb_f,
                                               clu_f=clu_f,
                                               plane_n=plane_n,
                                               axon_n=roi_row['roi_n'],
                                               params=analysis_params)

                _, rf_off_new = dt.get_rf_properties(srf=srf_off,
                                                     polarity='negative',
                                                     sigma=analysis_params['gaussian_filter_sigma_rf'],
                                                     interpolate_rate=analysis_params['interpolate_rate_rf'],
                                                     z_thr_abs=analysis_params['rf_z_thr_abs'],
                                                     z_thr_rel=analysis_params['rf_z_thr_rel'])

            else:
                raise ValueError

            curr_s1_off_grp = s1_off_grp.create_group(roi_row['roi_n'])
            rf_off_new.to_h5_group(curr_s1_off_grp)

    save_f.close()




