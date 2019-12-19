# import sys
# sys.path.extend(['/home/junz/PycharmProjects/corticalmapping'])
import os
import numpy as np
import h5py
import datetime
import pandas as pd
import NeuroAnalysisTools.DatabaseTools as dt
import NeuroAnalysisTools.SingleCellAnalysis as sca
from shutil import copyfile

table_folder = 'dataframes_190529210731'
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

save_folder = os.path.join(save_folder, 'rf_maps_' + table_folder)
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

copyfile(os.path.realpath(__file__),
         os.path.join(save_folder,
                      'script_log_{}.py'.format(datetime.datetime.now().strftime('%y%m%d%H%M%S'))))

table_fns = [f for f in os.listdir(table_folder) if f[-5:] == '.xlsx']
table_fns.sort()
print('number of planes: {}'.format(len(table_fns)))

for table_i, table_fn in enumerate(table_fns):
    print('\nanalyzing {}, {} / {} ... '.format(table_fn, table_i+1, len(table_fns)))

    save_fn = table_fn[0:-5] + '_{}.hdf5'.format(response_dir)

    if os.path.isfile(os.path.join(save_folder, save_fn)):
        print('\tAlready analyzed. Skip.')
        continue

    df = pd.read_excel(os.path.join(table_folder, table_fn), sheetname='sheet1')
    subdf = df[np.logical_not(df['rf_pos_on_peak_z'].isnull())]
    subdf = subdf[subdf['skew_fil'] >= skew_thr]

    subdf = subdf[(subdf['rf_{}_on_peak_z'.format(response_dir)] >= analysis_params['rf_z_thr_abs']) |
                  (subdf['rf_{}_off_peak_z'.format(response_dir)] >= analysis_params['rf_z_thr_abs'])]

    if len(subdf) > 0:

        save_f = h5py.File(os.path.join(save_folder, save_fn))

        nwb_fn = table_fn[0:-11] + '110_repacked.nwb'
        nwb_f = h5py.File(os.path.join(nwb_folder, nwb_fn), 'r')
        plane_n = table_fn[-11:-5]

        # S2
        s2_df = subdf[(subdf['rf_{}_on_peak_z'.format(response_dir)] >= analysis_params['rf_z_thr_abs']) &
                      (subdf['rf_{}_off_peak_z'.format(response_dir)] >= analysis_params['rf_z_thr_abs'])].reset_index()

        if len(s2_df) > 0:
            s2_grp = save_f.create_group(table_fn[0:-5] + '_{}_ONOFF'.format(response_dir))
            s1_on_grp = save_f.create_group(table_fn[0:-5] + '_{}_ON'.format(response_dir))
            s1_off_grp = save_f.create_group(table_fn[0:-5] + '_{}_OFF'.format(response_dir))

            for roi_i, roi_row in s2_df.iterrows():

                print('\t s2 receptive fields, {}, {} / {} ...'.format(roi_row['roi_n'], roi_i+1, len(s2_df)))

                if response_dir == 'pos':
                    _, _, _, srf_on, srf_off, _, _, _, _, _, _, _, _, \
                    _ = dt.get_everything_from_roi(nwb_f=nwb_f,
                                                   plane_n=plane_n,
                                                   roi_n=roi_row['roi_n'],
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
                    _ = dt.get_everything_from_roi(nwb_f=nwb_f,
                                                   plane_n=plane_n,
                                                   roi_n=roi_row['roi_n'],
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

            s1_on_grp_n = table_fn[0:-5] + '_{}_ON'.format(response_dir)

            if s1_on_grp_n in save_f.keys():
                s1_on_grp = save_f[s1_on_grp_n]
            else:
                s1_on_grp = save_f.create_group(s1_on_grp_n)

            for roi_i, roi_row in s1_on_df.iterrows():

                print('\t s1 ON receptive fields, {}, {} / {} ...'.format(roi_row['roi_n'], roi_i + 1, len(s1_on_df)))

                if response_dir == 'pos':
                    _, _, _, srf_on, _, _, _, _, _, _, _, _, _, \
                    _ = dt.get_everything_from_roi(nwb_f=nwb_f,
                                                   plane_n=plane_n,
                                                   roi_n=roi_row['roi_n'],
                                                   params=analysis_params)

                    _, rf_on_new = dt.get_rf_properties(srf=srf_on,
                                                        polarity='positive',
                                                        sigma=analysis_params['gaussian_filter_sigma_rf'],
                                                        interpolate_rate=analysis_params['interpolate_rate_rf'],
                                                        z_thr_abs=analysis_params['rf_z_thr_abs'],
                                                        z_thr_rel=analysis_params['rf_z_thr_rel'])
                elif response_dir == 'neg':
                    _, _, _, _, _, srf_on, _, _, _, _, _, _, _, \
                    _ = dt.get_everything_from_roi(nwb_f=nwb_f,
                                                   plane_n=plane_n,
                                                   roi_n=roi_row['roi_n'],
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

            s1_off_grp_n = table_fn[0:-5] + '_{}_OFF'.format(response_dir)

            if s1_off_grp_n in save_f.keys():
                s1_off_grp = save_f[s1_off_grp_n]
            else:
                s1_off_grp = save_f.create_group(s1_off_grp_n)

            for roi_i, roi_row in s1_off_df.iterrows():

                print('\t s1 OFF receptive fields, {}, {} / {} ...'.format(roi_row['roi_n'], roi_i + 1, len(s1_off_df)))

                if response_dir == 'pos':
                    _, _, _, _, srf_off, _, _, _, _, _, _, _, _, \
                    _ = dt.get_everything_from_roi(nwb_f=nwb_f,
                                                   plane_n=plane_n,
                                                   roi_n=roi_row['roi_n'],
                                                   params=analysis_params)

                    _, rf_off_new = dt.get_rf_properties(srf=srf_off,
                                                         polarity='positive',
                                                         sigma=analysis_params['gaussian_filter_sigma_rf'],
                                                         interpolate_rate=analysis_params['interpolate_rate_rf'],
                                                         z_thr_abs=analysis_params['rf_z_thr_abs'],
                                                         z_thr_rel=analysis_params['rf_z_thr_rel'])

                elif response_dir == 'neg':

                    _, _, _, _, _, _, srf_off, _, _, _, _, _, _, \
                    _ = dt.get_everything_from_roi(nwb_f=nwb_f,
                                                   plane_n=plane_n,
                                                   roi_n=roi_row['roi_n'],
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