import os
import NeuroAnalysisTools.DatabaseTools as dt
import time
import pandas as pd
import numpy as np
import h5py
from multiprocessing import Pool
import shutil

date_range = [180301, 190601]
nwb_folder = 'nwbs'
df_folder = r'other_dataframes\dataframes_190530171338'
clu_folder = r'intermediate_results\bouton_clustering\AllStimuli_DistanceThr_1.30'
process_num = 6
is_overwrite = True

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

print('pandas version: {}\n'.format(pd.__version__))

columns = [
    'date',
    'mouse_id',
    'plane_n',
    'roi_n',
    'depth',  # microns under pia, float

    # roi mask
    'roi_area',  # square micron
    'roi_center_row',  # center of roi mask in field of view, row
    'roi_center_col',  # center of roi mask in field of view, column

    # trace skewness
    'skew_raw',  # skewness of unfiltered trace (neuropil subtracted), float
    'skew_fil',  # skewness of highpassed trace, float

    # receptive fields
    'rf_pos_on_peak_z',
    'rf_pos_on_area',
    'rf_pos_on_center_alt',
    'rf_pos_on_center_azi',

    'rf_pos_off_peak_z',
    'rf_pos_off_area',
    'rf_pos_off_center_alt',
    'rf_pos_off_center_azi',

    'rf_pos_onoff_peak_z',
    'rf_pos_onoff_area',
    'rf_pos_onoff_center_alt',
    'rf_pos_onoff_center_azi',

    'rf_pos_lsi',

    'rf_neg_on_peak_z',
    'rf_neg_on_area',
    'rf_neg_on_center_alt',
    'rf_neg_on_center_azi',

    'rf_neg_off_peak_z',
    'rf_neg_off_area',
    'rf_neg_off_center_alt',
    'rf_neg_off_center_azi',

    'rf_neg_onoff_peak_z',
    'rf_neg_onoff_area',
    'rf_neg_onoff_center_alt',
    'rf_neg_onoff_center_azi',

    'rf_neg_lsi',

    # drifting grating peak response
    'dgc_pos_peak_df',
    'dgc_neg_peak_df',
    'dgc_pos_p_ttest_df',
    'dgc_neg_p_ttest_df',
    'dgc_p_anova_df',

    'dgc_pos_peak_dff',
    'dgc_neg_peak_dff',
    'dgc_pos_p_ttest_dff',
    'dgc_neg_p_ttest_dff',
    'dgc_p_anova_dff',

    'dgc_pos_peak_z',
    'dgc_neg_peak_z',
    'dgc_pos_p_ttest_z',
    'dgc_neg_p_ttest_z',
    'dgc_p_anova_z',

    # direction / orientation tuning, pos, df
    'dgc_pos_osi_raw_df',
    'dgc_pos_dsi_raw_df',
    'dgc_pos_gosi_raw_df',
    'dgc_pos_gdsi_raw_df',
    'dgc_pos_osi_ele_df',
    'dgc_pos_dsi_ele_df',
    'dgc_pos_gosi_ele_df',
    'dgc_pos_gdsi_ele_df',
    'dgc_pos_osi_rec_df',
    'dgc_pos_dsi_rec_df',
    'dgc_pos_gosi_rec_df',
    'dgc_pos_gdsi_rec_df',
    'dgc_pos_peak_dire_raw_df',
    'dgc_pos_vs_dire_raw_df',
    'dgc_pos_vs_dire_ele_df',
    'dgc_pos_vs_dire_rec_df',

    # direction / orientation tuning, neg, df
    'dgc_neg_osi_raw_df',
    'dgc_neg_dsi_raw_df',
    'dgc_neg_gosi_raw_df',
    'dgc_neg_gdsi_raw_df',
    'dgc_neg_osi_ele_df',
    'dgc_neg_dsi_ele_df',
    'dgc_neg_gosi_ele_df',
    'dgc_neg_gdsi_ele_df',
    'dgc_neg_osi_rec_df',
    'dgc_neg_dsi_rec_df',
    'dgc_neg_gosi_rec_df',
    'dgc_neg_gdsi_rec_df',
    'dgc_neg_peak_dire_raw_df',
    'dgc_neg_vs_dire_raw_df',
    'dgc_neg_vs_dire_ele_df',
    'dgc_neg_vs_dire_rec_df',

    # direction / orientation tuning, pos, dff
    'dgc_pos_osi_raw_dff',
    'dgc_pos_dsi_raw_dff',
    'dgc_pos_gosi_raw_dff',
    'dgc_pos_gdsi_raw_dff',
    'dgc_pos_osi_ele_dff',
    'dgc_pos_dsi_ele_dff',
    'dgc_pos_gosi_ele_dff',
    'dgc_pos_gdsi_ele_dff',
    'dgc_pos_osi_rec_dff',
    'dgc_pos_dsi_rec_dff',
    'dgc_pos_gosi_rec_dff',
    'dgc_pos_gdsi_rec_dff',
    'dgc_pos_peak_dire_raw_dff',
    'dgc_pos_vs_dire_raw_dff',
    'dgc_pos_vs_dire_ele_dff',
    'dgc_pos_vs_dire_rec_dff',

    # direction / orientation tuning, neg, dff
    'dgc_neg_osi_raw_dff',
    'dgc_neg_dsi_raw_dff',
    'dgc_neg_gosi_raw_dff',
    'dgc_neg_gdsi_raw_dff',
    'dgc_neg_osi_ele_dff',
    'dgc_neg_dsi_ele_dff',
    'dgc_neg_gosi_ele_dff',
    'dgc_neg_gdsi_ele_dff',
    'dgc_neg_osi_rec_dff',
    'dgc_neg_dsi_rec_dff',
    'dgc_neg_gosi_rec_dff',
    'dgc_neg_gdsi_rec_dff',
    'dgc_neg_peak_dire_raw_dff',
    'dgc_neg_vs_dire_raw_dff',
    'dgc_neg_vs_dire_ele_dff',
    'dgc_neg_vs_dire_rec_dff',

    # direction / orientation tuning, pos, zscore
    'dgc_pos_osi_raw_z',
    'dgc_pos_dsi_raw_z',
    'dgc_pos_gosi_raw_z',
    'dgc_pos_gdsi_raw_z',
    'dgc_pos_osi_ele_z',
    'dgc_pos_dsi_ele_z',
    'dgc_pos_gosi_ele_z',
    'dgc_pos_gdsi_ele_z',
    'dgc_pos_osi_rec_z',
    'dgc_pos_dsi_rec_z',
    'dgc_pos_gosi_rec_z',
    'dgc_pos_gdsi_rec_z',
    'dgc_pos_peak_dire_raw_z',
    'dgc_pos_vs_dire_raw_z',
    'dgc_pos_vs_dire_ele_z',
    'dgc_pos_vs_dire_rec_z',

    # direction / orientation tuning, neg, zscore
    'dgc_neg_osi_raw_z',
    'dgc_neg_dsi_raw_z',
    'dgc_neg_gosi_raw_z',
    'dgc_neg_gdsi_raw_z',
    'dgc_neg_osi_ele_z',
    'dgc_neg_dsi_ele_z',
    'dgc_neg_gosi_ele_z',
    'dgc_neg_gdsi_ele_z',
    'dgc_neg_osi_rec_z',
    'dgc_neg_dsi_rec_z',
    'dgc_neg_gosi_rec_z',
    'dgc_neg_gdsi_rec_z',
    'dgc_neg_peak_dire_raw_z',
    'dgc_neg_vs_dire_raw_z',
    'dgc_neg_vs_dire_ele_z',
    'dgc_neg_vs_dire_rec_z',

    # sf tuning, pos, df
    'dgc_pos_peak_sf_raw_df',
    'dgc_pos_weighted_sf_raw_df',
    'dgc_pos_weighted_sf_log_raw_df',
    'dgc_pos_weighted_sf_ele_df',
    'dgc_pos_weighted_sf_log_ele_df',
    'dgc_pos_weighted_sf_rec_df',
    'dgc_pos_weighted_sf_log_rec_df',

    # sf tuning, neg, df
    'dgc_neg_peak_sf_raw_df',
    'dgc_neg_weighted_sf_raw_df',
    'dgc_neg_weighted_sf_log_raw_df',
    'dgc_neg_weighted_sf_ele_df',
    'dgc_neg_weighted_sf_log_ele_df',
    'dgc_neg_weighted_sf_rec_df',
    'dgc_neg_weighted_sf_log_rec_df',

    # sf tuning, pos, dff
    'dgc_pos_peak_sf_raw_dff',
    'dgc_pos_weighted_sf_raw_dff',
    'dgc_pos_weighted_sf_log_raw_dff',
    'dgc_pos_weighted_sf_ele_dff',
    'dgc_pos_weighted_sf_log_ele_dff',
    'dgc_pos_weighted_sf_rec_dff',
    'dgc_pos_weighted_sf_log_rec_dff',

    # sf tuning, neg, dff
    'dgc_neg_peak_sf_raw_dff',
    'dgc_neg_weighted_sf_raw_dff',
    'dgc_neg_weighted_sf_log_raw_dff',
    'dgc_neg_weighted_sf_ele_dff',
    'dgc_neg_weighted_sf_log_ele_dff',
    'dgc_neg_weighted_sf_rec_dff',
    'dgc_neg_weighted_sf_log_rec_dff',

    # sf tuning, pos, zscore
    'dgc_pos_peak_sf_raw_z',
    'dgc_pos_weighted_sf_raw_z',
    'dgc_pos_weighted_sf_log_raw_z',
    'dgc_pos_weighted_sf_ele_z',
    'dgc_pos_weighted_sf_log_ele_z',
    'dgc_pos_weighted_sf_rec_z',
    'dgc_pos_weighted_sf_log_rec_z',

    # sf tuning, neg, zscore
    'dgc_neg_peak_sf_raw_z',
    'dgc_neg_weighted_sf_raw_z',
    'dgc_neg_weighted_sf_log_raw_z',
    'dgc_neg_weighted_sf_ele_z',
    'dgc_neg_weighted_sf_log_ele_z',
    'dgc_neg_weighted_sf_rec_z',
    'dgc_neg_weighted_sf_log_rec_z',

    # tf tuning, pos, df
    'dgc_pos_peak_tf_raw_df',
    'dgc_pos_weighted_tf_raw_df',
    'dgc_pos_weighted_tf_log_raw_df',
    'dgc_pos_weighted_tf_ele_df',
    'dgc_pos_weighted_tf_log_ele_df',
    'dgc_pos_weighted_tf_rec_df',
    'dgc_pos_weighted_tf_log_rec_df',

    # tf tuning, neg, df
    'dgc_neg_peak_tf_raw_df',
    'dgc_neg_weighted_tf_raw_df',
    'dgc_neg_weighted_tf_log_raw_df',
    'dgc_neg_weighted_tf_ele_df',
    'dgc_neg_weighted_tf_log_ele_df',
    'dgc_neg_weighted_tf_rec_df',
    'dgc_neg_weighted_tf_log_rec_df',

    # tf tuning, pos, dff
    'dgc_pos_peak_tf_raw_dff',
    'dgc_pos_weighted_tf_raw_dff',
    'dgc_pos_weighted_tf_log_raw_dff',
    'dgc_pos_weighted_tf_ele_dff',
    'dgc_pos_weighted_tf_log_ele_dff',
    'dgc_pos_weighted_tf_rec_dff',
    'dgc_pos_weighted_tf_log_rec_dff',

    # tf tuning, neg, dff
    'dgc_neg_peak_tf_raw_dff',
    'dgc_neg_weighted_tf_raw_dff',
    'dgc_neg_weighted_tf_log_raw_dff',
    'dgc_neg_weighted_tf_ele_dff',
    'dgc_neg_weighted_tf_log_ele_dff',
    'dgc_neg_weighted_tf_rec_dff',
    'dgc_neg_weighted_tf_log_rec_dff',

    # tf tuning, pos, zscore
    'dgc_pos_peak_tf_raw_z',
    'dgc_pos_weighted_tf_raw_z',
    'dgc_pos_weighted_tf_log_raw_z',
    'dgc_pos_weighted_tf_ele_z',
    'dgc_pos_weighted_tf_log_ele_z',
    'dgc_pos_weighted_tf_rec_z',
    'dgc_pos_weighted_tf_log_rec_z',

    # tf tuning, neg, zscore
    'dgc_neg_peak_tf_raw_z',
    'dgc_neg_weighted_tf_raw_z',
    'dgc_neg_weighted_tf_log_raw_z',
    'dgc_neg_weighted_tf_ele_z',
    'dgc_neg_weighted_tf_log_ele_z',
    'dgc_neg_weighted_tf_rec_z',
    'dgc_neg_weighted_tf_log_rec_z',
]

def process_one_nwb_for_multi_thread(inputs):

    nwb_path, df_folder, clu_folder, params, columns, save_folder, t0, nwb_i, nwb_f_num, is_overwrite = inputs

    nwb_fn = os.path.splitext(os.path.split(nwb_path)[1])[0]

    date, mid, _, _ = nwb_fn.split('_')

    nwb_f = h5py.File(nwb_path, 'r')
    plane_ns = dt.get_plane_ns(nwb_f=nwb_f)
    plane_ns.sort()

    for plane_n in plane_ns:
        print('\tt: {:5.0f} minutes, processing {}, {} / {}, {} ...'.format((time.time() - t0) / 60.,
                                                                             nwb_fn,
                                                                             nwb_i + 1,
                                                                             nwb_f_num,
                                                                             plane_n))

        roi_df_fn = '{}_{}_{}.csv'.format(date, mid, plane_n)
        roi_df = pd.read_csv(os.path.join(df_folder, roi_df_fn))

        clu_fn = '{}_{}_{}_axon_grouping.hdf5'.format(date, mid, plane_n)
        clu_f = h5py.File(os.path.join(clu_folder, clu_fn), 'r')

        axon_ns = clu_f['axons'].keys()
        axon_ns.sort()

        axon_df = pd.DataFrame(np.nan, index=range(len(axon_ns)), columns=columns)

        for axon_i, axon_n in enumerate(axon_ns):

            roi_lst = clu_f['axons/{}'.format(axon_n)].value

            if len(roi_lst) == 1:
                curr_roi_df = roi_df[roi_df['roi_n'] == roi_lst[0]].reset_index()
                for col in columns:
                    axon_df.loc[axon_i, col] = curr_roi_df.loc[0, col]
                axon_df.loc[axon_i, 'roi_n'] = axon_n
            else:
                axon_properties, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
                                dt.get_everything_from_axon(nwb_f=nwb_f,
                                                            clu_f=clu_f,
                                                            plane_n=plane_n,
                                                            axon_n=axon_n,
                                                            params=params,
                                                            verbose=False)
                for rp_name, rp_value in axon_properties.items():
                        axon_df.loc[axon_i, rp_name] = rp_value

        save_path = os.path.join(save_folder, '{}_{}_{}.csv'.format(date, mid, plane_n))

        if os.path.isfile(save_path):
            if is_overwrite:
                os.remove(save_path)
                axon_df.to_csv(save_path)
            else:
                raise IOError('Axon dataframe file already exists. \npath: {}'.format(save_path))
        else:
            axon_df.to_csv(save_path)

def run():

    t0 = time.time()

    with open(os.path.join(df_folder, 'script_log.py')) as script_f:
        script = script_f.readlines()

    for line in script:
        if line[0:6] == 'params':
            exec(line)

    nwb_fns = []
    for fn in os.listdir(os.path.realpath(nwb_folder)):
        if fn[-4:] == '.nwb' and date_range[0] <= int(fn[0:6]) <= date_range[1]:
            nwb_fns.append(fn)
    nwb_fns.sort()
    print('\nnwb file list:')
    print('\n'.join(nwb_fns))

    save_folder = df_folder + '_axon_' + os.path.split(clu_folder)[1]

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    shutil.copyfile(os.path.realpath(__file__), os.path.join(save_folder, 'script_log.py'))

    inputs_lst = [(os.path.join(curr_folder, nwb_folder, nwb_fn),
                   os.path.realpath(df_folder),
                   os.path.realpath(clu_folder),
                   params,
                   columns,
                   save_folder,
                   t0,
                   nwb_i,
                   len(nwb_fns),
                   is_overwrite) for nwb_i, nwb_fn in enumerate(nwb_fns)]

    print('\nprocessing individual nwb files ...')
    p = Pool(process_num)
    p.map(process_one_nwb_for_multi_thread, inputs_lst)


if __name__ == '__main__':
    run()