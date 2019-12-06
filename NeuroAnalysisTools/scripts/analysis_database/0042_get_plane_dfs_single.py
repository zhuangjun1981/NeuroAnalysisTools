import sys
sys.path.extend(['/home/junz/PycharmProjects/corticalmapping'])
import os
import time
import pandas as pd
import numpy as np
import h5py
import datetime
import corticalmapping.DatabaseTools as dt
from multiprocessing import Pool
from shutil import copyfile

nwb_fns = ['190510_M439939_110_repacked.nwb',
           '190523_M439939_110_repacked.nwb',
           '190524_M439939_110_repacked.nwb',
           '190509_M439943_110_repacked.nwb',
           '190521_M439943_110_repacked.nwb',
           '190523_M439943_110_repacked.nwb',]
database_folder = 'nwbs'
save_folder_n = "dataframes"
process_num = 6
is_overwrite = False

params = dt.ANALYSIS_PARAMS
params['trace_type'] = 'f_center_subtracted'
params['is_collapse_dire'] = False
params['is_collapse_sf'] = True
params['is_collapse_tf'] = True

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

    nwb_path, params, columns, save_folder, t0, nwb_i, nwb_f_num, is_overwrite = inputs
    nwb_fn = os.path.splitext(os.path.split(nwb_path)[1])[0]

    nwb_f = h5py.File(nwb_path, 'r')

    plane_ns = [k for k in nwb_f['processing'].keys() if k[0:16] == 'rois_and_traces_']
    plane_ns = [k[16:] for k in plane_ns]
    plane_ns.sort()
    # print('total plane number: {}'.format(len(plane_ns)))

    for plane_n in plane_ns:
        print('\tt: {:5.0f} minutes, processing {}, {} / {}, {} ...'.format((time.time() - t0) / 60.,
                                                                            nwb_fn,
                                                                            nwb_i + 1,
                                                                            nwb_f_num,
                                                                            plane_n))

        save_fn = '_'.join(nwb_fn.split('_')[0:2]) + '_' + plane_n + '.xlsx'
        save_path = os.path.join(save_folder, save_fn)
        if os.path.isfile(save_path):

            if is_overwrite: # overwrite existing xlsx files
                print('\t{}, file already exists. Overwirite.'.format(os.path.split(save_path)[1]))
                os.remove(save_path)


            else: # do not overwrite existing xlsx files
                print('\t{}, file already exists. Skip.'.format(os.path.split(save_path)[1]))
                return

        roi_ns = nwb_f['processing/rois_and_traces_{}/ImageSegmentation/imaging_plane/roi_list'.format(plane_n)].value
        roi_ns = [r.encode('utf-8') for r in roi_ns if r[0:4] == 'roi_']
        roi_ns.sort()

        df = pd.DataFrame(np.nan, index=range(len(roi_ns)), columns=columns)

        for roi_i, roi_n in enumerate(roi_ns):
            # print('\t\t\troi: {} / {}'.format(roi_i+1, len(roi_ns)))
            roi_properties, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
                dt.get_everything_from_roi(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n, params=params)
            for rp_name, rp_value in roi_properties.items():
                df.loc[roi_i, rp_name] = rp_value

        with pd.ExcelWriter(save_path, mode='w') as writer:
                df.to_excel(writer, sheet_name='sheet1')


def run():

    t0 = time.time()

    # nwb_fns = []
    # for fn in os.listdir(database_folder):
    #     if fn[-4:] == '.nwb' and date_range[0] <= int(fn[0:6]) <= date_range[1]:
    #         nwb_fns.append(fn)
    # nwb_fns.sort()
    print('\nnwb file list:')
    print('\n'.join(nwb_fns))

    date_str = datetime.datetime.now().strftime('%y%m%d%H%M%S')
    save_folder = os.path.join(curr_folder, '{}_{}'.format(save_folder_n, date_str))

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    copyfile(os.path.realpath(__file__), os.path.join(save_folder, 'script_log.py'))

    inputs_lst = [(os.path.join(curr_folder, database_folder, nwb_fn),
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
    # process_one_nwb_for_multi_thread(inputs_lst[0])

    # print('\nConcatenating indiviudal dataframes ...')
    # xlsx_fns = [f for f in os.listdir(os.path.join(curr_folder,save_folder)) if f[-5:] == '.xlsx']
    # xlsx_fns.sort()
    #
    # dfs = []
    # for xlsx_fn in xlsx_fns:
    #     curr_df = pd.read_excel(os.path.join(curr_folder, save_folder, xlsx_fn), sheetname='sheet1')
    #     # print(curr_df)
    #     dfs.append(curr_df)
    #
    # big_df = pd.concat(dfs, ignore_index=True)
    #
    # print('\nsaving ...')
    # date_str = datetime.datetime.now().strftime('%y%m%d%H%M%S')
    # save_path = os.path.join(curr_folder, 'big_roi_table_{}.xlsx'.format(date_str))
    #
    # if os.path.isfile(save_path):
    #     with pd.ExcelWriter(save_path, mode='a') as writer:
    #         big_df.to_excel(writer, sheet_name=params['trace_type'])
    # else:
    #     with pd.ExcelWriter(save_path, mode='w') as writer:
    #         big_df.to_excel(writer, sheet_name=params['trace_type'])

    print('\ndone!')


if __name__ == "__main__":
    run()
