import os
import stia.motion_correction as mc

date_recorded = '190503'
mouse_id = 'M439939'
data_folder_n = '110_LSNDGC_reorged'
imaging_mode = '2p' # '2p' or 'deepscope'

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\{}-{}-{}" \
              r"\{}".format(date_recorded, mouse_id, imaging_mode, data_folder_n)

def correct(data_folder):

    ref_ch_n = 'red'
    apply_ch_ns = ['green', 'red']

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curr_folder)

    ref_data_folder = os.path.join(data_folder, ref_ch_n)

    mc.motion_correction(input_folder=ref_data_folder,
                         input_path_identifier='.tif',
                         process_num=6,
                         output_folder=os.path.join(ref_data_folder, 'corrected'),
                         anchor_frame_ind_chunk=10,
                         anchor_frame_ind_projection=0,
                         iteration_chunk=10,
                         iteration_projection=10,
                         max_offset_chunk=(100., 100.),
                         max_offset_projection=(100., 100.),
                         align_func=mc.phase_correlation,
                         preprocessing_type=6,
                         fill_value=0.)

    offsets_path = os.path.join(ref_data_folder, 'corrected', 'correction_offsets.hdf5')
    ref_fns = [f for f in os.listdir(ref_data_folder) if f[-4:] == '.tif']
    ref_fns.sort()
    ref_paths = [os.path.join(ref_data_folder, f) for f in ref_fns]
    print('\nreference paths:')
    print('\n'.join(ref_paths))

    for apply_ch_i, apply_ch_n in enumerate(apply_ch_ns):
        apply_data_folder = os.path.join(data_folder, apply_ch_n)
        apply_fns = [f for f in os.listdir(apply_data_folder) if f[-4:] == '.tif']
        apply_fns.sort()
        apply_paths = [os.path.join(apply_data_folder, f) for f in apply_fns]
        print('\napply paths:')
        print('\n'.join(apply_paths))

        mc.apply_correction_offsets(offsets_path=offsets_path,
                                    path_pairs=zip(ref_paths, apply_paths),
                                    output_folder=os.path.join(apply_data_folder, 'corrected'),
                                    process_num=6,
                                    fill_value=0.,
                                    avi_downsample_rate=10,
                                    is_equalizing_histogram=False)

def run():

    plane_folders = [f for f in os.listdir(data_folder) if f[0:5] == 'plane' and
                     os.path.isdir(os.path.join(data_folder, f))]
    plane_folders.sort()
    print('folders to be corrected:')
    print('\n'.join(plane_folders))

    for plane_folder in plane_folders:
        correct(os.path.join(data_folder, plane_folder))

if __name__ == "__main__":
    run()