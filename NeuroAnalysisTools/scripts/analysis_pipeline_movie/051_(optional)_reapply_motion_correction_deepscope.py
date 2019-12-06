import os
import stia.motion_correction as mc

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\190118-M417949-deepscope\movie\110_LSNDGC"

# apply the correction offsets from one plane to other planes
reference_plane = 'plane1'
apply_plane_ns = ['plane0', 'plane2']
ref_ch_n = 'red'
apply_ch_ns = ['green', 'red']

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

def run():
    ref_folder = os.path.join(data_folder, reference_plane, ref_ch_n)
    offsets_path = os.path.join(ref_folder, 'corrected', 'correction_offsets.hdf5')
    ref_paths = [f for f in os.listdir(ref_folder) if f[-4:] == '.tif']
    ref_paths.sort()
    ref_paths = [os.path.join(ref_folder, f) for f in ref_paths]
    print('\nreference paths:')
    print('\n'.join(ref_paths))


    for apply_plane_n in apply_plane_ns:
        for apply_ch_n in apply_ch_ns:
            print('\n\tapply to {}, channel: {}'.format(apply_plane_n, apply_ch_n))
            working_folder = os.path.join(data_folder, apply_plane_n, apply_ch_n)
            apply_paths = [f for f in os.listdir(working_folder) if f[-4:] == '.tif']
            apply_paths.sort()
            apply_paths = [os.path.join(working_folder, f) for f in apply_paths]
            print('\n\tapply paths:')
            print('\t'+'\n\t'.join(apply_paths))

            mc.apply_correction_offsets(offsets_path=offsets_path,
                                        path_pairs=zip(ref_paths, apply_paths),
                                        output_folder=os.path.join(working_folder, 'corrected'),
                                        process_num=6,
                                        fill_value=0.,
                                        avi_downsample_rate=10,
                                        is_equalizing_histogram=False)

if __name__ == "__main__":
    run()