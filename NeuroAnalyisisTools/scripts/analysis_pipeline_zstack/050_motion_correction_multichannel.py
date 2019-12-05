import os
import numpy as np
import h5py
import tifffile as tf
import stia.motion_correction as mc
from warnings import warn
from multiprocessing import Pool

def run():
    data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
                  r"\180816-M376019-zstack\zstack_2p_zoom2"
    ref_ch_n = 'red'
    n_process = 8

    anchor_frame_ind_chunk = 10
    iteration_chunk = 10
    max_offset_chunk = (50., 50.)
    preprocessing_type = 0
    fill_value = 0.

    is_apply = True
    avi_downsample_rate = None
    is_equalizing_histogram = False

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curr_folder)

    ref_data_folder = os.path.join(data_folder, ref_ch_n)

    steps = [f for f in os.listdir(ref_data_folder) if os.path.isdir(os.path.join(ref_data_folder, f))
             and f[0:5] == 'step_']
    steps.sort()
    print('\n'.join(steps))

    params = []
    for step in steps:

        folder_ref = os.path.join(data_folder, ref_ch_n, step)
        params.append((folder_ref, anchor_frame_ind_chunk, iteration_chunk, max_offset_chunk, preprocessing_type,
                       fill_value, is_apply, avi_downsample_rate, is_equalizing_histogram))

    chunk_p = Pool(n_process)
    chunk_p.map(correct_single_step, params)


def correct_single_step(param):

    folder_ref, anchor_frame_ind_chunk, iteration_chunk, max_offset_chunk, preprocessing_type, fill_value,\
        is_apply, avi_downsample_rate, is_equalizing_histogram= param

    step_n = os.path.split(folder_ref)[1]
    print('\nStart correcting step {} ...'.format(step_n))

    mov_ref_n = [f for f in os.listdir(folder_ref) if f[-4:] == '.tif' and step_n in f]
    if len(mov_ref_n) != 1:
        warn('step {}: number of green movie does not equal 1.'.format(step_n))
        return

    mov_paths, _ = mc.motion_correction(input_folder=folder_ref,
                                        input_path_identifier='.tif',
                                        process_num=1,
                                        output_folder=folder_ref,
                                        anchor_frame_ind_chunk=anchor_frame_ind_chunk,
                                        anchor_frame_ind_projection=0,
                                        iteration_chunk=iteration_chunk,
                                        iteration_projection=10,
                                        max_offset_chunk=max_offset_chunk,
                                        max_offset_projection=(30., 30.),
                                        align_func=mc.phase_correlation,
                                        preprocessing_type=preprocessing_type,
                                        fill_value=fill_value)

    if is_apply:

        offsets_path = os.path.join(folder_ref, 'correction_offsets.hdf5')
        offsets_f = h5py.File(offsets_path)
        ref_path = offsets_f['file_0000'].attrs['path']
        offsets_f.close()

        movie_path = mov_paths[0]

        mc.apply_correction_offsets(offsets_path=offsets_path,
                                    path_pairs=[[ref_path, movie_path]],
                                    output_folder=folder_ref,
                                    process_num=1,
                                    fill_value=fill_value,
                                    avi_downsample_rate=avi_downsample_rate,
                                    is_equalizing_histogram=is_equalizing_histogram)



if __name__ == "__main__":
    run()





