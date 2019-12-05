import os
import numpy as np
import h5py
import time
import corticalmapping.core.ImageAnalysis as ia
import corticalmapping.core.PlottingTools as pt
import corticalmapping.core.FileTools as ft
import corticalmapping.NwbTools as nt
import matplotlib.pyplot as plt
from multiprocessing import Pool

CHUNK_SIZE = 2000
PROCESS_NUM = 5

def get_chunk_frames(frame_num, chunk_size):
    chunk_num = frame_num // chunk_size
    if frame_num % chunk_size > 0:
        chunk_num = chunk_num + 1

    print("total number of frames:", frame_num)
    print("total number of chunks:", chunk_num)

    chunk_ind = []
    chunk_starts = []
    chunk_ends = []

    for chunk_i in range(chunk_num):
        chunk_ind.append(chunk_i)
        chunk_starts.append(chunk_i * chunk_size)

        if chunk_i < chunk_num - 1:
            chunk_ends.append((chunk_i + 1) * chunk_size)
        else:
            chunk_ends.append(frame_num)

    return zip(chunk_ind, chunk_starts, chunk_ends)

def get_traces(params):
    t0 = time.time()

    chunk_ind, chunk_start, chunk_end, nwb_path, data_path, curr_folder, center_array, surround_array = params

    nwb_f = h5py.File(nwb_path, 'r')
    print('\nstart analyzing chunk: {}'.format(chunk_ind))
    curr_mov = nwb_f[data_path][chunk_start: chunk_end]
    nwb_f.close()

    # print 'extracting traces'
    curr_traces_center = np.empty((center_array.shape[0], curr_mov.shape[0]), dtype=np.float32)
    curr_traces_surround = np.empty((center_array.shape[0], curr_mov.shape[0]), dtype=np.float32)
    for i in range(center_array.shape[0]):
        curr_center = ia.WeightedROI(center_array[i])
        curr_surround = ia.ROI(surround_array[i])
        curr_traces_center[i, :] = curr_center.get_weighted_trace_pixelwise(curr_mov)

        # scale surround trace to be similar as center trace
        mean_center_weight = curr_center.get_mean_weight()
        curr_traces_surround[i, :] = curr_surround.get_binary_trace_pixelwise(curr_mov) * mean_center_weight

    # print 'saveing chunk {} ...'.format(chunk_ind)
    chunk_folder = os.path.join(curr_folder, 'chunks')
    if not os.path.isdir(chunk_folder):
        os.mkdir(chunk_folder)
    chunk_f = h5py.File(os.path.join(chunk_folder, 'chunk_temp_' + ft.int2str(chunk_ind, 4) + '.hdf5'))
    chunk_f['traces_center'] = curr_traces_center
    chunk_f['traces_surround'] = curr_traces_surround
    chunk_f.close()

    print('\n\t{:06d} seconds: chunk: {}; demixing finished.'.format(int(time.time() - t0), chunk_ind))

    return None

def run():

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curr_folder)

    plane_n = os.path.split(curr_folder)[1]
    print(plane_n)

    print('getting masks ...')
    rois_f = h5py.File('rois_and_traces.hdf5')
    center_array = rois_f['masks_center'].value
    surround_array = rois_f['masks_surround'].value

    print('\nanalyzing movie in chunks of size:', CHUNK_SIZE    , 'frames.')

    nwb_folder = os.path.dirname(curr_folder)
    nwb_fn = [f for f in os.listdir(nwb_folder) if f[-4:] == '.nwb'][0]
    nwb_path = os.path.join(nwb_folder, nwb_fn)
    print('\n' + nwb_path)
    data_path = '/processing/motion_correction/MotionCorrection/' + plane_n + '/corrected/data'

    nwb_f = h5py.File(nwb_path, 'r')
    total_frame = nwb_f[data_path].shape[0]
    nwb_f.close()

    chunk_frames = get_chunk_frames(total_frame, CHUNK_SIZE)
    chunk_params = [(cf[0], cf[1], cf[2], nwb_path, data_path,
                     curr_folder, center_array, surround_array) for cf in chunk_frames]

    p = Pool(PROCESS_NUM)
    p.map(get_traces, chunk_params)

    chunk_folder = os.path.join(curr_folder, 'chunks')
    chunk_fns = [f for f in os.listdir(chunk_folder) if f[0:11] == 'chunk_temp_']
    chunk_fns.sort()
    print('\nreading chunks files ...')
    print('\n'.join(chunk_fns))

    traces_raw = []
    traces_surround = []

    for chunk_fn in chunk_fns:
        curr_chunk_f = h5py.File(os.path.join(chunk_folder, chunk_fn))
        traces_raw.append(curr_chunk_f['traces_center'].value)
        traces_surround.append(curr_chunk_f['traces_surround'].value)

    print("saving ...")
    traces_raw = np.concatenate(traces_raw, axis=1)
    traces_surround = np.concatenate(traces_surround, axis=1)
    rois_f['traces_center_raw'] = traces_raw
    rois_f['traces_surround_raw'] = traces_surround
    print('done.')


if __name__ == '__main__':
    run()

