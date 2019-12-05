import os
import h5py
import numpy as np
import skimage.external.tifffile as tf

date_recorded = '190503'
mouse_id = 'M439939'
sess_id = '110'
channel = 'green'
data_folder_n = '110_LSNDGC_reorged'
imaging_mode = '2p' # '2p' or 'deepscope'
identifier = '110_LSNDGC'

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\{}-{}-{}" \
              r"\{}".format(date_recorded, mouse_id, imaging_mode, data_folder_n)


curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

file_prefix = '{}_{}_{}'.format(date_recorded, mouse_id, sess_id)

plane_fns = [f for f in os.listdir(data_folder) if f[:5] == 'plane']
plane_fns.sort()
print('\n'.join(plane_fns))

data_f = h5py.File(file_prefix + '_2p_movies.hdf5')

for plane_fn in plane_fns:
    print('\nprocessing {} ...'.format(plane_fn))
    plane_folder = os.path.join(data_folder, plane_fn, channel, 'corrected')
    # mov_fns = [f for f in os.listdir(plane_folder) if f[-14:] == '_corrected.tif']
    mov_fns = [f for f in os.listdir(plane_folder) if f[-4:] == '.tif' and identifier in f]
    mov_fns.sort()
    print('\n'.join(mov_fns))

    # get shape of concatenated movie
    z1, y, x = tf.imread(os.path.join(plane_folder, mov_fns[0])).shape
    z0, _, _ = tf.imread(os.path.join(plane_folder, mov_fns[-1])).shape
    z = z0 + z1 * (len(mov_fns) - 1)

    # for mov_fn in mov_fns:
    #     print('reading {} ...'.format(mov_fn))
    #     curr_z, curr_y, curr_x = tf.imread(os.path.join(plane_folder, mov_fn)).shape
    #
    #     if y is None:
    #         y = curr_y
    #     else:
    #         if y != curr_y:
    #             raise ValueError('y dimension ({}) of file "{}" does not agree with previous file(s) ({}).'
    #                              .format(curr_y, mov_fn, y))
    #
    #     if x is None:
    #         x = curr_x
    #     else:
    #         if x != curr_x:
    #             raise ValueError('x dimension ({}) of file "{}" does not agree with previous file(s) ({}).'
    #                              .format(curr_x, mov_fn, x))
    #
    #     z = z + curr_z

    print((z,y,x))
    dset = data_f.create_dataset(plane_fn, (z, y, x), dtype=np.int16, compression='lzf')

    start_frame = 0
    end_frame = 0
    for mov_fn in mov_fns:
        print('reading {} ...'.format(mov_fn))
        curr_mov = tf.imread(os.path.join(plane_folder, mov_fn))
        end_frame = start_frame + curr_mov.shape[0]
        dset[start_frame : end_frame] = curr_mov
        start_frame = end_frame

    dset.attrs['conversion'] = 1.
    dset.attrs['resolution'] = 1.
    dset.attrs['unit'] = 'arbiturary_unit'

data_f.close()