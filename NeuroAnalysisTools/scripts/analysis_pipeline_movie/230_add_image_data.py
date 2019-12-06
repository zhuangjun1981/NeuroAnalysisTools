import os
import h5py
import corticalmapping.NwbTools as nt

dset_ns = ['plane0'] # ['plane0', 'plane1', 'plane2']
imaging_depths = [150]
temporal_downsample_rate = 5 # down sample rate before motion correction times down sample rate after motion correction
scope = 'sutter' # 'sutter' or 'DeepScope'
zoom = 4

description = '2-photon imaging data'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

if scope == 'DeepScope':
    pixel_size = 0.0000009765 / zoom # meter
elif scope == 'sutter':
    pixel_size = 0.0000014 / zoom # meter
else:
    raise LookupError('do not understand scope type')

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = nt.RecordedFile(nwb_fn)

ts_2p_tot = nwb_f.file_pointer['/acquisition/timeseries/digital_vsync_2p_rise/timestamps'].value

# if scope == 'sutter':
#     ts_2p_tot = nwb_f.file_pointer['/acquisition/timeseries/digital_vsync_2p_rise/timestamps'].value
# elif scope == 'DeepScope':
#     ts_2p_tot = nwb_f.file_pointer['/acquisition/timeseries/digital_2p_vsync_rise/timestamps'].value
# else:
#     raise LookupError('do not understand scope type')
# print('total 2p timestamps count: {}'.format(len(ts_2p_tot)))

mov_fn = os.path.splitext(nwb_fn)[0] + '_2p_movies.hdf5'
mov_f = h5py.File(mov_fn, 'r')

for mov_i, mov_dn in enumerate(dset_ns):

    if mov_dn is not None:

        curr_dset = mov_f[mov_dn]
        if mov_dn is not None:
            mov_ts = ts_2p_tot[mov_i::len(dset_ns)]
            print('\n{}: total 2p timestamps count: {}'.format(mov_dn, len(mov_ts)))

            mov_ts_d = mov_ts[::temporal_downsample_rate]
            print('{}: downsampled 2p timestamps count: {}'.format(mov_dn, len(mov_ts_d)))
            print('{}: downsampled 2p movie frame num: {}'.format(mov_dn, curr_dset.shape[0]))

            # if len(mov_ts_d) == curr_dset.shape[0]:
            #     pass
            # elif len(mov_ts_d) == curr_dset.shape[0] + 1:
            #     mov_ts_d = mov_ts_d[0: -1]
            # else:
            #     raise ValueError('the timestamp count of {} movie ({}) does not equal (or is not greater by one) '
            #                      'the frame cound in the movie ({})'.format(mov_dn, len(mov_ts_d), curr_dset.shape[0]))
            mov_ts_d = mov_ts_d[:curr_dset.shape[0]]

            curr_description = '{}. Imaging depth: {} micron.'.format(description, imaging_depths[mov_i])
            nwb_f.add_acquired_image_series_as_remote_link('2p_movie_' + mov_dn, image_file_path=mov_fn,
                                                           dataset_path=mov_dn, timestamps=mov_ts_d,
                                                           description=curr_description, comments='',
                                                           data_format='zyx', pixel_size=[pixel_size, pixel_size],
                                                           pixel_size_unit='meter')

mov_f.close()
nwb_f.close()