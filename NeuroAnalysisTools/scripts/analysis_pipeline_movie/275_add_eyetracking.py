import os
import numpy as np
import h5py
import corticalmapping.NwbTools as nt

diagonal_length = 9.0 # mm, the length of diagonal line of eyetracking field of view
side = 'right' # right eye or left eye
scope = 'deepscope' # 'sutter' or 'deepscope'

comments = 'small_x=temporal; big_x=nasal; small_y=dorsal; big_y=ventral'

if scope == 'sutter':
    eyetracking_ts_name = 'digital_vsync_right_eye_mon_rise'
    frame_shape = (658, 492)
elif scope == 'deepscope':
    eyetracking_ts_name = 'digital_cam_eye_rise'
    frame_shape = (640, 480)
else:
    raise LookupError('do not understand scope type.')

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

eye_folder = os.path.join(curr_folder, 'videomon')
fn = [f for f in os.listdir(eye_folder) if f[-12:] == '_output.hdf5']
if len(fn) != 1:
    print('could not find processed eyetracking data.')
else:
    fn = fn[0]

pixel_size = diagonal_length / np.sqrt(658. ** 2 + 492. ** 2)
print('eyetracking pixel size: {:5.3f} mm/pix'.format(pixel_size))

eye_file = h5py.File(os.path.join(eye_folder, fn), 'r')
led_pos = eye_file['led_positions'].value
pup_pos = eye_file['pupil_positions'].value
pup_shape = eye_file['pupil_shapes'].value
pup_shape[:, 0] = pup_shape[:, 0] * pixel_size
pup_shape[:, 1] = pup_shape[:, 1] * pixel_size
pup_shape_meta = 'format: {}; unit: [millimeter, millimeter, degree]'.format(eye_file['pupil_shapes'].attrs['format'])

if scope == 'sutter':
    pup_x = frame_shape[0] - pup_pos[:, 1]
elif scope == 'deepscope':
    pup_x = pup_pos[:, 1]
else:
    raise LookupError('do not understand scope type.')

pup_y = pup_pos[:,0]

pup_pos = (pup_pos - led_pos) * pixel_size
pup_area = pup_shape[:, 0] * pup_shape[:, 1] * np.pi * pixel_size * pixel_size

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = nt.RecordedFile(nwb_fn)
nwb_f.add_eyetracking_data(ts_path=eyetracking_ts_name,
                           pupil_x=pup_x,
                           pupil_y=pup_y,
                           pupil_area=pup_area,
                           module_name='eye_tracking',
                           unit='millimeter',
                           side=side,
                           comments=comments,
                           description='',
                           source="Jun's eyetracker with adaptive thresholding",
                           pupil_shape=pup_shape,
                           pupil_shape_meta=pup_shape_meta)
nwb_f.close()
