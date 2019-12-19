import os
import io.StringIO as StringIO
import numpy as np
import h5py
import scipy.ndimage as ni
import matplotlib.pyplot as plt
from multiprocessing import Pool
import NeuroAnalysisTools.core.ImageAnalysis as ia
import NeuroAnalysisTools.core.PlottingTools as pt
import cv2
import PIL

plt.ioff()

chunk_size = 1000
process_num = 5
downsample_r = 10
frame_size = 8 # inch

def downsample_for_multiprocessing(params):

    nwb_path, dset_path, frame_start_i, frame_end_i,  dr = params

    print('\tdownsampling frame {} - {}'.format(frame_start_i, frame_end_i))

    ff = h5py.File(nwb_path, 'r')
    chunk = ff[dset_path][frame_start_i:frame_end_i, :, :]
    ff.close()
    chunk_d = ia.z_downsample(chunk, downSampleRate=dr, is_verbose=False)
    return chunk_d

def downsample_mov(nwb_path, dset_path, dr):

    ff = h5py.File(nwb_path, 'r')
    frame_num = ff[dset_path].shape[0]
    print('\tshape of movie: {}'.format(ff[dset_path].shape))
    chunk_starts = np.array(range(0, frame_num, chunk_size))
    chunk_ends = chunk_starts + chunk_size
    chunk_ends[-1] = frame_num

    params = []
    for i, chunk_start in enumerate(chunk_starts):
        params.append((nwb_path, dset_path, chunk_start, chunk_ends[i], dr))

    p = Pool(process_num)
    mov_d = p.map(downsample_for_multiprocessing, params)

    return np.concatenate(mov_d, axis=0)

def run():

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curr_folder)

    print('getting total mask ...')
    cell_f = h5py.File('cells_refined.hdf5', 'r')
    h, w = cell_f['cell0000']['roi'].attrs['dimension']
    total_mask = np.zeros((h, w), dtype=np.uint8)
    for cell_n, cell_grp in cell_f.items():
        curr_roi = ia.WeightedROI.from_h5_group(cell_grp['roi'])
        curr_mask = curr_roi.get_binary_mask()
        total_mask = np.logical_or(total_mask, curr_mask)
    cell_f.close()
    total_mask = ni.binary_dilation(total_mask, iterations=1)
    # plt.imshow(total_mask)
    # plt.title('total_mask')
    # plt.show()

    nwb_folder = os.path.dirname(curr_folder)
    nwb_fn = [f for f in os.listdir(nwb_folder) if f[-4:] == '.nwb'][0]
    nwb_path = os.path.join(nwb_folder, nwb_fn)

    plane_n = os.path.split(curr_folder)[1]
    dset_path = 'processing/motion_correction/MotionCorrection/{}/corrected/data'.format(plane_n)

    print('downsampling movie ...')
    print('\tnwb_path: {}'.format(nwb_path))
    print('\tdset_path: {}'.format(dset_path))

    nwb_f = h5py.File(nwb_path, 'r')
    dset = nwb_f[dset_path]
    print('\ttotal shape: {}'.format(dset.shape))
    nwb_f.close()

    mov_d = downsample_mov(nwb_path=nwb_path, dset_path=dset_path, dr=downsample_r)
    v_min = np.amin(mov_d)
    v_max = np.amax(mov_d)
    print('\tshape of downsampled movie: {}'.format(mov_d.shape))

    print('\n\tgenerating avi ...')

    if cv2.__version__[0:3] == '3.1':
        codex = 'XVID'
        fourcc = cv2.VideoWriter_fourcc(*codex)
        out = cv2.VideoWriter('marked_mov.avi', fourcc, 30, (frame_size * 100, frame_size * 100), isColor=True)
    elif cv2.__version__[0:6] == '2.4.11':
        out = cv2.VideoWriter('marked_mov.avi', -1, 30, (frame_size * 100, frame_size * 100), isColor=True)
    elif cv2.__version__[0:3] == '2.4':
        codex = 'XVID'
        fourcc = cv2.cv.CV_FOURCC(*codex)
        out = cv2.VideoWriter('marked_mov.avi', fourcc, 30, (frame_size * 100, frame_size * 100), isColor=True)
    else:
        raise EnvironmentError('Do not understand opencv cv2 version: {}.'.format(cv2.__version__))

    f = plt.figure(figsize=(frame_size, frame_size))
    for frame_i, frame in enumerate(mov_d):
        print('\tframe: {} / {}'.format(frame_i, mov_d.shape[0]))
        f.clear()
        ax = f.add_subplot(111)
        ax.imshow(frame, vmin=v_min, vmax=v_max*0.5, cmap='gray', interpolation='nearest')
        pt.plot_mask_borders(total_mask, plotAxis=ax, color='#ff0000', zoom=1, borderWidth=1)
        ax.set_aspect('equal')
        # plt.show()

        buffer_ = StringIO()
        pt.save_figure_without_borders(f, buffer_, dpi=100)
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)
        curr_frame = np.asarray(image)
        r, g, b, a = np.rollaxis(curr_frame, axis=-1)
        curr_frame = (np.dstack((b, g, r)))
        # print(r.dtype)
        # print(curr_frame.shape)

        out.write(curr_frame)

    out.release()
    cv2.destroyAllWindows()
    print('\t.avi movie generated.')

if __name__ == '__main__':
    run()
