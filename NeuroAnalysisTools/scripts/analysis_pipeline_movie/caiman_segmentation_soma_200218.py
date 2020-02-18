"""
This not a part of NeuroAnalysisTools
run it in an environment with caiman installed
in command line

for example:
>>> activate ciaman
"""

import os
import glob
import numpy as np
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
import h5py
from shutil import copyfile

date_recorded = '200210'
mouse_id = 'M504408'
resolution = (512, 512)
channel = 'green'
data_folder_n = '110_LSVDGCUC_reorged'
imaging_mode = '2p' # '2p' or 'deepscope'
n_prosess = 4


# =========================== caiman parameters for soma =================================================
# ============ sutter scope, zoom 2, 5 frames online average, 5 frames offline average ===================
# ============ deepscope, zoom 1.4, 2 frames online average, 3 frames offline average ====================
fr = 2  # frame rate (Hz)
decay_time = 0.5  # approximate length of transient event in seconds
gSig = (8, 8)  # expected half size of neurons
p = 1  # order of AR indicator dynamics
min_SNR = 1  # minimum SNR for accepting new components
rval_thr = 0.80  # correlation threshold for new component inclusion
ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
gnb = 2  # number of background components
gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int'))  # recompute gSig if downsampling is involved
mot_corr = False  # flag for online motion correction
pw_rigid = False  # flag for pw-rigid motion correction (slower but potentially more accurate)
max_shifts_online = np.ceil(10. / ds_factor).astype('int')  # maximum allowed shift during motion correction
sniper_mode = True  # flag using a CNN to detect new neurons (o/w space correlation is used)
init_batch = 200  # number of frames for initialization (presumably from the first file)
expected_comps = 500  # maximum number of expected components used for memory pre-allocation (exaggerate here)
dist_shape_update = True  # flag for updating shapes in a distributed way
min_num_trial = 10  # number of candidate components per frame
K = 2  # initial number of components
epochs = 2  # number of passes over the data
show_movie = False  # show the movie with the results as the data gets processed
# =========================== caiman parameters for soma =================================================

curr_folder = os.path.dirname(os.path.realpath(__file__))

c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=n_process, single_thread=False)

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\{}-{}-{}" \
              r"\{}".format(date_recorded, mouse_id, imaging_mode, data_folder_n)


plane_ns = [f for f in os.listdir(data_folder) if
            os.path.isdir(os.path.join(data_folder, f)) and
            f[:5] == 'plane']
plane_ns.sort()
print('planes:')
print('\n'.join(plane_ns))

for plane_n in plane_ns:

    print('\nsegmenting plane: {}'.format(plane_n))

    plane_folder = os.path.join(data_folder, plane_n, channel, 'corrected')
    os.chdir(plane_folder)

    fn = [f for f in os.listdir(plane_folder) if len(f) > 16 and f[-16:] == '_for_caiman.hdf5']
    if len(fn) > 1:
        print('\n'.join(fn))
        raise LookupError('more than one file found.')
    elif len(fn) == 0:
        raise LookupError('no file found.')
    else:
        fn = fn[0]

    fp = os.path.join(os.path.realpath(plane_folder), fn)

    params_dict = {'fnames': [fp],
                   'fr': fr,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': p,
                   'min_SNR': min_SNR,
                   'rval_thr': rval_thr,
                   'ds_factor': ds_factor,
                   'nb': gnb,
                   'motion_correct': mot_corr,
                   'init_batch': init_batch,
                   'init_method': 'bare',
                   'normalize': True,
                   'expected_comps': expected_comps,
                   'sniper_mode': sniper_mode,
                   'dist_shape_update': dist_shape_update,
                   'min_num_trial': min_num_trial,
                   'K': K,
                   'epochs': epochs,
                   'max_shifts_online': max_shifts_online,
                   'pw_rigid': pw_rigid,
                   'show_movie': show_movie}

    opts = cnmf.params.CNMFParams(params_dict=params_dict)

    cnm1 = cnmf.CNMF(n_process, params=opts, dview=dview)
    cnm1.fit_file(motion_correct=False)

    roi_num = cnm1.estimates.A.shape[1]
    print('saving ...')
    save_f = h5py.File('caiman_segmentation_results.hdf5', 'w')
    save_f.create_dataset('masks',
                          data=np.array(cnm1.estimates.A.todense()).T.reshape((roi_num, resolution[0], resolution[1]),
                                                                              order='F'), compression='lzf')
    save_f.create_dataset('traces', data=cnm1.estimates.C)
    save_f.close()

    copyfile(os.path.join(plane_folder, 'caiman_segmentation_results.hdf5'),
             os.path.join(curr_folder, plane_n, 'caiman_segmentation_results.hdf5'))

    # %% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)



