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

def run():

  date_recorded = '200306'
  mouse_id = 'M504408'
  resolution = (512, 512)
  channel = 'green'
  data_folder_n = '110_LSNDGCUC_reorged'
  imaging_mode = 'deepscope' # '2p' or 'deepscope'
  n_process = 6

  # ========================= caiman parameters for boutons ================================================
  # ============ sutter scope, zoom 4, 5 frames online average, 5 frames offline average ===================
  # fr = 2.  # frame rate (Hz)
  # decay_time = 0.5  # approximate length of transient event in seconds
  gSig = (5, 5)  # expected half size of neurons, (8, 8) for soma at zoom 2 on sutter scope
  p = 2  # order of AR indicator dynamics
  # min_SNR = 3  # minimum SNR for accepting new components
  # rval_thr = 0.80  # correlation threshold for new component inclusion
  # ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
  # gnb = 2  # number of background components
  # gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int'))  # recompute gSig if downsampling is involved
  mot_corr = False  # flag for online motion correction
  pw_rigid = False  # flag for pw-rigid motion correction (slower but potentially more accurate)
  # max_shifts_online = np.ceil(10. / ds_factor).astype('int')  # maximum allowed shift during motion correction
  # sniper_mode = True  # flag using a CNN to detect new neurons (o/w space correlation is used)
  # init_batch = 200  # number of frames for initialization (presumably from the first file)
  expected_comps = 500  # maximum number of expected components used for memory pre-allocation (exaggerate here)
  # dist_shape_update = True  # flag for updating shapes in a distributed way
  # min_num_trial = 10  # number of candidate components per frame
  K = 10  # initial number of components
  # epochs = 2  # number of passes over the data
  show_movie = False  # show the movie with the results as the data gets processed

  method_init = 'sparse_nmf'
  do_merge = False
  ssub = 1
  tsub = 1
  alpha_snmf = 10e1
  rolling_sum = False
  rf = 256
  p_ssub = 1
  p_tsub = 1
  Ain = None
  # method_deconvolution = 'oasis'
  border_pix = 0
  # ========================= caiman parameters for boutons ================================================
  curr_folder = os.path.dirname(os.path.realpath(__file__))

  data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\{}-{}-{}" \
                r"\{}".format(date_recorded, mouse_id, imaging_mode, data_folder_n)


  plane_ns = [f for f in os.listdir(data_folder) if
              os.path.isdir(os.path.join(data_folder, f)) and
              f[:5] == 'plane']
  plane_ns.sort()
  print('planes:')
  print('\n'.join(plane_ns))

  c, dview, _ = cm.cluster.setup_cluster(backend='local', n_processes=n_process, single_thread=False)

  for plane_n in plane_ns:

      print('\nsegmenting plane: {}'.format(plane_n))

      plane_folder = os.path.join(data_folder, plane_n, channel, 'corrected')
      os.chdir(plane_folder)

      fn = [f for f in os.listdir(plane_folder) if len(f) > 16 and f[-5:] == '.mmap']
      if len(fn) > 1:
          print('\n'.join(fn))
          raise LookupError('more than one file found.')
      elif len(fn) == 0:
          raise LookupError('no file found.')
      else:
          fn = fn[0]

      fp = os.path.join(os.path.realpath(plane_folder), fn)

      params_dict = {'fnames': [fp],
                     # 'fr': fr,
                     # 'decay_time': decay_time,
                     'gSig': gSig,
                     'p': p,
                     # 'min_SNR': min_SNR,
                     # 'rval_thr': rval_thr,
                     # 'ds_factor': ds_factor,
                     # 'nb': gnb,
                     'motion_correct': mot_corr,
                     # 'init_batch': init_batch,
                     # 'init_method': 'bare',
                     # 'normalize': True,
                     'expected_comps': expected_comps,
                     # 'sniper_mode': sniper_mode,
                     # 'dist_shape_update': dist_shape_update,
                     # 'min_num_trial': min_num_trial,
                     'K': K,
                     # 'epochs': epochs,
                     # 'max_shifts_online': max_shifts_online,
                     'pw_rigid': pw_rigid,
                     'show_movie': show_movie,

                     # testing parameters
                     'method_init': method_init,
                     'do_merge': do_merge,
                     'ssub': ssub,
                     'tsub': tsub,
                     'alpha_snmf': alpha_snmf,
                     'rolling_sum': rolling_sum,
                     'rf': rf,
                     'p_ssub': p_ssub,
                     'p_tsub': p_tsub,
                     # 'Ain': Ain,
                     # 'method_deconvolution': method_deconvolution,
                     'border_pix': border_pix
                      }

      opts = cnmf.params.CNMFParams(params_dict=params_dict)

      cnm1 = cnmf.CNMF(n_process, params=opts, dview=dview)
      cnm1.fit_file(motion_correct=False)

      roi_num = cnm1.estimates.A.shape[1]
      print('saving ...')
      save_f = h5py.File('caiman_segmentation_results.hdf5', 'a')
      save_f.create_dataset('masks',
                            data=np.array(cnm1.estimates.A.todense()).T.reshape((roi_num, resolution[0], resolution[1]),
                                                                                order='F'), compression='lzf')
      save_f.create_dataset('traces', data=cnm1.estimates.C)
      save_f.close()

      copyfile(os.path.join(plane_folder, 'caiman_segmentation_results.hdf5'),
               os.path.join(curr_folder, plane_n, 'caiman_segmentation_results.hdf5'))

      # %% STOP CLUSTER and clean up log files
      # cm.stop_server(dview=dview)
      log_files = glob.glob('*_LOG_*')
      for log_file in log_files:
          os.remove(log_file)


if __name__ == '__main__':
  run()

