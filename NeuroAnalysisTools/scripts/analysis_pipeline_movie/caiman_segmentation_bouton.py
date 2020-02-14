"""
This not a part of NeuroAnalysisTools
run it in an environment with caiman v1.0 installed
in command line

for example:
>>> activate ciaman
"""

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([r"E:\data\github_packages\CaImAn"])

import os
import numpy as np
import caiman as cm
import matplotlib.pyplot as plt
from caiman.source_extraction.cnmf import cnmf as cnmf
import h5py
from shutil import copyfile

def run():
    date_recorded = '190503'
    mouse_id = 'M439939'
    play_movie = False
    resolution = 512
    channel = 'green'

    data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\{}-{}-2p" \
                  r"\110_LSNDGC_reorged".format(date_recorded, mouse_id)

    curr_folder = os.path.dirname(os.path.realpath(__file__))

    plane_ns = [f for f in os.listdir(data_folder) if os.path.isdir(f) and f[:5] == 'plane']
    plane_ns.sort()
    print('planes:')
    print('\n'.join(plane_ns))

    # %% start cluster
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=6, single_thread=False)

    for plane_n in plane_ns:

        print('\nsegmenting plane: {}'.format(plane_n))

        plane_folder = os.path.join(data_folder, plane_n, channel, 'corrected')
        os.chdir(plane_folder)

        fn = [f for f in os.listdir(plane_folder) if f[-5:] == '.mmap']
        if len(fn) > 1:
            print('\n'.join(fn))
            raise LookupError('more than one file found.')
        elif len(fn) == 0:
            raise LookupError('no file found.')
        else:
            fn = fn[0]

        fn_parts = fn.split('_')
        d1 = int(fn_parts[fn_parts.index('d1') + 1]) # column, x
        d2 = int(fn_parts[fn_parts.index('d2') + 1]) # row, y
        d3 = int(fn_parts[fn_parts.index('d3') + 1]) # channel
        d4 = int(fn_parts[fn_parts.index('frames') + 1]) # frame, T
        order = fn_parts[fn_parts.index('order') + 1]

        print('playing {} ...'.format(fn))

        mov = np.memmap(filename=fn, shape=(d1, d2, d4), order=order, dtype=np.float32, mode='r')
        mov = mov.transpose((2, 1, 0))

        print('shape of joined movie: {}.'.format(mov.shape))

        #%% play movie, press q to quit
        if play_movie:
            cm.movie(mov).play(fr=50,magnification=1,gain=2.)

        #%% movie cannot be negative!
        mov_min = float(np.amin(mov))
        print('minimum pixel value: {}.'.format(mov_min))
        if mov_min < 0:
            raise Exception('Movie too negative, add_to_movie should be larger')

        #%% correlation image. From here infer neuron size and density
        Cn = cm.movie(mov).local_correlations(swap_dim=False)
        # plt.imshow(Cn, cmap='gray')
        # plt.show()

        K = 100  # number of neurons expected per patch
        gSig = [5, 5]  # expected half size of neurons
        merge_thresh = 0.9  # merging threshold, max correlation allowed
        p = 2  # order of the autoregressive system
        cnm = cnmf.CNMF(n_processes,
                        k=10, # number of neurons expected per patch
                        gSig=[5, 5] , # expected half size of neurons
                        merge_thresh=0.9,  # merging threshold, max correlation allowed
                        p=2, # order of the autoregressive system
                        dview=dview,
                        Ain=None,
                        method_deconvolution='oasis',
                        rolling_sum = False,
                        method_init='sparse_nmf',
                        alpha_snmf=10e1,
                        ssub=1,
                        tsub=1,
                        p_ssub=1,
                        p_tsub=1,
                        rf=int(resolution / 2), # half-size of the patches in pixels
                        border_pix=0,
                        do_merge=False)
        cnm = cnm.fit(mov)
        A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
        #%%
        # crd = cm.utils.visualization.plot_contours(cnm.A, Cn)
        # plt.show()
        # input("Press enter to continue ...")

        roi_num = cnm.A.shape[1]
        save_fn = h5py.File('caiman_segmentation_results.hdf5')
        bias = save_fn['bias_added_to_movie'].value
        save_fn['masks'] = np.array(cnm.A.todense()).T.reshape((roi_num, 512, 512), order='F')
        save_fn['traces'] = cnm.C - bias
        save_fn.close()

        copyfile(os.path.join(plane_folder, 'caiman_segmentation_results.hdf5'),
                 os.path.join(curr_folder, plane_n, 'caiman_segmentation_results.hdf5'))

        plt.close('all')


if __name__ == '__main__':
    run()