import sys
import os
import h5py
import numpy as np
import NeuroAnalysisTools.HighLevel as hl
import NeuroAnalysisTools.core.FileTools as ft
import matplotlib.pyplot as plt
from multiprocessing import Pool


lam = 0.05
plot_chunk_size = 5000
process_num = 5


def plot_traces_chunks(traces, labels, chunk_size, roi_ind):
    """

    :param traces: np.array, shape=[trace_type, t_num]
    :param labels:
    :param chunk_size:
    :param figures_folder:
    :param roi_ind:
    :return:
    """

    t_num = traces.shape[1]
    chunk_num = t_num // chunk_size

    chunks = []
    for chunk_ind in range(chunk_num):
        chunks.append([chunk_ind * chunk_size, (chunk_ind + 1) * chunk_size])

    if t_num % chunk_size != 0:
        chunks.append([chunk_num * chunk_size, t_num])

    v_max = np.amax(traces)
    v_min = np.amin(traces)

    fig = plt.figure(figsize=(75, 20))
    fig.suptitle('neuropil subtraction for ROI: {}'.format(roi_ind))
    for chunk_ind, chunk in enumerate(chunks):
        curr_ax = fig.add_subplot(len(chunks), 1, chunk_ind + 1)
        for trace_ind in range(traces.shape[0]):
           curr_ax.plot(traces[trace_ind, chunk[0]: chunk[1]], label=labels[trace_ind])

        curr_ax.set_xlim([0, chunk_size])
        curr_ax.set_ylim([v_min, v_max * 1.2])
        curr_ax.legend()

    return fig

def plot_traces_for_multi_process(params):

    curr_traces, plot_chunk_size, roi_ind, figures_folder = params

    print('roi_{:04d}'.format(roi_ind))

    curr_fig = plot_traces_chunks(traces=curr_traces,
                                  labels=['center', 'surround', 'subtracted'],
                                  chunk_size=plot_chunk_size,
                                  roi_ind=roi_ind)
    curr_fig.savefig(os.path.join(figures_folder, 'neuropil_subtraction_ROI_{:04d}.png'.format(roi_ind)))
    curr_fig.clear()
    plt.close(curr_fig)

def run():
    curr_folder = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curr_folder)

    data_f = h5py.File('rois_and_traces.hdf5')
    traces_raw = data_f['traces_center_raw'].value
    traces_srround = data_f['traces_surround_raw'].value

    traces_subtracted = np.zeros(traces_raw.shape, np.float32)
    ratio = np.zeros(traces_raw.shape[0], np.float32)
    err = np.zeros(traces_raw.shape[0], np.float32)

    for i in range(traces_raw.shape[0]):
        curr_trace_c = traces_raw[i]
        curr_trace_s = traces_srround[i]
        curr_r, curr_err, curr_trace_sub = hl.neural_pil_subtraction(curr_trace_c, curr_trace_s, lam=lam)
        print("roi_%s \tr = %.4f; error = %.4f." % (ft.int2str(i, 5), curr_r, curr_err))
        traces_subtracted[i] = curr_trace_sub
        ratio[i] = curr_r
        err[i] = curr_err

    print('\nplotting neuropil subtraction results ...')
    figures_folder = 'figures/neuropil_subtraction_lam_{}'.format(lam)
    if not os.path.isdir(figures_folder):
        os.makedirs(figures_folder)

    params = []
    for roi_ind in range(traces_raw.shape[0]):

        curr_traces = np.array([traces_raw[roi_ind], traces_srround[roi_ind], traces_subtracted[roi_ind]])

        params.append((curr_traces, plot_chunk_size, roi_ind, figures_folder))

    p = Pool(process_num)
    p.map(plot_traces_for_multi_process, params)

    # wait for keyboard abortion
    # msg = raw_input('Do you want to save? (y/n)\n')
    # while True:
    #     if msg == 'y':
    #         break
    #     elif msg == 'n':
    #         sys.exit('Stop process without saving.')
    #     else:
    #         msg = raw_input('Do you want to save? (y/n)\n')

    data_f['traces_center_subtracted'] = traces_subtracted
    data_f['neuropil_r'] = ratio
    data_f['neuropil_err'] = err

    data_f.close()

if __name__ == "__main__":
    run()