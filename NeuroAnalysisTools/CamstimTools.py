import os
import numpy as np
import tkinter
import warnings
from .core import FileTools as ft

def align_visual_display_time(pkl_dict, ts_pd_fall, ts_display_rise, max_mismatch=0.1, verbose=True,
                              refresh_rate=60., allowed_jitter=0.01):

    """
    align photodiode and display frame TTL for Brain Observatory stimulus. During display, sync square
    alternate between black (duration 1 second) and white (duration 1 second) start with black.
    The beginning of the display was indicated by a quick flash of [black, white, black, white, black, white],
    16.6667 ms each. This function will find the frame indices of each onset of black syncsquare during display,
    and compare them to the corresponding photodiode timestamps (digital), and calculate mean display lag.

    :param pkl_dict: the dictionay of display log
    :param ts_pd_fall: 1d array, timestameps of photodiode fall
    :param ts_display_rise: 1d array, timestamps of the rise of display frames
    :param max_mismatch: positive float, second, If any single display lag is larger than 'max_mismatch'.
                         a ValueException will raise
    :param refresh_rate: positive float, monitor refresh rate, Hz
    :param allowed_jitter: positive float, allowed jitter to evaluate time interval, sec
    :return: ts_display_real, onset timestamps of each display frames after correction for display lag
             display_lag: 2d array, with two columns
                          first column: timestamps of black sync square onsets
                          second column: display lag at that time
    """

    if not (len(ts_pd_fall.shape) == 1 and ts_pd_fall.shape[0] > 8):
        raise ValueError('input "ts_pd_fall" should be a 1d array with more than 8 items.')

    if not len(ts_display_rise.shape) == 1:
        raise ValueError('input "ts_display_rise" should be a 1d array.')

    if not pkl_dict[b'items'][b'sync_square'][b'colorSequence'][0] == -1:
        raise ValueError('The visual display did not start with black sync_square!')

    frame_period = pkl_dict[b'items'][b'sync_square'][b'frequency'] * \
                   len(pkl_dict[b'items'][b'sync_square'][b'colorSequence'])

    ts_onset_frame_ttl = ts_display_rise[::frame_period]

    if verbose:
        print('Number of onset frame TTLs of black sync square: {}'.format(len(ts_onset_frame_ttl)))

    # import matplotlib.pyplot as plt
    # plt.plot(ts_pd_fall, np.ones(len(ts_pd_fall)), '.', label='square fall')
    # plt.plot(ts_onset_frame_ttl, np.ones(len(ts_onset_frame_ttl))+0.1, '.', label='vsync fall')
    # plt.legend()
    # plt.show()

    # detect display start in photodiode signal
    refresh_rate = float(refresh_rate)
    for i in range(3, len(ts_pd_fall) - 1):
        post_interval = ts_pd_fall[i + 1] - ts_pd_fall[i]
        pre_interval_1 = ts_pd_fall[i] - ts_pd_fall[i - 1]
        pre_interval_2 = ts_pd_fall[i - 1] - ts_pd_fall[i - 2]
        pre_interval_3 = ts_pd_fall[i - 2] - ts_pd_fall[i - 3]

        # print(pre_interval_3, pre_interval_2, pre_interval_1, post_interval)

        if (abs(post_interval - frame_period / refresh_rate) <= allowed_jitter) and \
            (abs(pre_interval_1 - 20. / refresh_rate) <= allowed_jitter) and \
            (abs(pre_interval_2 - 20. / refresh_rate) <= allowed_jitter) and \
            (abs(pre_interval_3 - 20. / refresh_rate) <= allowed_jitter):
            pd_start_ind = i
            break
        else:
            raise ValueError('Did not find photodiode signal marking the start of display.')

    for j in range(0, len(ts_pd_fall) - 1)[::-1]:
        pre_interval_1 = ts_pd_fall[j] - ts_pd_fall[j - 1]
        pre_interval_2 = ts_pd_fall[j - 1] - ts_pd_fall[j - 2]

        if (abs(pre_interval_1 - 20. / refresh_rate) <= allowed_jitter) and \
            (abs(pre_interval_2 - 20. / refresh_rate) <= allowed_jitter):
            pd_end_ind = j - 2
            break

        raise ValueError('Did not find photodiode signal marking the end of display.')

    ts_onset_frame_pd = ts_pd_fall[pd_start_ind : pd_end_ind]

    if verbose:
        print('Number of onset frame photodiode falls of black sync square: {}'.format(len(ts_onset_frame_pd)))

    # import matplotlib.pyplot as plt
    # plt.plot(ts_onset_frame_pd, np.ones(len(ts_onset_frame_pd)), '.', label='square fall')
    # plt.plot(ts_onset_frame_ttl, np.ones(len(ts_onset_frame_ttl)) + 0.1, '.', label='vsync fall')
    # plt.legend()
    # plt.show()

    # import matplotlib.pyplot as plt
    # min_len = min([len(ts_onset_frame_pd), len(ts_onset_frame_ttl)])
    # pd_temp = ts_onset_frame_pd[:min_len]
    # vsync_temp = ts_onset_frame_ttl[:min_len]
    # f, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    # axs[0].plot(pd_temp - vsync_temp, label='diff')
    # axs[1].hist(pd_temp - vsync_temp)
    # axs[0].legend()
    # plt.show()

    if not len(ts_onset_frame_ttl) == len(ts_onset_frame_pd):
        warnings.warn('Number of onset frame TTLs ({}) and Number of onset frame photodiode signals ({}) '
                         'do not match. Truncate!'.format(len(ts_onset_frame_ttl), len(ts_onset_frame_pd)))

        min_len = min([len(ts_onset_frame_ttl), len(ts_onset_frame_pd)])
        ts_onset_frame_ttl = ts_onset_frame_ttl[:min_len]
        ts_onset_frame_pd = ts_onset_frame_pd[:min_len]

    display_lag = ts_onset_frame_pd - ts_onset_frame_ttl

    # print(ts_onset_frame_ttl[0:50])
    # print(ts_onset_frame_pd[0:50])
    #
    # import matplotlib.pyplot as plt
    # plt.plot(display_lag)
    # plt.show()

    display_lag_mean = np.mean(display_lag)
    if verbose:
        print('Average display lag: {} sec.'.format(display_lag_mean))

    ts_display_real = ts_display_rise + display_lag_mean

    # check max display lag
    display_lag_max = np.max(np.abs(display_lag))
    display_lag_max_ind = np.argmax(np.abs(display_lag))

    if display_lag_max > max_mismatch:
        warnings.warn('Display lag number {} : {}(sec) is greater than allow max_mismatch {} sec. '
                      'Overwrite visual frame timestamps by interpolating photodiode signal.'
                         .format(display_lag_max_ind, display_lag_max, max_mismatch))

        x_ttl = np.arange(len(ts_onset_frame_pd) * frame_period)
        x_pd = x_ttl[::frame_period]
        ts_display_real = np.interp(x=x_ttl, xp=x_pd, fp=ts_onset_frame_pd)

        # import matplotlib.pyplot as plt
        # plt.plot(ts_onset_frame_pd,
        #          ts_onset_frame_pd - ts_display_real[::frame_period])
        # plt.show()

        # import matplotlib.pyplot as plt
        # plt.plot(ts_onset_frame_pd, np.ones(len(ts_onset_frame_pd)), '.', label='square fall')
        # plt.plot(ts_display_real[::frame_period], np.ones(len(ts_display_real[::frame_period])) + 0.1,
        #          '.', label='vsync fall')
        # plt.legend()
        # plt.show()

    return ts_display_real, np.array([ts_onset_frame_pd, display_lag]).transpose()

def get_stim_dict_drifting_grating(input_dict, stim_name):

    sweep_table = input_dict[b'sweep_table']
    sweep_order = input_dict[b'sweep_order']
    sweep_frames = input_dict[b'sweep_frames']

    # get sweep table
    sweeps = []
    blank_sweep = np.array([np.nan, np.nan, np.nan, np.nan, 1.], dtype=np.float32)
    for sweep_i in sweep_order:
        if sweep_i == -1:
            sweeps.append(blank_sweep)
        else:
            curr_s = sweep_table[sweep_i]
            sweeps.append(np.array([curr_s[0], curr_s[1], curr_s[2], curr_s[3], 0.], dtype=np.float32))
    sweeps = np.array(sweeps, dtype=np.float32)

    # get sweep onset frames
    sweep_onset_frames = [int(sf[0]) for sf in sweep_frames]
    sweep_onset_frames = np.array(sweep_onset_frames, dtype=np.uint64)

    stim_dict = {}
    stim_dict['stim_name'] = stim_name
    stim_dict['sweeps'] = sweeps
    stim_dict['sweep_onset_frames'] = sweep_onset_frames
    stim_dict['data_formatting'] = [b'contrast', b'temporal_frequency', b'spatial_frequency', b'direction', b'is_blank']
    stim_dict['iterations'] = input_dict[b'runs']
    stim_dict['temporal_frequency_list'] = input_dict[b'sweep_params'][b'TF'][0]
    stim_dict['spatial_frequency_list'] = input_dict[b'sweep_params'][b'SF'][0]
    stim_dict['contrast_list'] = input_dict[b'sweep_params'][b'Contrast'][0]
    stim_dict['direction_list'] = input_dict[b'sweep_params'][b'Ori'][0]
    stim_dict['sweep_dur_sec'] = input_dict[b'sweep_length']
    stim_dict['midgap_dur_sec'] = input_dict[b'blank_length']
    stim_dict['num_of_blank_sweeps'] = input_dict[b'blank_sweeps']
    stim_dict['stim_text'] = input_dict[b'stim_text']
    stim_dict['frame_rate_hz'] = input_dict[b'fps']

    stim_dict['source'] = 'camstim'
    stim_dict['comments'] = 'The timestamps of this stimulus is the display frame index, not the actual time in seconds. ' \
                            'To get the real timestamps in seconds, please use these indices to find the timestamps ' \
                            'of displayed frames in "/processing/visual_display/frame_timestamps".'
    stim_dict['description'] = 'This stimulus is extracted from the pkl file saved by camstim software.'
    stim_dict['total_frame_num'] = input_dict[b'total_frames']

    return stim_dict

def analyze_LSN_movie(arr, alt_lst=None, azi_lst=None, dark=0, bright=255, verbose=False):
    """
    extract the frame indices of every square in the LSN movie displayed by CamStim
    :param arr: input 3-d array. frame * y * x
    :param azi_lst: list of azimuth locations of square center (same size as arr.shape[2])
    :param alt_lst: list of altitude locations of square center (same size as arr.shape[1])
    :param dark: int, the intensity level of dark probe
    :param bright: int, the intensity level of brigh probe

    :return: probes, list of lists of displayed probes for each frame.
                     length should be the same as number of frames of input arr.
                     each item is a list of displayed probes for the given frame.
                     each probe is a list of three numbers:
                        0: altitude (float)
                        1: azimuth (float)
                        2: sign (int, -1 or 1)
    """

    if not np.issubdtype(arr.dtype, np.uint8):
        raise ValueError('input array should have dtype as np.uint8')

    if not len(arr.shape) == 3:
        raise ValueError('input array should be 3-d.')

    if azi_lst is None:
        azi_lst = range(arr.shape[2])
    else:
        if not len(azi_lst) == arr.shape[2]:
            raise ValueError('the length of azi_lst should match arr.shape[2]')

    if alt_lst is None:
        alt_lst = range(arr.shape[1])
    else:
        if not len(alt_lst) == arr.shape[1]:
            raise ValueError('the length of alt_lst should match arr.shape[1]')

    probes = []

    for frame_i, frame in enumerate(arr):
        frame_probes = []

        #this can be optimized by using np.where
        for alt_i, line in enumerate(frame):
            for azi_i, probe in enumerate(line):

                if (probe != dark) & (probe != bright):
                    continue
                elif probe == dark:
                    frame_probes.append([float(alt_lst[alt_i]), float(azi_lst[azi_i]), -1])
                else:
                    frame_probes.append([float(alt_lst[alt_i]), float(azi_lst[azi_i]), 1])
        probes.append(frame_probes)

    if verbose:
        for f_i, f_p in enumerate(probes):
            print('frame index: {}'.format(f_i))
            for p in f_p:
                print('\talt:{}, azi:{}, sign:{}'.format(p[0], p[1], p[2]))

    return probes

def get_stim_dict_locally_sparse_noise(input_dict, stim_name, npy_path=None):

    if npy_path is None:
        root = tkinter.Tk()
        root.withdraw()
        movie_path = tkinter.filedialog.askopenfilename()
        mov = np.load(movie_path)
    else:
        mov = np.load(npy_path)

    print('loaded movie with shape: {}'.format(mov.shape))
    if mov.shape[1] == 8 and mov.shape[2] == 14:
        alt_lst = np.arange(8) * 9.3 - (9.3 * 3.5)
        azi_lst = np.arange(14) * 9.3 - (9.3 * 6.5)
        probe_size = 9.3
    else:
        alt_lst = None
        azi_lst = None
        probe_size = 'unknown'

    probes = analyze_LSN_movie(arr=mov, alt_lst=alt_lst, azi_lst=azi_lst, verbose=False)

    runs = input_dict['runs']
    sweep_frames = input_dict['sweep_frames']
    '''sweep_frames is a list of tuples, each tuple has two integers, representing start and end
    visual frame indices for a given template frame'''

    #check runs
    if len(probes) * runs != len(sweep_frames):
        raise ValueError('template frame number ({}) x runs ({}) = {} does not match saved displayed'
                         'frame number ({}).'.format(len(probes), runs, len(probes)*runs, len(sweep_frames)))

    template_frame_ind = range(len(probes)) * runs # sequence of template frame ind displayed

    single_probes = [] # list of probes displayed chronologically
    local_frame_ind = [] # same length as single probes, local visual frame indices for each single probes
    for sweep_i, template_i in enumerate(template_frame_ind):
        curr_probe_onset = sweep_frames[sweep_i][0]
        for p_f in probes[template_i]:
            single_probes.append(p_f)
            local_frame_ind.append(curr_probe_onset)

    stim_dict = {}
    stim_dict['stim_name'] = stim_name
    stim_dict['probes'] = np.array(single_probes, dtype=np.float32)
    stim_dict['template_frame_ind'] = template_frame_ind
    stim_dict['data_formatting'] = ['alt', 'azi', 'sign']
    stim_dict['probe_frame_num'] = int(input_dict['sweep_length'] * 60.)
    stim_dict['local_frame_ind'] = np.array(local_frame_ind, dtype=np.uint64)

    #meta data
    stim_dict['stim_text'] = input_dict['stim_text']
    stim_dict['frame_rate_hz'] = input_dict['fps']
    stim_dict['source'] = 'camstim'
    stim_dict['comments'] = 'The timestamps of this stimulus is the display frame index, not the actual time in seconds. ' \
                      'To get the real timestamps in seconds, please use these indices to find the timestamps ' \
                      'of displayed frames in "/processing/visual_display/frame_timestamps".'
    stim_dict['description'] = 'This stimulus is extracted from the pkl file saved by camstim software.'
    stim_dict['total_frame_num'] = input_dict['total_frames']

    return stim_dict

def get_stim_dict_list(pkl_path, lsn_npy_path=None):
    pkl_dict = ft.loadFile(pkl_path)
    stimuli = pkl_dict[b'stimuli']
    pre_blank_sec = pkl_dict[b'pre_blank_sec']
    post_blank_sec = pkl_dict[b'post_blank_sec']
    total_fps = pkl_dict[b'fps']

    start_frame_num = int(total_fps * pre_blank_sec)

    assert(pkl_dict[b'vsynccount'] == pkl_dict[b'total_frames'] + (pre_blank_sec + post_blank_sec) * total_fps)

    # print('\n'.join(pkl_dict.keys()))

    stim_dict_lst = []

    for stim_ind, stim in enumerate(stimuli):
        # print('\n'.join(stim.keys()))
        # print stim['stim_path']

        # get stim_type
        stim_str = stim[b'stim']
        if b'(' in stim_str:

            if stim_str[0:stim_str.index(b'(')] == b'GratingStim':

                if b'Phase' in stim[b'sweep_params'].keys():
                    stim_type = 'static_grating_camstim'
                elif b'TF' in stim[b'sweep_params'].keys():
                    stim_type = 'drifting_grating_camstim'
                else:
                    print('\n\nunknow stimulus type:')
                    print(stim[b'stim_path'])
                    print(stim[b'stim_text'])
                    stim_type = None

            elif stim_str[0:stim_str.index(b'(')] == b'ImageStimNumpyuByte':

                if b'locally_sparse_noise' in stim[b'stim_path']:
                    stim_type = 'locally_sparse_noise'
                else:
                    print('\n\nunknow stimulus type:')
                    print(stim[b'stim_path'])
                    print(stim[b'stim_text'])
                    stim_type = None

            else:
                print('\n\nunknow stimulus type:')
                print(stim[b'stim_path'])
                print(stim[b'stim_text'])
                stim_type = None

        else:
            print('\n\nunknow stimulus type:')
            print(stim[b'stim_path'])
            print(stim[b'stim_text'])
            stim_type = None

        if stim_type == 'drifting_grating_camstim':
            stim_name = '{:03d}_DriftingGratingCamStim'.format(stim_ind)
            print('\n\nextracting stimulus: ' + stim_name)
            stim_dict = get_stim_dict_drifting_grating(input_dict=stim, stim_name=stim_name)
            stim_dict['sweep_onset_frames'] = stim_dict['sweep_onset_frames'] + start_frame_num
            stim_dict.update({'stim_type': 'drifting_grating_camstim'})
            start_frame_num = stim_dict['total_frame_num']
        elif stim_type == 'locally_sparse_noise':
            stim_name = '{:03d}_LocallySparseNoiseCamStim'.format(stim_ind)
            print('\n\nextracting stimulus: ' + stim_name)
            stim_dict = get_stim_dict_locally_sparse_noise(input_dict=stim, stim_name=stim_name, npy_path=lsn_npy_path)
            stim_dict['global_frame_ind'] = stim_dict['local_frame_ind'] + start_frame_num
            stim_dict.update({'stim_type': 'locally_sparse_noise_camstim'})
            start_frame_num = stim_dict['total_frame_num']
        elif stim_type == 'static_gratings':
            print('\n\nskip static_gratings stimulus. stim index: {}.'.format(stim_ind))

            # needs to fill in
            stim_dict = {'stim_name': '{:03d}_StaticGratingCamStim'.format(stim_ind),
                         'stim_type': 'static_grating_camstim',
                         'total_frame_num': stim[b'total_frames']}


            start_frame_num = stim[b'total_frame_num']
        else:
            print('\nskip unknow stimstimulus. stim index: {}.'.format(stim_ind))

            # place holder
            stim_dict = {'stim_name': '{:03d}_UnknownCamStim'.format(stim_ind),
                         'stim_type': 'unknow_camstim',
                         'total_frame_num': stim[b'total_frames']}
            start_frame_num = stim[b'total_frame_num']

        stim_dict_lst.append(stim_dict)

    return stim_dict_lst


if __name__ == '__main__':
    # ================================================================================================================
    # pkl_path = '/media/junz/m2ssd/2017-09-25-preprocessing-test/m255_presynapticpop_vol1_bessel_DriftingGratingsTemp.pkl'

    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/642817351_338502_20171010_stim.pkl'
    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/642244262_338502_20171006_stim.pkl'
    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/642499066_338502_20171009_stim.pkl'
    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/642817351_338502_20171010_stim.pkl'
    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/643543433_338502_20171016_stim.pkl'
    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/643646020_338502_20171017_stim.pkl'
    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/643792098_338502_20171018_stim.pkl'
    #
    # pkl_path = '/media/junz/data3/data_soumya/2018-10-23-Soumya-LSN-analysis/1' \
    #            '/m255_presynapticpop_vol1_2nd_pass_LocallySparseNoiseTemp.pkl'
    # lsn_npy_path = '/media/junz/data3/data_soumya/2018-10-23-Soumya-LSN-analysis/sparse_noise_8x14_short.npy'
    # stim_dicts = get_stim_dict_list(pkl_path=pkl_path, lsn_npy_path=lsn_npy_path)
    # ================================================================================================================

    # ================================================================================================================
    pkl_path = r"Z:\v1dd\example_stim.pkl"
    lsn_npy_path = r"Z:\v1dd\v1dd_jun\meta\sparse_noise_no_boundary_16x28_scaled.npy"
    stim_dicts = get_stim_dict_list(pkl_path=pkl_path, lsn_npy_path=lsn_npy_path)
    # ================================================================================================================
    print('for debug')
