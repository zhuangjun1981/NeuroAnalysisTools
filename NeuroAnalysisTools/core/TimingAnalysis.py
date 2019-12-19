__author__ = 'junz'

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import numbers

plt.ioff()

def up_crossings(data, threshold=0):
    """
    find the index where the data up cross the threshold. return the indices of all up crossings (the onset data point
    that is greater than threshold, 1d-array). The input data should be 1d array.
    """
    if len(data.shape) != 1:
        raise ValueError('Input data should be 1-d array.')

    pos = data > threshold
    return (~pos[:-1] & pos[1:]).nonzero()[0] + 1


def down_crossings(data, threshold=0):
    """
    find the index where the data down cross the threshold. return the indices of all down crossings (the onset data
    point that is less than threshold, 1d-array). The input data should be 1d array.
    """
    if len(data.shape) != 1:
        raise ValueError('Input data should be 1-d array.')

    pos = data < threshold
    return (~pos[:-1] & pos[1:]).nonzero()[0] + 1


def all_crossings(data, threshold=0):
    """
    find the index where the data cross the threshold in either directions. return the indices of all crossings (the
    onset data point that is less or greater than threshold, 1d-array). The input data should be 1d array.
    """
    if len(data.shape) != 1:
        raise ValueError('Input data should be 1-d array.')

    pos_up = data > threshold
    pos_down = data < threshold
    return ((~pos_up[:-1] & pos_up[1:]) | (~pos_down[:-1] & pos_down[1:])).nonzero()[0] + 1


def threshold_onset(data, threshold=0, direction='up', fs=10000.):
    '''

    :param data: time trace
    :param threshold: threshold value
    :param direction: 'up', 'down', 'both'
    :param fs: sampling rate
    :return: timing of each crossing
    '''

    if direction == 'up': onsetInd = up_crossings(data, threshold)
    elif direction == 'down': onsetInd = down_crossings(data, threshold)
    elif direction == 'both': onsetInd = all_crossings(data, threshold)
    return onsetInd/float(fs)


# def discrete_cross_correlation2(ts1, ts2, t_range=(-1., 1.), bins=100, isPlot=False):
#     """
#     cross correlation of two time series of discrete events, return crosscorrelogram of total event 2 counts triggered
#     by event 1.
#
#     :param ts1: numpy.array, timestamps of the first event
#     :param ts2: numpy.array, timestamps of the second event
#     :param t_range: tuple of two elements, temporal window of crosscorrelogram, the first element should be smaller than
#                     the second element.
#     :param bins: int, number of bins
#     :param isPlot:
#     :return: t: numpy.array, time axis of crosscorrelorgam, mark the left edges of each time bin
#              value: numpy array, total event 2 counts in each time bin
#     """
#
#     # todo: optimize this
#
#     binWidth = (float(t_range[1]) - float(t_range[0])) / bins
#     t = np.arange(bins) * binWidth + t_range[0]
#     intervals = zip(t, t + binWidth)
#     values = np.zeros(bins, dtype=np.int64)
#
#     for ts in list(ts1):
#         currIntervals = [x + ts for x in intervals]
#         for i, interval in enumerate(currIntervals):
#             values[i] += len(np.where(np.logical_and(ts2>interval[0],ts2<=interval[1]))[0])
#
#     if isPlot:
#         f = plt.figure(figsize=(15,4)); ax = f.add_subplot(111)
#         ax.bar([a[0] for a in intervals],values,binWidth*0.9);ax.set_xticks(t)
#
#     return t, values


def discrete_cross_correlation(ts1, ts2, t_range=(-1., 1.), bins=100, isPlot=False):
    """
    cross correlation of two time series of discrete events, return crosscorrelogram of total event 2 counts triggered
    by event 1.

    :param ts1: numpy.array, timestamps of the first event
    :param ts2: numpy.array, timestamps of the second event
    :param t_range: tuple of two elements, temporal window of crosscorrelogram, the first element should be smaller than
                    the second element.
    :param bins: int, number of bins
    :param isPlot:
    :return: t: numpy.array, time axis of crosscorrelorgam, mark the left edges of each time bin
             value: numpy array, total event counts in each time bin per trigger
    """

    bin_width = (float(t_range[1]) - float(t_range[0])) / bins
    t = np.arange(bins).astype(np.float64) * bin_width + t_range[0]
    intervals = zip(t, t + bin_width)
    values = np.zeros(bins, dtype=np.int64)
    ts1s = np.sort(ts1)  # sort first timestamps array
    ts2s = np.sort(ts2)  # sort second timestamps array

    # pick up valid timestamps in first sorted timestamps
    ts1s = ts1s[(ts1s >= (ts2s[0] - t_range[0])) & (ts1s < (ts2s[-1] - t_range[1]))]

    n = len(ts1s)

    if n == 0:
        print('no overlapping time range (defined as ' + str(t_range) + ' between two input timestamp arrays')
        # return None
    else:
        ts2_start_ind = 0

        for curr_ts1 in ts1s:
            ts2s_short = ts2s[ts2_start_ind:]
            for i, curr_ts2 in enumerate(ts2s_short):
                if curr_ts2 >= curr_ts1 + t_range[0]:
                    break
            ts2s_short = ts2s_short[i:]
            ts2_start_ind += i
            ts2s_short = ts2s_short[ts2s_short < curr_ts1 + t_range[1]]

            for j, curr_int in enumerate(intervals):
                curr_start = curr_ts1 + curr_int[0]
                curr_end = curr_ts1 + curr_int[1]

                values[j] = values[j] + np.sum(np.logical_and((ts2s_short >= curr_start),
                                                              (ts2s_short < curr_end)))

    # values = values.astype(np.float64) / n

    if isPlot:
        f = plt.figure(figsize=(15, 4))
        ax = f.add_subplot(111)
        ax.bar([a[0] for a in intervals], values, bin_width * 0.9)

    return t, values


def find_nearest(trace, value, direction=0):
    '''
    return the index in "trace" having the closest value to "value"

    direction: int, can be 0, 1 or -1
               if 0, look for all elements in the trace
               if 1, only look for elements not smaller than the value, return None if all elements in the trace are
                     smaller than the value
               if -1, only look for elements not larger than the value, return None if all elements in the trace are
                     larger than the value
    '''

    diff = (trace - value).astype(np.float32)

    if direction == -1:
        diff[diff > 0] = np.nan
        diff = diff * -1
    elif direction == 1:
        diff[diff < 0] = np.nan
    elif direction == 0:
        diff = np.abs(diff)
    else:
        raise ValueError('"direction" should be 0, 1 or -1.')

    if np.isnan(diff).all():
        return None
    else:
        return np.nanargmin(diff)


def get_event_with_pre_iei(ts_events, iei=None):
    """
    get events which has a pre inter event interval (IEI) longer than a certain period of time

    :param ts_events: 1-d array, timestamps of events
    :param iei: float, criterion for pre IEI duration

    :return: 1-d array, refined timestamps
    """

    if iei is None:
        ts_events.sort()
        return ts_events

    else:

        if not check_monotonicity(ts_events, direction='non-decreasing'):
            print('the input event timestamps are not monotonically non-decreasing.\n Sort timestamps ...')
            ts_events.sort()

        ts_refined = ts_events[1:][np.diff(ts_events) > iei]

        return ts_refined


def get_onset_timeStamps(trace, Fs=10000., threshold = 3., onsetType='raising'):
    '''
    param trace: time trace of digital signal recorded as analog
    param Fs: sampling rate
    return onset time stamps
    '''

    pos = trace > threshold
    if onsetType == 'raising':
        return ((~pos[:-1] & pos[1:]).nonzero()[0]+1)/float(Fs)
    elif onsetType == 'falling':
        return ((pos[:-1] & ~pos[1:]).nonzero()[0]+1)/float(Fs)
    else:
        raise LookupError('onsetType should be either "raising" or "falling"!')


def power_spectrum(trace, fs, freq_range=(0., 300.), freq_bins=300, is_plot=False):
    '''
    return power spectrum of a signal trace (should be real numbers) at sampling rate of fs

    :param: trace, numpy.array, input trace
    :param: fs, float, sampling rate (Hz)
    :param: freq_range, tuple of two floats, range of analyzed frequencies
    :param: freq_bins, int, number of freq_bins of frequency axis
    '''
    spectrum_full = np.abs(np.fft.rfft(trace))**2 / float(len(trace))
    freqs = np.fft.rfftfreq(trace.size, 1. / fs)

    freq_bin_width = (freq_range[1] - freq_range[0]) / freq_bins
    freq_axis = np.arange(freq_bins, dtype=np.float32) * freq_bin_width + freq_range[0]

    spectrum = np.zeros(freq_axis.shape, dtype=np.float32)

    for i, freq in enumerate(freq_axis):
        curr_spect = spectrum_full[(freqs >= freq) & (freqs < (freq + freq_bin_width))]
        spectrum[i] = sum(curr_spect)

    if is_plot:
        f=plt.figure()
        ax=f.add_subplot(111)
        ax.plot(freq_axis, spectrum)
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('power')
        plt.show()

    return spectrum, freq_axis


def sliding_power_spectrum(trace, fs, sliding_window_length=5., sliding_step_length=None, freq_range=(0., 300.),
                           freq_bins=300, is_plot=False, **kwargs):
    '''
    calculate power_spectrum of a given trace over time

    :param: trace: input signal trace
    :param: fs: sampling rate (Hz)
    :param: sliding_window_length: length of sliding window (sec)
    :param: sliding_step_length: length of sliding step (sec), if None, equal to sliding_window_length
    :param: freq_range, tuple of two floats, range of analyzed frequencies
    :param: freq_bins, int, number of freq_bins of frequency axis
    :param: is_plot: bool, to plot or not

    :param: **kwargs, inputs to plt.imshow function

    :return
    spectrum: 2d array, power at each frequency at each time,
              time is from the first column to the last column
              frequence is from the last row to the first row
    times: time stamp for each column (starting point of each sliding window)
    freq_axis: frequency for each row (from low to high)
    '''

    if len(trace.shape) != 1: raise ValueError('Input trace should be 1d array!')

    total_length = len(trace) / float(fs)

    time_line = np.arange(len(trace)) * (1. / fs)

    freq_bin_width = (freq_range[1] - freq_range[0]) / freq_bins
    freq_axis = np.arange(freq_bins, dtype=np.float32) * freq_bin_width + freq_range[0]

    if sliding_step_length is None:
        sliding_step_length = sliding_window_length

    if sliding_step_length > sliding_window_length:
        print("Step length larger than window length, not using all data points!")

    times = np.arange(0., total_length, sliding_step_length)
    times = times[(times + sliding_window_length) < total_length]

    if len(times) == 0: raise ValueError('No time point found.')
    else:
        points_in_window = int(sliding_window_length * fs)
        if points_in_window <= 0: raise ValueError('Sliding window length too short!')
        else:
            spectrum = np.zeros((len(freq_axis), len(times)))
            for idx, start_time in enumerate(times):
                starting_point = find_nearest(time_line, start_time)
                ending_point = starting_point + points_in_window
                current_trace = trace[starting_point:ending_point]
                current_spectrum, freq_axis = power_spectrum(current_trace, fs, freq_range=freq_range, freq_bins=freq_bins,
                                                             is_plot=False)
                spectrum[:,idx] = current_spectrum

    if is_plot:
        f = plt.figure(figsize=(15, 6)); ax = f.add_subplot(111)
        fig = ax.imshow(spectrum, interpolation='nearest', **kwargs)
        ax.set_xlabel('times (sec)')
        ax.set_ylabel('frequency (Hz)')
        ax.set_xticks(list(range(len(times)))[::(len(times)//10)])
        ax.set_yticks(list(range(len(freq_axis)))[::(len(freq_axis)//10)])
        ax.set_xticklabels(times[::(len(times)//10)])
        ax.set_yticklabels(freq_axis[::(len(freq_axis)//10)])
        ax.invert_yaxis()
        ax.set_aspect(float(len(times)) * 0.5 / float(len(freq_axis)))
        f.colorbar(fig)

        return spectrum, times, freq_axis, f
    else:
        return spectrum, times, freq_axis


def get_burst(spikes, pre_isi=(-np.inf, -0.1), inter_isi=0.004, spk_num_thr=2):
    """

    detect bursts with certain pre inter-spike-interval (ISI) and within ISI.

    :param spikes: timestamps of the spike train
    :param pre_isi: the criterion of the pre burst ISI. all burst should have pre ISIs within this duration.
         unit: second. default: [-inf, -0.1]
    :param inter_isi:  the criterion of the inter burst ISI. The spikes within a burst should have ISIs no longer than
        this duration. unit: second. default: 0.004
    :param spk_num_thr: the criterion of the number of spike within a burst. All bursts should have no less than this
        number of spikes, int (larger than 1), default: 2
    :return:
        burst_ts: timestamps of each burst
        burst_ind: N x 2 np.array, np.uint32, each row is a burst, first column is the onset index if this burst in the
                   spike train, second column is the number of spikes in this burst
    """

    if inter_isi >= -pre_isi[1]:
        raise ValueError('inter_isi should be way shorter than pre_isi threshold.')

    burst_ts = []
    burst_ind = []

    i = 1

    while i <= len(spikes)-2:

        curr_pre_isi = spikes[i-1] - spikes[i]
        curr_post_isi = spikes[i+1] - spikes[i]

        if pre_isi[0] <= curr_pre_isi <= pre_isi[1] and curr_post_isi<=inter_isi:
            burst_ts.append(spikes[i])

            j = 2

            while (i + j) <= len(spikes) - 1:
                next_isi = spikes[i + j] - spikes[i + j - 1]
                if next_isi <= inter_isi:
                    j += 1
                else:
                    break

            burst_ind.append([i, j])

            i += j

        else:

            i += 1

    burst_ts = np.array(burst_ts, dtype=np.float)
    burst_ind = np.array(burst_ind, dtype=np.uint)

    burst_ts = burst_ts[burst_ind[:, 1] >= spk_num_thr]
    burst_ind = burst_ind[burst_ind[:, 1] >= spk_num_thr]

    return burst_ts, burst_ind


def possion_event_ts(duration=600., firing_rate = 1., refractory_dur=0.001, is_plot=False):
    """
    return possion event timestamps given firing rate and durantion
    """

    curr_t = 0.
    ts = []
    isi = []

    while curr_t < duration:
        curr_isi = np.random.exponential(1. / firing_rate)

        while curr_isi <= refractory_dur:
            curr_isi = np.random.exponential(1. / firing_rate)

        ts.append(curr_t + curr_isi)
        isi.append(curr_isi)
        curr_t += curr_isi

    if is_plot:
        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(111)
        ax.hist(isi, bins=1000)

    return np.array(ts)


def check_monotonicity(arr, direction='increasing'):
    """
    check monotonicity of a 1-d array, usually a time series

    :param arr: input array, should be 1 dimensional
    :param direction: 'increasing', 'decreasing', 'non-increasing', 'non-decreasing'
    :return: True or False
    """

    if len(arr.shape) != 1:
        raise ValueError('Input array should be one dimensional!')

    if arr.shape[0] < 2:
        raise ValueError('Input array should have at least two elements!')

    diff = np.diff(arr)
    min_diff = np.min(diff)
    max_diff = np.max(diff)

    if direction == 'increasing':
        if min_diff > 0:
            return True
        else:
            return False

    elif direction == 'decreasing':
        if max_diff < 0:
            return True
        else:
            return False

    elif direction == 'non-increasing':
        if max_diff <= 0:
            return True
        else:
            return False

    elif direction == 'non-decreasing':
        if min_diff >= 0:
            return True
        else:
            return False

    else:
        raise LookupError('direction should one of the following: "increasing", "decreasing", '
                          '"non-increasing", "non-decreasing"!')


def butter_bandpass_filter(cutoffs=(300., 6000.), fs=30000., order=5, is_plot=False):
    """
    bandpass digital butterworth filter design
    :param cutoffs: [low cutoff frequency, high cutoff frequency], Hz
    :param fs: sampling rate, Hz
    :param order:
    :param is_plot:
    :return: b, a
    """
    nyq = 0.5 * fs
    low = cutoffs[0] / nyq
    high = cutoffs[1] / nyq

    b, a = sig.butter(N=order, Wn=[low, high], btype='band', analog=False, output='ba')

    if is_plot:
        w, h = sig.freqz(b, a, worN=2000)
        f = plt.figure(figsize=(10, 10))
        plt.loglog((fs * 0.5 / np.pi) * w, abs(h))
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude')
        # plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(cutoffs[0], color='green')
        plt.axvline(cutoffs[1], color='red')
        # plt.xlim([0, 10000])
        plt.show()

    return b, a


def butter_lowpass_filter(cutoff=300., fs=30000., order=5, is_plot=False):
    """
    bandpass digital butterworth filter design
    :param cutoffs: cutoff frequency, Hz
    :param fs: sampling rate, Hz
    :param order:
    :param is_plot:
    :return: b, a
    """
    nyq = 0.5 * fs
    low = cutoff / nyq

    b, a = sig.butter(N=order, Wn=low, btype='low', analog=False, output='ba')

    if is_plot:
        w, h = sig.freqz(b, a, worN=2000)
        f = plt.figure(figsize=(10, 10))
        plt.loglog((fs * 0.5 / np.pi) * w, abs(h))
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude')
        # plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(cutoff, color='red')
        # plt.xlim([0, 10000])
        plt.show()

    return b, a


def butter_highpass_filter(cutoff=300., fs=30000., order=5, is_plot=False):
    """
    bandpass digital butterworth filter design
    :param cutoffs: cutoff frequency, Hz
    :param fs: sampling rate, Hz
    :param order:
    :param is_plot:
    :return: b, a
    """
    nyq = 0.5 * fs
    high = cutoff / nyq

    b, a = sig.butter(N=order, Wn=high, btype='high', analog=False, output='ba')

    if is_plot:
        w, h = sig.freqz(b, a, worN=2000)
        f = plt.figure(figsize=(10, 10))
        plt.loglog((fs * 0.5 / np.pi) * w, abs(h))
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude')
        # plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(cutoff, color='red')
        # plt.xlim([0, 10000])
        plt.show()

    return b, a


def butter_bandpass(trace, fs=30000., cutoffs=(300., 6000.), order=5):
    """
    band pass filter a 1-d signal using digital butterworth filter design

    :param trace: input signal
    :param cutoffs: [low cutoff frequency, high cutoff frequency], Hz
    :param fs: sampling rate, Hz
    :param order:
    :return: filtered signal
    """

    b, a = butter_bandpass_filter(cutoffs=cutoffs, fs=fs, order=order)
    filtered = sig.lfilter(b, a, trace)
    return filtered


def butter_lowpass(trace, fs=30000., cutoff=300., order=5):
    """
    lowpass filter a 1-d signal using digital butterworth filter design

    :param trace: input signal
    :param cutoff: cutoff frequency, Hz
    :param fs: sampling rate, Hz
    :param order:
    :return: filtered signal
    """
    b, a = butter_lowpass_filter(cutoff=cutoff, fs=fs, order=order)
    filtered = sig.lfilter(b, a, trace)
    return filtered


def butter_highpass(trace, fs=30000., cutoff=300., order=5):
    """
    highpass filter a 1-d signal using digital butterworth filter design

    :param trace: input signal
    :param cutoff: cutoff frequency, Hz
    :param fs: sampling rate, Hz
    :param order:
    :return: filtered signal
    """
    b, a = butter_highpass_filter(cutoff=cutoff, fs=fs, order=order)
    filtered = sig.lfilter(b, a, trace)
    return filtered


def notch_filter(trace, fs=30000., freq_base=60., bandwidth=1., harmonics=4, order=2):
    """
    filter out signal at power frequency band and its harmonics. for each harmonic, signal at this band is extracted
    by using butter_bandpass function. Then the extraced signal was subtracted from the original signal

    :param trace: 1-d array, input trace
    :param fs: float, sampling rate, Hz
    :param freq_base: float, Hz, base frequency of contaminating signal
    :param bandwidth: float, Hz, filter bandwidth at each side of center frequency
    :param harmonics: int, number of harmonics to filter out
    :param order: int, order of butterworth filter, for a narrow band, shouldn't be larger than 2
    :return: filtered signal
    """

    sig_extracted = np.zeros(trace.shape, dtype=np.float32)

    for har in (np.arange(harmonics) + 1):
        cutoffs = [freq_base * har - bandwidth, freq_base * har + bandwidth]
        curr_sig = butter_bandpass(trace, fs=fs, cutoffs=cutoffs, order=order)
        sig_extracted = sig_extracted + curr_sig

    trace_filtered = trace.astype(np.float32) - sig_extracted

    return trace_filtered.astype(trace.dtype)


def event_triggered_average_irregular(ts_event, continuous, ts_continuous, t_range=(-1., 1.), bins=100, is_plot=False):
    """
    event triggered average of an analog signal trigger by discrete events. The timestamps of the analog signal may not
    be regular

    :param ts_event: 1-d array, float, timestamps of trigging event
    :param continuous: 1-d array, float, value of the analog signal
    :param ts_continuous: 1-d array, float, timestamps of the analog signal, should have same length as continuous
    :param t_range: tuple of 2 floats, temporal range of calculated average
    :param bins: int, number of bins of calculated average
    :param is_plot:
    :return: eta: 1-d array, float, event triggered average
             n: 1-d array, unit, number of time point of each bin
             t: 1-d array, float, time axis of event triggered average
             std: 1-d array, float, standard deviation of each bin in the event triggered average

             all four returned arrays should have same length
    """

    if t_range[0] >= t_range[1]:
        raise ValueError('t_range[0] should be smaller than t_range[1].')

    # sort continuous channel to be monotonic increasing in temporal domain
    sort_ind = np.argsort(ts_continuous)
    ts_continuous = ts_continuous[sort_ind]
    continuous = continuous[sort_ind]

    # initiation
    bin_width = (t_range[1] - t_range[0]) / bins
    t = t_range[0] + np.arange(bins, dtype=np.float32) * bin_width

    eta_list = [[] for bin in t]
    n = np.zeros(t.shape, dtype=np.uint64)
    std = np.zeros(t.shape, dtype=np.float32)
    eta = np.zeros(t.shape, dtype=np.float32)
    eta[:] = np.nan

    print('\nStart calculating event triggered average ...')
    percentage = None

    for ind_eve, eve in enumerate(ts_event):

        # for display
        curr_percentage =  int((float(ind_eve) * 100. / float(len(ts_event))) // 10) * 10
        if curr_percentage != percentage:
            print('progress: ' + str(curr_percentage) + '%')
            # print eve, ':', ts_continuous[-1]
            percentage = curr_percentage

        if ((eve + t_range[0]) > ts_continuous[0]) and ((eve + t_range[1]) < ts_continuous[-1]):

            # slow algorithm
            # bin_starts = t + eve
            # for i, bin_start in enumerate(bin_starts):
            #     curr_datapoint =  continuous[(ts_continuous >= bin_start) & (ts_continuous < (bin_start + bin_width))]
            #     if len(curr_datapoint) != 0:
            #         eta_list[i] += list(curr_datapoint)

            # fast algorithm
            all_bin_start = eve + t_range[0]
            all_bin_end = eve + t_range[1]# - 1e-10
            for i, curr_t in enumerate(ts_continuous):
                if curr_t < all_bin_start:
                    continue
                elif curr_t >= all_bin_end:
                    break
                else:
                    bin_ind = int((curr_t - all_bin_start) // bin_width)
                    eta_list[bin_ind].append(continuous[i])

    for j, datapoints in enumerate(eta_list):
        if len(datapoints) > 0:
            eta[j] = np.mean(datapoints)
            n[j] = len(datapoints)
        if len(datapoints) > 1:
            std[j] = np.std(datapoints)

    if is_plot:
        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(111)
        ax.fill_between(t, eta - std, eta + std, edgecolor='none', facecolor='#888888', alpha=0.5)
        ax.plot(t, eta, '-k')
        ax.set_title('event triggered average')
        ax.set_xlabel('time (sec)')
        ax.set_xlim([t_range[0], t_range[1] - 1. / fs])
        plt.show()

    return eta, t, n, std


def event_triggered_average_regular(ts_event, continuous, fs_continuous, start_time_continuous, t_range=(-1., 1.),
                                    is_normalize=False, is_plot=False):
    """
    event triggered average of an analog signal trigger by discrete events. The time sampling of the analog signal
    should be regular

    :param ts_event: 1-d array, float, timestamps of trigging event
    :param continuous: 1-d array, float, value of the analog signal
    :param fs_continuous: float, sampling rate (Hz) of the analog signal
    :param start_time_continuous: float, the timestamp of the first element of continuous
    :param t_range: tuple of 2 floats, temporal range of calculated average
    :param is_normalize: if True: all event triggered trace will subtract baseline, the baseline is defined as
                         following:
                             if t_range[0] is smaller than zero: baseline is defined as the mean intensity
                             before trigger
                             if t_range[0] is no smaller than zero: baseline is defined as the mean of entire event
                             triggered trace
                         if False: keep the raw trace
    :param is_plot:
    :return: eta: 1-d array, float, event triggered average
             n: int, number of triggers actually used for getting eta
             t: 1-d array, float, time axis of event triggered average
             std: 1-d array, float, standard deviation of each bin in the event triggered average

             all three returned arrays should have same length
    """

    sample_dur = 1. / fs_continuous
    ts_event_a = ts_event - start_time_continuous

    ind_range = [int(t_range[0] / sample_dur), int(t_range[1] / sample_dur)]
    chunk_dur = ind_range[1] - ind_range[0]
    t = np.arange(chunk_dur) * sample_dur + t_range[0]

    if t_range[0] < 0:
        base_point_num = -int(t_range[0] / sample_dur)
    else:
        base_point_num = ind_range[1] - ind_range[0]

    unit_inds = np.round(ts_event_a / sample_dur).astype(np.int64)
    unit_inds = np.array([ind for ind in unit_inds if (ind + ind_range[0]) >= 0 and
                          (ind + ind_range[1]) < len(continuous)])

    eta_all = np.empty((len(unit_inds), chunk_dur), dtype=np.float32)

    for j, unit_ind in enumerate(unit_inds):
        curr_trace = continuous[unit_ind + ind_range[0]: unit_ind + ind_range[1]]

        if is_normalize:
            eta_all[j, :] = curr_trace - np.mean(curr_trace[0:base_point_num])
        else:
            eta_all[j, :] = curr_trace

    eta = np.mean(eta_all, axis=0)
    std = np.std(eta_all, axis=0)

    if is_plot:
        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(111)
        ax.fill_between(t, eta-std, eta+std, edgecolor='none', facecolor='#888888', alpha=0.5)
        ax.plot(t, eta, '-k')
        ax.set_title('event triggered average')
        ax.set_xlabel('time (sec)')
        ax.set_xlim([t_range[0], t_range[1]-1./ fs_continuous])
        plt.show()

    return eta, len(unit_inds), t, std


def event_triggered_event_trains(event_ts, triggers, t_range=(-1., 2.)):
    """
    calculate peri-trigger event timestamp trains.

    :param event_ts: array of float, discrete event timestamps
    :param triggers: array of float, trigger timestamps
    :param t_range: tuple of two floats, start and end time of the time window around the trigger
    :return: list of arrays, each array is a triggered train of event timestamps (relative to trigger time)
    """

    # event triggered timestamps
    etts = []

    for trigger in triggers:
        curr_st = trigger + t_range[0]
        curr_et = trigger + t_range[1]
        curr_train = event_ts[(event_ts >= curr_st) & (event_ts < curr_et)]
        etts.append(curr_train - trigger)

    return etts, t_range


def threshold_to_intervals(trace, thr, comparison='>='):
    """
    threshold a 1d trace, return intervals of indices that are above the threshold.

    :param trace: 1d array
    :param thr: float
    :param comparison: str, '>', '>=', '<' or '<='
    :return: list of tuples, each tuple contains two non-negative integers representing
             the start index and the end index of thresholded intervals. the first int should
             be smaller than the second int.
    """

    if len(trace.shape) != 1:
        raise ValueError("the input 'trace' should be a 1d array.")

    flag = False

    start = []
    end = []

    for pi, pv in enumerate(trace):

        if comparison == '>=':
            if pv >= thr and (not flag):
                start.append(pi)
                flag = True

            if pv < thr and flag:
                end.append(pi)
                flag = False

        elif comparison == '>':
            if pv > thr and (not flag):
                start.append(pi)
                flag = True

            if pv <= thr and flag:
                end.append(pi)
                flag = False

        elif comparison == '<=':
            if pv <= thr and (not flag):
                start.append(pi)
                flag = True

            if pv > thr and flag:
                end.append(pi)
                flag = False

        elif comparison == '<':
            if pv < thr and (not flag):
                start.append(pi)
                flag = True

            if pv >= thr and flag:
                end.append(pi)
                flag = False

        else:
            raise LookupError('Do not understand input "comparison", should be ">=", ">", "<=", "<".')

    if len(start) - len(end) == 1:
        end.append(len(trace))

    return list(zip(start, end))


def haramp(trace, periods, ceil_f=4):
    """
    get amplitudes of first couple harmonic components from a time series corresponding to a sinusoidal stimulus.

    :param trace: 1d array, input time series
    :param periods: positive int, number of stimulus periods contained in the input trace
    :param ceil_f: positive int, the number of harmonic component you want to get (default=4)
    :return: list of amplitude of each harmonic component [F0, F1, F2, ...], len will be ceil_f
    """

    if not (isinstance(periods, numbers.Integral) and periods > 0):
        raise ValueError('period should be a positive integer.')

    if not (isinstance(ceil_f, numbers.Integral) and ceil_f > 0):
        raise ValueError('ceil_f should be a positive integer.')

    if len(trace.shape) != 1:
        raise ValueError('input trace should be an 1d array.')

    if (ceil_f * periods * 1) > trace.shape[0]:
        raise ValueError('ceiling harmonic number: ceil_f is too high.')

    amp = np.abs(np.fft.fft(trace))
    harmonic = []

    for f in range(ceil_f):
        bin = f * periods
        if f == 0:
            harmonic.append(amp[bin] / float(trace.shape[0]))
        else:
            harmonic.append(2 * amp[bin] / float(trace.shape[0]))

    return harmonic


class TimeIntervals(object):
    """
    class to describe time intervals, designed to represent epochs

    self.data save the (start, end) timestamps of each epochs. Shape (n, 2).
    Each row: a single interval
    column 0: start timestamps
    column 1: end timestamps

    the intervals are incremental in time and should not have overlap within them.
    """

    def __init__(self, intervals):
        self._intervals = self.check_integraty(intervals)

    def get_intervals(self):
        return self._intervals

    @staticmethod
    def check_integraty(intervals):

        intervals_cp = np.array([np.array(d, dtype=np.float64) for d in intervals])
        intervals_cp = intervals_cp.astype(np.float64)

        if len(intervals_cp.shape) != 2:
            raise ValueError('intervals should be 2d.')

        if intervals_cp.shape[1] != 2:
            raise ValueError('intervals.shape[1] should be 2. (start, end) of the interval')

        # for interval_i, interval in enumerate(intervals_cp):
        #     if interval[1] <= interval[0]:
        #         raise ValueError('the {}th interval: end time ({}) earlier than start time ({})'.
        #                          format(interval_i, interval[1], interval[0]))

        intervals_cp = intervals_cp[intervals_cp[:, 0].argsort()]

        ts_list = np.concatenate(intervals_cp, axis=0)
        if not check_monotonicity(arr=ts_list, direction='increasing'):
            raise ValueError('The intervals should be incremental in time and should not have overlap within them.')

        return intervals_cp

    def overlap(self, time_intervals):
        """
        return a new TimeIntervals object that represents the overlap between self and the input Timeintervals

        :param time_intervals: corticalmapping.core.TimingAnalysis.TimeIntervals object
        """

        starts0 = [[s, 1] for s in self._intervals[:, 0]]
        ends0 = [[e, -1] for e in self._intervals[:, 1]]

        starts1 = [[s, 1] for s in time_intervals.get_intervals()[:, 0]]
        ends1 = [[e, -1] for e in time_intervals.get_intervals()[:, 1]]

        events_lst = starts0 + ends0 + starts1 + ends1
        # print(events_lst)

        ts_arr = np.array([e[0] for e in events_lst])
        events_lst = [events_lst[i] for i in np.argsort(ts_arr)]
        # print(events_lst)

        mask = np.cumsum([e[1] for e in events_lst])

        new_starts = []
        new_ends = []

        flag = 0 # 1: within overlap, 0: outside overlap
        for ts_i, msk in enumerate(mask):

            if flag == 0 and msk == 2:
                new_starts.append(events_lst[ts_i][0])
                flag = 1
            elif flag == 1 and msk < 2:
                new_ends.append(events_lst[ts_i][0])
                flag = 0
            elif flag == 1 and msk == 2:
                raise ValueError('do not understand the timestamps: flag={}, msk={}.'.format(flag, msk))
            else:
                pass

        if len(new_starts) != len(new_ends):
            raise ValueError('the length of new_starts ({}) does not equal the length of new_ends ({}).'.
                             format(len(new_starts), len(new_ends)))

        if new_starts:
            new_intervals = np.array([np.array(new_starts), np.array(new_ends)]).transpose()
            return TimeIntervals(intervals=new_intervals)
        else:
            return None

    def is_contain(self, time_interval):
        """
        :param time_interval: list or tuple of two floats, representing one single time interval
        :return: bool, if the input interval is completely contained by self
        """

        if len(time_interval) != 2:
            raise ValueError('input "time_interval" should have two and only two elements.')

        if time_interval[0] >= time_interval[1]:
            raise ValueError('the start of input "time_interval" should be earlier than the end.')

        for interval in self._intervals:

            if interval[0] > time_interval[0]: # current interval starts after input time_interval
                return False
            else: # current interval starts before input time_interval
                if interval[1] < time_interval[0]: # current interval ends before input time_interval
                    pass
                elif interval[1] < time_interval[1]: # current interval ends within input time_interval
                    return False
                else:
                    return True # current interval contains input time_interval

        # all intervals in self end before input time_interval
        # or self._intervals is empty
        return False

    def to_h5_group(self, grp):
        pass

    @staticmethod
    def from_h5_group(grp):
        pass


if __name__=='__main__':

    #============================================================================================================
    # a=np.arange(100,dtype=np.float)
    # b=a+0.5+(np.random.rand(100)-0.5)*0.1
    # c=discrete_cross_correlation(a,b,range=(0,1),bins=50,isPlot=True)
    # plt.show()
    #============================================================================================================

    #============================================================================================================
    # trace = np.array(([0.] * 5 + [5.] * 5) * 5)
    # ts = get_onset_timeStamps(trace, Fs=10000., onsetType='raising')
    # assert(ts[2] == 0.0025)
    # ts2 = get_onset_timeStamps(trace, Fs=10000., onsetType='falling')
    # assert(ts2[2] == 0.0030)
    #============================================================================================================

    #============================================================================================================
    # trace = np.random.rand(300) - 0.5
    # _, _ = power_spectrum(trace, 0.1, True)
    # plt.show()
    #============================================================================================================

    #============================================================================================================
    # time_line = np.arange(5000) * 0.01
    # trace = np.sin(time_line * (2 * np.pi))
    # trace2 = np.cos(np.arange(2500) * 0.05 * (2 * np.pi))
    # trace3 = np.cos(np.arange(2500) * 0.1 * (2 * np.pi))
    # trace = trace + np.concatenate((trace2, trace3))
    #
    # spectrum, times, freqs = sliding_power_spectrum(trace, 100, 1., is_plot=True)
    # print 'times:',times
    # print 'freqs:', freqs
    #============================================================================================================

    # ============================================================================================================
    # spikes = [0.3, 0.5, 0.501, 0.503, 0.505, 0.65, 0.7, 0.73, 0.733, 0.734, 0.735, 0.9, 1.5, 1.6,
    #           1.602, 1.603, 1.605, 1.94, 1.942]
    #
    # burst_ts, burst_ind = get_burst(spikes,  pre_isi=(-np.inf, -0.1), inter_isi=0.004, spk_num_thr=2)
    #
    # print burst_ts
    # print burst_ind
    # ============================================================================================================

    # ============================================================================================================
    # trace = np.arange(10)
    # print find_nearest(trace, 1.6)
    # ============================================================================================================

    # ============================================================================================================
    # possion_event_ts(firing_rate=1., duration=1000., refractory_dur=0.1, is_plot=True)
    # plt.show()
    # ============================================================================================================

    # ============================================================================================================
    # continuous = np.arange(1000) * 0.1
    # ts_continuous = np.arange(1000)
    # ts_event = [100, 101, 102, 200, 205]
    # eta, t, n, std = event_triggered_average_irregular(ts_event, continuous, ts_continuous, t_range=(-10., 10.), bins=20,
    #                                          is_plot=True)
    # print eta
    # print t
    # print n
    # print std
    # ============================================================================================================

    # ============================================================================================================
    # np.random.seed(100)
    # ts = np.arange(100) + np.random.rand(100) * 0.4
    # print ts
    # print np.min(np.diff(ts))
    # ts2 = get_event_with_pre_iei(ts, iei=0.8)
    # print ts2
    # print len(ts2)
    # print np.min(np.diff(ts2))
    # ============================================================================================================

    # ============================================================================================================
    # con = np.random.rand(1000)
    # template = np.arange(5)
    # # ts_s = np.array([0, 1, 4, 5, 6, 8, 23, 45, 65, 78, 90, 200, 230, 245, 255, 267, 355, 488, 593, 600, 615, 635, 644,
    # #                 700, 845, 846, 879, 900, 902, 908, 913, 922, 945, 950, 978, 999])
    #
    # ts_s = np.arange(0, 800, 5)
    #
    # for ts in ts_s:
    #     if ts > 2 and ts <997:
    #         con[ts-2: ts+3] = con[ts-2: ts+3] + template
    #
    # fs = 1.
    # t_range = (-2., 3.)
    # eta, n, t, std = event_triggered_average_regular(ts_s, con, fs, 0., t_range, is_plot=True)
    # print t
    # print len(ts_s)
    # print n
    # print eta
    # ============================================================================================================

    # ============================================================================================================
    # butter_highpass_filter(is_plot=True)
    # ============================================================================================================

    print('for debugging...')