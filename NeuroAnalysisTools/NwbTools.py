import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import warnings
# import corticalmapping.ephys.OpenEphysWrapper as oew
# import corticalmapping.ephys.KilosortWrapper as kw
from .core import FileTools as ft
from .core import TimingAnalysis as ta
from .core import PlottingTools as pt
from . import HighLevel as hl
from . import CamstimTools as ct
from .nwb.nwb import NWB as NWB

DEFAULT_GENERAL = {
    'session_id': '',
    'experimenter': '',
    'institution': 'Allen Institute for Brain Science',
    # 'lab': '',
    # 'related_publications': '',
    'notes': '',
    'experiment_description': '',
    # 'data_collection': '',
    'stimulus': '',
    # 'pharmacology': '',
    # 'surgery': '',
    # 'protocol': '',
    'subject': {
        'subject_id': '',
        # 'description': '',
        'species': 'Mus musculus',
        'genotype': '',
        'sex': '',
        'age': '',
        # 'weight': '',
    },
    # 'virus': '',
    # 'slices': '',
    'extracellular_ephys': {
        'electrode_map': '',
        'sampling_rate': 30000.,
        # 'electrode_group': [],
        # 'impedance': [],
        # 'filtering': []
    },
    'optophysiology': {
        # 'indicator': '',
        # 'excitation_lambda': '',
        # 'imaging_rate': '',
        # 'location': '',
        # 'device': '',
    },
    # 'optogenetics': {},
    'devices': {}
}
SPIKE_WAVEFORM_TIMEWINDOW = (-0.002, 0.002)


def plot_waveforms(waveforms, ch_locations=None, stds=None, waveforms_filtered=None, stds_filtered=None,
                   f=None, ch_ns=None, axes_size=(0.2, 0.2), **kwargs):
    """
    plot spike waveforms at specified channel locations

    :param waveforms: 2-d array, time point x channel, mean spike waveforms at each channel
    :param ch_locations: list of tuples, (x_location, y_location) for each channel, if None, waveform will be plotted
                         in a linear fashion
    :param stds: 2-d array, same size as waveform, measurement of variance of each time point at each channel
    :param waveforms_filtered: 2-d array, same size as waveform, waveforms from unfiltered analog signal
    :param stds_filtered: 2-d array, same size as waveform, measurement of variance of each time point at
                            each channel of unfiltered analog signal
    :param f: matplotlib figure object
    :param ch_ns: list of strings, names of each channel
    :param axes_size: tuple, sise of subplot, (width, height), only implemented when ch_locations is not None
    :return: f
    """

    if f is None:
        f = plt.figure(figsize=(10, 10))

    ch_num = waveforms.shape[1]

    if ch_locations is None:
        ax_s = []
        for i in range(ch_num):
            ax_s.append(f.add_subplot(1, ch_num, i + 1))
    else:
        ax_s = pt.distributed_axes(f, axes_pos=ch_locations, axes_size=axes_size)

    #  detect uniform y axis scale
    max_trace = waveforms.flatten()
    min_trace = waveforms.flatten()
    if stds is not None:
        max_trace = np.concatenate((max_trace, waveforms.flatten() + stds.flatten()))
        min_trace = np.concatenate((min_trace, waveforms.flatten() - stds.flatten()))
    if waveforms_filtered is not None:
        max_trace = np.concatenate((max_trace, waveforms_filtered.flatten()))
        min_trace = np.concatenate((min_trace, waveforms_filtered.flatten()))
        if stds_filtered is not None:
            max_trace = np.concatenate((max_trace, waveforms_filtered.flatten() + stds_filtered.flatten()))
            min_trace = np.concatenate((min_trace, waveforms_filtered.flatten() - stds_filtered.flatten()))

    peak_min = np.min(min_trace)
    peak_max = np.max(max_trace)

    for j, ax in enumerate(ax_s):

        # plot unfiltered data
        if waveforms_filtered is not None:
            curr_wf_f = waveforms_filtered[:, j]
            if stds_filtered is not None:
                curr_std_f = stds_filtered[:, j]
                ax.fill_between(range(waveforms_filtered.shape[0]), curr_wf_f - curr_std_f,
                                curr_wf_f + curr_std_f, color='#888888', alpha=0.5, edgecolor='none')
            ax.plot(curr_wf_f, '-', color='#555555', label='filtered', **kwargs)

        # plot filtered data
        curr_wf = waveforms[:, j]
        if stds is not None:
            curr_std = stds[:, j]
            ax.fill_between(range(waveforms.shape[0]), curr_wf - curr_std, curr_wf + curr_std,
                            color='#8888ff', alpha=0.5, edgecolor='none')
        ax.plot(curr_wf, '-', color='#3333ff', label='unfiltered', **kwargs)

        # plot title
        if ch_ns is not None:
            ax.set_title(ch_ns[j], y=0.9)

        ax.set_xlim([0, waveforms.shape[0] - 1])
        ax.set_ylim([peak_min, peak_max])
        ax.set_axis_off()

        if waveforms_filtered is not None and j == 0:
            ax.legend(frameon=False, loc='upper left', fontsize='small', bbox_to_anchor=(0, 0.9))

    return f


class RecordedFile(NWB):
    """
    Jun's wrapper of nwb file. Designed for LGN-ephys/V1-ophys dual recording experiments. Should be able to save
    ephys, wide field, 2-photon data in a single file.
    """

    def __init__(self, filename, is_manual_check=False, **kwargs):

        if os.path.isfile(filename):
            if is_manual_check:
                keyboard_input = ''
                while keyboard_input != 'y' and keyboard_input != 'n':
                    keyboard_input = input('\nthe path "' + filename + '" already exists. Modify it? (y/n) \n')
                    if keyboard_input == 'y':
                        super(RecordedFile, self).__init__(filename=filename, modify=True, **kwargs)
                    elif keyboard_input == 'n':
                        raise IOError('file already exists.')
            else:
                print('\nModifying existing nwb file: ' + filename)
                super(RecordedFile, self).__init__(filename=filename, modify=True, **kwargs)
        else:
            print('\nCreating a new nwb file: ' + filename)
            super(RecordedFile, self).__init__(filename=filename, modify=False, **kwargs)

    def add_general(self, general=DEFAULT_GENERAL, is_overwrite=True):
        """
        add general dictionary to the general filed
        """
        slf = self.file_pointer
        ft.write_dictionary_to_h5group_recursively(target=slf['general'], source=general, is_overwrite=is_overwrite)

    def add_sync_data(self, f_path, analog_downsample_rate=None, by_label=True, digital_labels=None,
                      analog_labels=None):

        sync_dict = ft.read_sync(f_path=f_path, analog_downsample_rate=analog_downsample_rate,
                                 by_label=by_label, digital_labels=digital_labels,
                                 analog_labels=analog_labels)

        # add digital channel
        if 'digital_channels' in sync_dict.keys():

            digital_channels = sync_dict['digital_channels']

            # get channel names
            for d_chn, d_ch in digital_channels.items():
                if ft.is_integer(d_chn):
                    curr_chn = 'digital_CH_' + ft.int2str(d_chn, 3)
                else:
                    curr_chn = 'digital_' + d_chn

                curr_rise = d_ch['rise']
                ch_series_rise = self.create_timeseries('TimeSeries', curr_chn + '_rise', 'acquisition')
                ch_series_rise.set_data([], unit='', conversion=np.nan, resolution=np.nan)
                if len(curr_rise) == 0:
                    curr_rise = np.array([np.nan])
                    ch_series_rise.set_time(curr_rise)
                    ch_series_rise.set_value('num_samples', 0)
                else:
                    ch_series_rise.set_time(curr_rise)
                ch_series_rise.set_description('timestamps of rise cross of digital channel: ' + curr_chn)
                ch_series_rise.set_source('sync program')
                ch_series_rise.set_comments('digital')
                ch_series_rise.finalize()

                curr_fall = d_ch['fall']
                ch_series_fall = self.create_timeseries('TimeSeries', curr_chn + '_fall', 'acquisition')
                ch_series_fall.set_data([], unit='', conversion=np.nan, resolution=np.nan)
                if len(curr_fall) == 0:
                    curr_fall = np.array([np.nan])
                    ch_series_fall.set_time(curr_fall)
                    ch_series_fall.set_value('num_samples', 0)
                else:
                    ch_series_fall.set_time(curr_fall)
                ch_series_fall.set_description('timestamps of fall cross of digital channel: ' + curr_chn)
                ch_series_fall.set_source('sync program')
                ch_series_fall.set_comments('digital')
                ch_series_fall.finalize()

        # add analog channels
        if 'analog_channels' in sync_dict.keys():

            analog_channels = sync_dict['analog_channels']
            analog_fs = sync_dict['analog_sample_rate']

            # get channel names
            for a_chn, a_ch in analog_channels.items():
                if ft.is_integer(a_chn):
                    curr_chn = 'analog_CH_' + ft.int2str(a_chn, 3)
                else:
                    curr_chn = 'analog_' + a_chn

                ch_series = self.create_timeseries('TimeSeries', curr_chn, 'acquisition')
                ch_series.set_data(a_ch, unit='voltage', conversion=1., resolution=1.)
                ch_series.set_time_by_rate(time_zero=0.0, rate=analog_fs)
                ch_series.set_value('num_samples', len(a_ch))
                ch_series.set_comments('continuous')
                ch_series.set_description('analog channel recorded by sync program')
                ch_series.set_source('sync program')
                ch_series.finalize()

    def add_acquisition_image(self, name, img, format='array', description=''):
        """
        add arbitrarily recorded image into acquisition group, mostly surface vasculature image
        :param name:
        :param img:
        :param format:
        :param description:
        :return:
        """
        img_dset = self.file_pointer['acquisition/images'].create_dataset(name, data=img)
        img_dset.attrs['format'] = format
        img_dset.attrs['description'] = description

    def get_analog_data(self, ch_n):
        """
        :param ch_n: string, analog channel name
        :return: 1-d array, analog data, data * conversion
                 1-d array, time stamps
        """
        grp = self.file_pointer['acquisition/timeseries'][ch_n]
        data = grp['data'][()]
        if not np.isnan(grp['data'].attrs['conversion']):
            data = data.astype(np.float32) * grp['data'].attrs['conversion']
        if 'timestamps' in grp.keys():
            t = grp['timestamps']
        elif 'starting_time' in grp.keys():
            fs = grp['starting_time'].attrs['rate']
            sample_num = grp['num_samples'][()]
            t = np.arange(sample_num) / fs + grp['starting_time'][()]
        else:
            raise ValueError('can not find timing information of channel:' + ch_n)
        return data, t

    def _check_display_order(self, display_order=None):
        """
        check display order make sure each presentation has a unique position, and move from increment order.
        also check the given display_order is of the next number
        """
        stimuli = self.file_pointer['stimulus/presentation'].keys()

        print('\nExisting visual stimuli:')
        print('\n'.join(stimuli))

        stimuli = [int(s[0:s.find('_')]) for s in stimuli]
        stimuli.sort()
        if stimuli != range(len(stimuli)):
            raise ValueError('display order is not incremental.')

        if display_order is not None:

            if display_order != len(stimuli):
                raise ValueError('input display order not the next display.')

    # ===========================photodiode related=====================================================================
    def add_photodiode_onsets(self, photodiode_ch_path='acquisition/timeseries/photodiode',
                              digitizeThr=0.9, filterSize=0.01, segmentThr=0.01, smallestInterval=0.03,
                              expected_onsets_number=None):
        """
        intermediate processing step for analysis of visual display. Containing the information about the onset of
        photodiode signal. Timestamps are extracted from photodiode signal, should be aligned to the master clock.
        extraction is done by NeuroAnalysisTools.HighLevel.segmentPhotodiodeSignal() function. The raw signal
        was first digitized by the digitize_threshold, then filtered by a gaussian fileter with filter_size. Then
        the derivative of the filtered signal was calculated by numpy.diff. The derivative signal was then timed
        with the digitized signal. Then the segmentation_threshold was used to detect rising edge of the resulting
        signal. Any onset with interval from its previous onset smaller than smallest_interval will be discarded.
        the resulting timestamps of photodiode onsets will be saved in 'analysis/photodiode_onsets' timeseries

        :param digitizeThr: float
        :param filterSize: float
        :param segmentThr: float
        :param smallestInterval: float
        :param expected_onsets_number: int, expected number of photodiode onsets, may extract from visual display
                                       log. if extracted onset number does not match this number, the process will
                                       be abort. If None, no such check will be performed.
        :return:
        """

        pd_grp = self.file_pointer[photodiode_ch_path]

        fs = pd_grp['starting_time'].attrs['rate']

        pd = pd_grp['data'][()] * pd_grp['data'].attrs['conversion']

        pd_onsets = hl.segmentPhotodiodeSignal(pd, digitizeThr=digitizeThr, filterSize=filterSize,
                                               segmentThr=segmentThr, Fs=fs, smallestInterval=smallestInterval)

        if pd_onsets.shape[0] == 0:
            return

        if expected_onsets_number is not None:
            if len(pd_onsets) != expected_onsets_number:
                raise ValueError('The number of photodiode onsets (' + str(len(pd_onsets)) + ') and the expected '
                                                                                             'number of sweeps ' + str(
                    expected_onsets_number) + ' do not match. Abort.')

        pd_ts = self.create_timeseries('TimeSeries', 'photodiode_onsets', modality='other')
        pd_ts.set_time(pd_onsets)
        pd_ts.set_data([], unit='', conversion=np.nan, resolution=np.nan)
        pd_ts.set_description('intermediate processing step for analysis of visual display. '
                              'Containing the information about the onset of photodiode signal. Timestamps '
                              'are extracted from photodiode signal, should be aligned to the master clock.'
                              'extraction is done by NeuroAnalysisTools.HighLevel.segmentPhotodiodeSignal()'
                              'function. The raw signal was first digitized by the digitize_threshold, then '
                              'filtered by a gaussian fileter with filter_size. Then the derivative of the filtered '
                              'signal was calculated by numpy.diff. The derivative signal was then timed with the '
                              'digitized signal. Then the segmentation_threshold was used to detect rising edge of '
                              'the resulting signal. Any onset with interval from its previous onset smaller than '
                              'smallest_interval will be discarded.')
        pd_ts.set_path('/analysis/PhotodiodeOnsets')
        pd_ts.set_value('digitize_threshold', digitizeThr)
        pd_ts.set_value('fileter_size', filterSize)
        pd_ts.set_value('segmentation_threshold', segmentThr)
        pd_ts.set_value('smallest_interval', smallestInterval)
        pd_ts.finalize()

    # ===========================photodiode related=====================================================================


    # ===========================ephys related==========================================================================
    # def add_open_ephys_data(self, folder, prefix, digital_channels=()):
    #     """
    #     add open ephys raw data to self, in acquisition group, less useful, because the digital events needs to be
    #     processed before added in
    #     :param folder: str, the folder contains open ephys raw data
    #     :param prefix: str, prefix of open ephys files
    #     :param digital_channels: list of str, digital channel
    #     :return:
    #     """
    #     output = oew.pack_folder_for_nwb(folder=folder, prefix=prefix, digital_channels=digital_channels)

    #     for key, value in output.items():

    #         if 'CH' in key:  # analog channel for electrode recording
    #             ch_ind = int(key[key.find('CH') + 2:])
    #             ch_name = 'ch_' + ft.int2str(ch_ind, 4)
    #             ch_trace = value['trace']
    #             ch_series = self.create_timeseries('ElectricalSeries', ch_name, 'acquisition')
    #             ch_series.set_data(ch_trace, unit='bit', conversion=float(value['header']['bitVolts']),
    #                                resolution=1.)
    #             ch_series.set_time_by_rate(time_zero=0.0,  # value['header']['start_time'],
    #                                        rate=float(value['header']['sampleRate']))
    #             ch_series.set_value('electrode_idx', ch_ind)
    #             ch_series.set_value('num_samples', len(ch_trace))
    #             ch_series.set_comments('continuous')
    #             ch_series.set_description('extracellular continuous voltage recording from tetrode')
    #             ch_series.set_source('open ephys')
    #             ch_series.finalize()

    #         elif key != 'events':  # other continuous channels
    #             ch_name = key[len(prefix) + 1:]
    #             ch_trace = value['trace']
    #             ch_series = self.create_timeseries('AbstractFeatureSeries', ch_name, 'acquisition')
    #             ch_series.set_data(ch_trace, unit='bit', conversion=float(value['header']['bitVolts']),
    #                                resolution=1.)
    #             ch_series.set_time_by_rate(time_zero=0.0,  # value['header']['start_time'],
    #                                        rate=float(value['header']['sampleRate']))
    #             ch_series.set_value('features', ch_name)
    #             ch_series.set_value('feature_units', 'bit')
    #             ch_series.set_value('num_samples', len(ch_trace))
    #             ch_series.set_value('help', 'continuously recorded analog channels with same sampling times as '
    #                                         'of electrode recordings')
    #             ch_series.set_comments('continuous')
    #             ch_series.set_description('continuous voltage recording from IO board')
    #             ch_series.set_source('open ephys')
    #             ch_series.finalize()

    #         else:  # digital events

    #             for key2, value2 in value.items():

    #                 ch_rise_ts = value2['rise']
    #                 ch_series_rise = self.create_timeseries('TimeSeries', key2 + '_rise', 'acquisition')
    #                 ch_series_rise.set_data([], unit='', conversion=np.nan, resolution=np.nan)
    #                 if len(ch_rise_ts) == 0:
    #                     ch_rise_ts = np.array([np.nan])
    #                     ch_series_rise.set_time(ch_rise_ts)
    #                     ch_series_rise.set_value('num_samples', 0)
    #                 else:
    #                     ch_series_rise.set_time(ch_rise_ts)
    #                 ch_series_rise.set_description('timestamps of rise cross of digital channel: ' + key2)
    #                 ch_series_rise.set_source('open ephys')
    #                 ch_series_rise.set_comments('digital')
    #                 ch_series_rise.finalize()

    #                 ch_fall_ts = value2['fall']
    #                 ch_series_fall = self.create_timeseries('TimeSeries', key2 + '_fall', 'acquisition')
    #                 ch_series_fall.set_data([], unit='', conversion=np.nan, resolution=np.nan)
    #                 if len(ch_fall_ts) == 0:
    #                     ch_fall_ts = np.array([np.nan])
    #                     ch_series_fall.set_time(ch_fall_ts)
    #                     ch_series_fall.set_value('num_samples', 0)
    #                 else:
    #                     ch_series_fall.set_time(ch_fall_ts)
    #                 ch_series_fall.set_description('timestamps of fall cross of digital channel: ' + key2)
    #                 ch_series_fall.set_source('open ephys')
    #                 ch_series_fall.set_comments('digital')
    #                 ch_series_fall.finalize()

    # def add_open_ephys_continuous_data(self, folder, prefix):
    #     """
    #     add open ephys raw continuous data to self, in acquisition group
    #     :param folder: str, the folder contains open ephys raw data
    #     :param prefix: str, prefix of open ephys files
    #     :param digital_channels: list of str, digital channel
    #     :return:
    #     """
    #     output = oew.pack_folder_for_nwb(folder=folder, prefix=prefix)

    #     for key, value in output.items():

    #         if 'CH' in key:  # analog channel for electrode recording
    #             ch_ind = int(key[key.find('CH') + 2:])
    #             ch_name = 'ch_' + ft.int2str(ch_ind, 4)
    #             ch_trace = value['trace']
    #             ch_series = self.create_timeseries('ElectricalSeries', ch_name, 'acquisition')
    #             ch_series.set_data(ch_trace, unit='bit', conversion=float(value['header']['bitVolts']),
    #                                resolution=1.)
    #             ch_series.set_time_by_rate(time_zero=0.0,  # value['header']['start_time'],
    #                                        rate=float(value['header']['sampleRate']))
    #             ch_series.set_value('electrode_idx', ch_ind)
    #             ch_series.set_value('num_samples', len(ch_trace))
    #             ch_series.set_comments('continuous')
    #             ch_series.set_description('extracellular continuous voltage recording from tetrode')
    #             ch_series.set_source('open ephys')
    #             ch_series.finalize()

    #         elif key != 'events':  # other continuous channels
    #             ch_name = key[len(prefix) + 1:]
    #             ch_trace = value['trace']
    #             ch_series = self.create_timeseries('AbstractFeatureSeries', ch_name, 'acquisition')
    #             ch_series.set_data(ch_trace, unit='bit', conversion=float(value['header']['bitVolts']),
    #                                resolution=1.)
    #             ch_series.set_time_by_rate(time_zero=0.0,  # value['header']['start_time'],
    #                                        rate=float(value['header']['sampleRate']))
    #             ch_series.set_value('features', ch_name)
    #             ch_series.set_value('feature_units', 'bit')
    #             ch_series.set_value('num_samples', len(ch_trace))
    #             ch_series.set_value('help', 'continuously recorded analog channels with same sampling times as '
    #                                         'of electrode recordings')
    #             ch_series.set_comments('continuous')
    #             ch_series.set_description('continuous voltage recording from IO board')
    #             ch_series.set_source('open ephys')
    #             ch_series.finalize()

    # def add_phy_template_clusters(self, folder, module_name, ind_start=None, ind_end=None,
    #                               is_add_artificial_unit=False, artificial_unit_firing_rate=2.,
    #                               spike_sorter=None):
    #     """
    #     extract phy-template clustering results to nwb format. Only extract spike times, no template for now.
    #     Usually the continuous channels of multiple files are concatenated for kilosort. ind_start and ind_end are
    #     Used to extract the data of this particular file.

    #     :param folder: folder containing phy template results.
    #                    expects cluster_groups.csv, spike_clusters.npy and spike_times.npy in the folder.
    #     :param module_name: str, name of clustering module group
    #     :param ind_start: int, the start index of continuous channel of the current file in the concatenated file.
    #     :param ind_end: int, the end index of continuous channel of the current file in the concatenated file.
    #     :param is_add_artificial_unit: bool, if True: a artificial unit with possion event will be added, this unit
    #                                    will have name 'aua' and refractory period 1 ms.
    #     :param artificial_unit_firing_rate: float, firing rate of the artificial unit
    #     :param spike_sorter: string, user ID of who manually sorted the clusters
    #     :return:
    #     """

    #     #  check integrity
    #     if ind_start == None:
    #         ind_start = 0

    #     if ind_end == None:
    #         ind_end = self.file_pointer['acquisition/timeseries/photodiode/num_samples'].value

    #     if ind_start >= ind_end:
    #         raise ValueError('ind_end should be larger than ind_start.')

    #     try:
    #         fs = self.file_pointer['general/extracellular_ephys/sampling_rate'].value
    #     except KeyError:
    #         print('\nCannot find "general/extracellular_ephys/sampling_rate" field. Abort process.')
    #         return

    #     #  get spike sorter
    #     if spike_sorter is None:
    #         spike_sorter = self.file_pointer['general/experimenter'].value

    #     #  set kilosort output file paths
    #     clusters_path = os.path.join(folder, 'spike_clusters.npy')
    #     spike_times_path = os.path.join(folder, 'spike_times.npy')

    #     #  generate dictionary of cluster timing indices
    #     try:
    #         #  for new version of kilosort
    #         phy_template_output = kw.get_clusters(kw.read_csv(os.path.join(folder, 'cluster_group.tsv')))
    #     except IOError:
    #         # for old version of kilosort
    #         phy_template_output = kw.get_clusters(kw.read_csv(os.path.join(folder, 'cluster_groups.csv')))

    #     spike_ind = kw.get_spike_times_indices(phy_template_output, spike_clusters_path=clusters_path,
    #                                            spike_times_path=spike_times_path)

    #     #  add artificial random unit
    #     if is_add_artificial_unit:
    #         file_length = (ind_end - ind_start) / fs
    #         au_ts = ta.possion_event_ts(duration=file_length, firing_rate=artificial_unit_firing_rate,
    #                                     refractory_dur=0.001, is_plot=False)
    #         spike_ind.update({'unit_aua': (au_ts * fs).astype(np.uint64) + ind_start})

    #     #  get channel related infomation
    #     ch_ns = self._get_channel_names()
    #     file_starting_time = self.get_analog_data(ch_ns[0])[1][0]
    #     channel_positions = kw.get_channel_geometry(folder, channel_names=ch_ns)

    #     #  create specificed module
    #     mod = self.create_module(name=module_name)
    #     mod.set_description('phy-template manual clustering after kilosort')
    #     mod.set_value('channel_list', [ch.encode('Utf8') for ch in ch_ns])
    #     mod.set_value('channel_xpos', [channel_positions[ch][0] for ch in ch_ns])
    #     mod.set_value('channel_ypos', [channel_positions[ch][1] for ch in ch_ns])

    #     #  create UnitTimes interface
    #     unit_times = mod.create_interface('UnitTimes')
    #     for unit in spike_ind.keys():

    #         #  get timestamps of current unit
    #         curr_ts = spike_ind[unit]
    #         curr_ts = curr_ts[np.logical_and(curr_ts >= ind_start, curr_ts < ind_end)] - ind_start
    #         curr_ts = curr_ts / fs + file_starting_time

    #         # array to store waveforms from all channels
    #         template = []
    #         template_uf = []  # wavefrom from unfiltered analog signal

    #         # array to store standard deviations of waveform from all channels
    #         std = []
    #         std_uf = []  # standard deviations of unfiltered analog signal

    #         # temporary variables to detect peak channels
    #         peak_channel = None
    #         peak_channel_ind = None
    #         peak_amp = 0

    #         for i, ch_n in enumerate(ch_ns):

    #             #  get current analog signals of a given channel
    #             curr_ch_data, curr_ch_ts = self.get_analog_data(ch_n)

    #             #  band pass this analog signal
    #             curr_ch_data_f = ta.butter_highpass(curr_ch_data, cutoff=300., fs=fs)

    #             #  calculate spike triggered average filtered signal
    #             curr_waveform_results = ta.event_triggered_average_regular(ts_event=curr_ts,
    #                                                                        continuous=curr_ch_data_f,
    #                                                                        fs_continuous=fs,
    #                                                                        start_time_continuous=file_starting_time,
    #                                                                        t_range=SPIKE_WAVEFORM_TIMEWINDOW,
    #                                                                        is_normalize=True,
    #                                                                        is_plot=False)
    #             curr_waveform, curr_n, curr_t, curr_std = curr_waveform_results

    #             curr_waveform_results_uf = ta.event_triggered_average_regular(ts_event=curr_ts,
    #                                                                           continuous=curr_ch_data,
    #                                                                           fs_continuous=fs,
    #                                                                           start_time_continuous=file_starting_time,
    #                                                                           t_range=SPIKE_WAVEFORM_TIMEWINDOW,
    #                                                                           is_normalize=True,
    #                                                                           is_plot=False)
    #             curr_waveform_uf, _, _, curr_std_uf = curr_waveform_results_uf

    #             #  append waveform and std for current channel
    #             template.append(curr_waveform)
    #             std.append(curr_std)
    #             template_uf.append(curr_waveform_uf)
    #             std_uf.append(curr_std_uf)

    #             #  detect the channel with peak amplitude
    #             if peak_channel is not None:
    #                 peak_channel = ch_n
    #                 peak_channel_ind = i
    #                 peak_amp = np.max(curr_waveform) - np.min(curr_waveform)
    #             else:
    #                 if np.max(curr_waveform) - np.min(curr_waveform) > peak_amp:
    #                     peak_channel = ch_n
    #                     peak_channel_ind = i
    #                     peak_amp = np.max(curr_waveform) - np.min(curr_waveform)

    #         #  add 'UnitTimes' field
    #         if unit == 'unit_aua':
    #             unit_times.add_unit(unit_name='unit_aua', unit_times=curr_ts,
    #                                 source='electrophysiology extracellular recording',
    #                                 description='Artificial possion unit for control. Spike time unit: seconds. '
    #                                             'Spike waveforms are band-pass filtered with cutoff frequency'
    #                                             ' (300, 6000) Hz.')
    #         else:
    #             unit_times.add_unit(unit_name=unit, unit_times=curr_ts,
    #                                 source='electrophysiology extracellular recording',
    #                                 description="Data spike-sorted by: " + spike_sorter +
    #                                             ' using phy-template. Spike time unit: seconds. Spike waveforms are'
    #                                             'band-pass filtered with cutoff frequency (300, 6000) Hz.')

    #         #  add relevant information to current UnitTimes field
    #         unit_times.append_unit_data(unit_name=unit, key='apply_channel_name', value=peak_channel)
    #         unit_times.append_unit_data(unit_name=unit, key='channel', value=peak_channel_ind)
    #         unit_times.append_unit_data(unit_name=unit, key='template_filtered', value=np.array(template).transpose())
    #         unit_times.append_unit_data(unit_name=unit, key='template',
    #                                     value=np.array(template_uf).transpose())
    #         unit_times.append_unit_data(unit_name=unit, key='template_std_filtered', value=np.array(std).transpose())
    #         unit_times.append_unit_data(unit_name=unit, key='template_std',
    #                                     value=np.array(std_uf).transpose())
    #         unit_times.append_unit_data(unit_name=unit, key='waveform', value=template_uf[peak_channel_ind])
    #         unit_times.append_unit_data(unit_name=unit, key='waveform_std', value=std_uf[peak_channel_ind])
    #         unit_times.append_unit_data(unit_name=unit, key='xpos_probe',
    #                                     value=[channel_positions[ch][0] for ch in ch_ns][peak_channel_ind])
    #         unit_times.append_unit_data(unit_name=unit, key='ypos_probe',
    #                                     value=[channel_positions[ch][1] for ch in ch_ns][peak_channel_ind])

    #     #  finalize
    #     unit_times.finalize()
    #     mod.finalize()

    # def add_external_LFP(self, traces, fs=30000., module_name=None, notch_base=60., notch_bandwidth=1.,
    #                      notch_harmonics=4,
    #                      notch_order=2, lowpass_cutoff=300., lowpass_order=5, resolution=0, conversion=0, unit='',
    #                      comments='', source=''):
    #     """
    #     add LFP of raw arbitrary electrical traces into LFP module into /procession field. the trace will be filtered
    #     by NeuroAnalysisTools.HighLevel.get_lfp() function. All filters are butterworth digital filters

    #     :param module_name: str, name of module to be added
    #     :param traces: dict, {str: 1d-array}, {name: trace}, input raw traces
    #     :param fs: float, sampling rate, Hz
    #     :param notch_base: float, Hz, base frequency of powerline contaminating signal
    #     :param notch_bandwidth: float, Hz, filter bandwidth at each side of center frequency
    #     :param notch_harmonics: int, number of harmonics to filter out
    #     :param notch_order: int, order of butterworth bandpass notch filter, for a narrow band, shouldn't be larger than 2
    #     :param lowpass_cutoff: float, Hz, cutoff frequency of lowpass filter
    #     :param lowpass_order: int, order of butterworth lowpass filter
    #     :param resolution: float, resolution of LFP time series
    #     :param conversion: float, conversion of LFP time series
    #     :param unit: str, unit of LFP time series
    #     :param comments: str, interface comments
    #     :param source: str, interface source
    #     """

    #     if module_name is None or module_name == '':
    #         module_name = 'external_LFP'

    #     lfp = {}
    #     for tn, trace in traces.items():
    #         curr_lfp = hl.get_lfp(trace, fs=fs, notch_base=notch_base, notch_bandwidth=notch_bandwidth,
    #                               notch_harmonics=notch_harmonics, notch_order=notch_order,
    #                               lowpass_cutoff=lowpass_cutoff, lowpass_order=lowpass_order)
    #         lfp.update({tn: curr_lfp})

    #     lfp_mod = self.create_module(module_name)
    #     lfp_mod.set_description('LFP from external traces')
    #     lfp_interface = lfp_mod.create_interface('LFP')
    #     lfp_interface.set_value('description', 'LFP of raw arbitrary electrical traces. The traces were filtered by '
    #                                            'NeuroAnalysisTools.HighLevel.get_lfp() function. First, the powerline contamination at '
    #                                            'multiplt harmonics were filtered out by a notch filter. Then the resulting traces were'
    #                                            ' filtered by a lowpass filter. All filters are butterworth digital filters')
    #     lfp_interface.set_value('comments', comments)
    #     lfp_interface.set_value('notch_base', notch_base)
    #     lfp_interface.set_value('notch_bandwidth', notch_bandwidth)
    #     lfp_interface.set_value('notch_harmonics', notch_harmonics)
    #     lfp_interface.set_value('notch_order', notch_order)
    #     lfp_interface.set_value('lowpass_cutoff', lowpass_cutoff)
    #     lfp_interface.set_value('lowpass_order', lowpass_order)
    #     lfp_interface.set_source(source)
    #     for tn, t_lfp in lfp.items():
    #         curr_ts = self.create_timeseries('ElectricalSeries', tn, modality='other')
    #         curr_ts.set_data(t_lfp, conversion=conversion, resolution=resolution, unit=unit)
    #         curr_ts.set_time_by_rate(time_zero=0., rate=fs)
    #         curr_ts.set_value('num_samples', len(t_lfp))
    #         curr_ts.set_value('electrode_idx', 0)
    #         lfp_interface.add_timeseries(curr_ts)
    #         lfp_interface.finalize()

    #     lfp_mod.finalize()

    # def add_internal_LFP(self, continuous_channels, module_name=None, notch_base=60., notch_bandwidth=1.,
    #                      notch_harmonics=4, notch_order=2, lowpass_cutoff=300., lowpass_order=5, comments='',
    #                      source=''):
    #     """
    #     add LFP of acquired electrical traces into LFP module into /procession field. the trace will be filtered
    #     by NeuroAnalysisTools.HighLevel.get_lfp() function. All filters are butterworth digital filters.

    #     :param continuous_channels: list of strs, name of continuous channels saved in '/acquisition/timeseries'
    #                                 folder, the time axis of these channels should be saved by rate
    #                                 (ephys sampling rate).
    #     :param module_name: str, name of module to be added
    #     :param notch_base: float, Hz, base frequency of powerline contaminating signal
    #     :param notch_bandwidth: float, Hz, filter bandwidth at each side of center frequency
    #     :param notch_harmonics: int, number of harmonics to filter out
    #     :param notch_order: int, order of butterworth bandpass notch filter, for a narrow band, shouldn't be larger than 2
    #     :param lowpass_cutoff: float, Hz, cutoff frequency of lowpass filter
    #     :param lowpass_order: int, order of butterworth lowpass filter
    #     :param comments: str, interface comments
    #     :param source: str, interface source
    #     """

    #     if module_name is None or module_name == '':
    #         module_name = 'LFP'

    #     lfp_mod = self.create_module(module_name)
    #     lfp_mod.set_description('LFP from acquired electrical traces')
    #     lfp_interface = lfp_mod.create_interface('LFP')
    #     lfp_interface.set_value('description', 'LFP of acquired electrical traces. The traces were filtered by '
    #                                            'NeuroAnalysisTools.HighLevel.get_lfp() function. First, the powerline '
    #                                            'contamination at multiplt harmonics were filtered out by a notch '
    #                                            'filter. Then the resulting traces were filtered by a lowpass filter. '
    #                                            'All filters are butterworth digital filters')
    #     lfp_interface.set_value('comments', comments)
    #     lfp_interface.set_value('notch_base', notch_base)
    #     lfp_interface.set_value('notch_bandwidth', notch_bandwidth)
    #     lfp_interface.set_value('notch_harmonics', notch_harmonics)
    #     lfp_interface.set_value('notch_order', notch_order)
    #     lfp_interface.set_value('lowpass_cutoff', lowpass_cutoff)
    #     lfp_interface.set_value('lowpass_order', lowpass_order)
    #     lfp_interface.set_source(source)

    #     for channel in continuous_channels:
    #         print('\n', channel, ': start adding LFP ...')

    #         trace = self.file_pointer['acquisition/timeseries'][channel]['data'].value
    #         fs = self.file_pointer['acquisition/timeseries'][channel]['starting_time'].attrs['rate']
    #         start_time = self.file_pointer['acquisition/timeseries'][channel]['starting_time'].value
    #         conversion = self.file_pointer['acquisition/timeseries'][channel]['data'].attrs['conversion']
    #         resolution = self.file_pointer['acquisition/timeseries'][channel]['data'].attrs['resolution']
    #         unit = self.file_pointer['acquisition/timeseries'][channel]['data'].attrs['unit']
    #         ts_source = self.file_pointer['acquisition/timeseries'][channel].attrs['source']

    #         print(channel, ': calculating LFP ...')

    #         t_lfp = hl.get_lfp(trace, fs=fs, notch_base=notch_base, notch_bandwidth=notch_bandwidth,
    #                            notch_harmonics=notch_harmonics, notch_order=notch_order, lowpass_cutoff=lowpass_cutoff,
    #                            lowpass_order=lowpass_order)

    #         curr_ts = self.create_timeseries('ElectricalSeries', channel, modality='other')
    #         curr_ts.set_data(t_lfp, conversion=conversion, resolution=resolution, unit=unit)
    #         curr_ts.set_time_by_rate(time_zero=start_time, rate=fs)
    #         curr_ts.set_value('num_samples', len(t_lfp))
    #         curr_ts.set_value('electrode_idx', int(channel.split('_')[1]))
    #         curr_ts.set_source(ts_source)
    #         lfp_interface.add_timeseries(curr_ts)
    #         print(channel, ': finished adding LFP.')

    #     lfp_interface.finalize()

    #     lfp_mod.finalize()

    # def plot_spike_waveforms(self, modulen, unitn, is_plot_filtered=False, fig=None, axes_size=(0.2, 0.2), **kwargs):
    #     """
    #     plot spike waveforms

    #     :param modulen: str, name of the module containing ephys recordings
    #     :param unitn: str, name of ephys unit, should be in '/processing/ephys_units/UnitTimes'
    #     :param is_plot_filtered: bool, plot unfiltered waveforms or not
    #     :param channel_names: list of strs, channel names in continuous recordings, should be in '/acquisition/timeseries'
    #     :param fig: matplotlib figure object
    #     :param t_range: tuple of two floats, time range to plot along spike time stamps
    #     :param kwargs: inputs to matplotlib.axes.plot() function
    #     :return: fig
    #     """
    #     if modulen not in self.file_pointer['processing'].keys():
    #         raise LookupError('Can not find module for ephys recording: ' + modulen + '.')

    #     if unitn not in self.file_pointer['processing'][modulen]['UnitTimes'].keys():
    #         raise LookupError('Can not find ephys unit: ' + unitn + '.')

    #     ch_ns = self._get_channel_names()

    #     unit_grp = self.file_pointer['processing'][modulen]['UnitTimes'][unitn]
    #     waveforms = unit_grp['template'].value

    #     if 'template_std' in unit_grp.keys():
    #         stds = unit_grp['template_std'].value
    #     else:
    #         stds = None

    #     if is_plot_filtered:
    #         if 'template_filtered' in unit_grp.keys():
    #             waveforms_f = unit_grp['template_filtered'].value
    #             if 'template_std_filtered' in unit_grp.keys():
    #                 stds_f = unit_grp['template_std_filtered'].value
    #             else:
    #                 stds_f = None
    #         else:
    #             print('can not find unfiltered spike waveforms for unit: ' + unitn)
    #             waveforms_f = None
    #             stds_f = None
    #     else:
    #         waveforms_f = None
    #         stds_f = None

    #     if 'channel_xpos' in self.file_pointer['processing'][modulen].keys():
    #         ch_xpos = self.file_pointer['processing'][modulen]['channel_xpos']
    #         ch_ypos = self.file_pointer['processing'][modulen]['channel_ypos']
    #         ch_locations = zip(ch_xpos, ch_ypos)
    #     else:
    #         ch_locations = None

    #     fig = plot_waveforms(waveforms, ch_locations=ch_locations, stds=stds, waveforms_filtered=waveforms_f,
    #                          stds_filtered=stds_f, f=fig, ch_ns=ch_ns, axes_size=axes_size, **kwargs)

    #     fig.suptitle(self.file_pointer['identifier'].value + ' : ' + unitn)

    #     return fig

    # def generate_dat_file_for_kilosort(self, output_folder, output_name, ch_ns, is_filtered=True, cutoff_f_low=300.,
    #                                    cutoff_f_high=6000.):
    #     """
    #     generate .dat file for kilolsort: "https://github.com/cortex-lab/KiloSort", it is binary raw code, with
    #     structure: ch0_t0, ch1_t0, ch2_t0, ...., chn_t0, ch0_t1, ch1_t1, ch2_t1, ..., chn_t1, ..., ch0_tm, ch1_tm,
    #     ch2_tm, ..., chn_tm

    #     :param output_folder: str, path to output directory
    #     :param output_name: str, output file name, an extension of '.dat' will be automatically added.
    #     :param ch_ns: list of strings, name of included analog channels
    #     :param is_filtered: bool, if Ture, another .dat file with same size will be generated in the output folder.
    #                         this file will contain temporally filtered data (filter done by
    #                         NeuroAnalysisTools.core.TimingAnalysis.butter_... functions). '_filtered' will be attached
    #                         to the filtered file name.
    #     :param cutoff_f_low: float, low cutoff frequency, Hz. if None, it will be low-pass
    #     :param cutoff_f_high: float, high cutoff frequency, Hz, if None, it will be high-pass
    #     :return: None
    #     """

    #     save_path = os.path.join(output_folder, output_name + '.dat')
    #     if os.path.isfile(save_path):
    #         raise IOError('Output file already exists.')

    #     data_lst = []
    #     for ch_n in ch_ns:
    #         data_lst.append(self.file_pointer['acquisition/timeseries'][ch_n]['data'].value)

    #     dtype = data_lst[0].dtype
    #     data = np.array(data_lst, dtype=dtype).flatten(order='F')
    #     data.tofile(save_path)

    #     if is_filtered:

    #         if cutoff_f_low is None and cutoff_f_high is None:
    #             print ('both low cutoff frequency and high cutoff frequency are None. Do nothing.')
    #             return

    #         save_path_f = os.path.join(output_folder, output_name + '_filtered.dat')
    #         if os.path.isfile(save_path_f):
    #             raise IOError('Output file for filtered data already existes.')

    #         fs = self.file_pointer['general/extracellular_ephys/sampling_rate'].value
    #         data_lst_f = []
    #         for data_r in data_lst:
    #             if cutoff_f_high is None:
    #                 data_lst_f.append(ta.butter_lowpass(data_r, fs=fs, cutoff=cutoff_f_low).astype(dtype))
    #             elif cutoff_f_low is None:
    #                 data_lst_f.append(ta.butter_highpass(data_r, fs=fs, cutoff=cutoff_f_high).astype(dtype))
    #             else:
    #                 data_lst_f.append(ta.butter_bandpass(data_r,
    #                                                      fs=fs,
    #                                                      cutoffs=(cutoff_f_low, cutoff_f_high)).astype(dtype))
    #         data_f = np.array(data_lst_f, dtype=dtype).flatten(order='F')
    #         data_f.tofile(save_path_f)

    # def _get_channel_names(self):
    #     """
    #     :return: sorted list of channel names, each channel name should have prefix 'ch_'
    #     """
    #     analog_chs = self.file_pointer['acquisition/timeseries'].keys()
    #     channel_ns = [cn for cn in analog_chs if cn[0:3] == 'ch_']
    #     channel_ns.sort()
    #     return channel_ns

    # # ===========================ephys related==========================================================================


    # ===========================2p movie related=======================================================================
    def add_acquired_image_series_as_remote_link(self, name, image_file_path, dataset_path, timestamps,
                                                 description='', comments='', data_format='zyx', pixel_size=np.nan,
                                                 pixel_size_unit=''):
        """
        add a required image series in to acquisition field as a link to an external hdf5 file.
        :param name: str, name of the image series
        :param image_file_path: str, the full file system path to the hdf5 file containing the raw image data
        :param dataset_path: str, the path within the hdf5 file pointing to the raw data. the object should have at
                             least 3 attributes: 'conversion', resolution, unit
        :param timestamps: 1-d array, the length of this array should be the same as number of frames in the image data
        :param data_format: str, required field for ImageSeries object
        :param pixel_size: array, size of pixel
        :param pixel_size_unit: str, unit of pixel size
        :return:
        """

        img_file = h5py.File(image_file_path, 'r')
        img_data = img_file[dataset_path]
        if timestamps.shape[0] != img_data.shape[0]:
            raise ValueError('Number of frames does not equal to the length of timestamps!')
        img_series = self.create_timeseries(ts_type='ImageSeries', name=name, modality='acquisition')
        img_series.set_data_as_remote_link(image_file_path, dataset_path)
        img_series.set_time(timestamps)
        img_series.set_description(description)
        img_series.set_comments(comments)
        img_series.set_value('bits_per_pixel', img_data.dtype.itemsize * 8)
        img_series.set_value('format', data_format)
        img_series.set_value('dimension', img_data.shape)
        img_series.set_value('image_file_path', image_file_path)
        img_series.set_value('image_data_path_within_file', dataset_path)
        img_series.set_value('pixel_size', pixel_size)
        img_series.set_value('pixel_size_unit', pixel_size_unit)
        img_series.finalize()

    def add_motion_correction_module(self, module_name, original_timeseries_path, corrected_file_path,
                                     corrected_dataset_path, xy_translation_offsets, interface_name='MotionCorrection',
                                     mean_projection=None, max_projection=None, description='', comments='',
                                     source=''):
        """
        add a motion corrected image series in to processing field as a module named 'motion_correction' and create a
        link to an external hdf5 file which contains the images.
        :param module_name: str, module name to be created
        :param interface_name: str, interface name of the image series
        :param original_timeseries_path: str, the path to the timeseries of the original images
        :param corrected_file_path: str, the full file system path to the hdf5 file containing the raw image data
        :param corrected_dataset_path: str, the path within the hdf5 file pointing to the motion corrected data.
                                       the object should have at least 3 attributes: 'conversion', resolution, unit
        :param xy_translation_offsets: 2d array with two columns,
        :param mean_projection: 2d array, mean_projection of corrected image, if None, no dataset will be
                                created
        :param max_projection: 2d array,  max_projection of corrected image, if None, no dataset will be
                                created
        :return:
        """

        orig = self.file_pointer[original_timeseries_path]
        timestamps = orig['timestamps'].value

        img_file = h5py.File(corrected_file_path)
        img_data = img_file[corrected_dataset_path]
        if timestamps.shape[0] != img_data.shape[0]:
            raise ValueError('Number of frames does not equal to the length of timestamps!')

        if xy_translation_offsets.shape[0] != timestamps.shape[0]:
            raise ValueError('Number of offsets does not equal to the length of timestamps!')

        corrected = self.create_timeseries(ts_type='ImageSeries', name='corrected', modality='other')
        corrected.set_data_as_remote_link(corrected_file_path, corrected_dataset_path)
        corrected.set_time_as_link(original_timeseries_path)
        corrected.set_description(description)
        corrected.set_comments(comments)
        corrected.set_source(source)
        for value_n in orig.keys():
            if value_n not in ['image_data_path_within_file', 'image_file_path', 'data', 'timestamps']:
                corrected.set_value(value_n, orig[value_n].value)

        xy_translation = self.create_timeseries(ts_type='TimeSeries', name='xy_translation', modality='other')
        xy_translation.set_data(xy_translation_offsets, unit='pixel', conversion=np.nan, resolution=np.nan)
        xy_translation.set_time_as_link(original_timeseries_path)
        xy_translation.set_value('num_samples', xy_translation_offsets.shape[0])
        xy_translation.set_description('Time series of x, y shifts applied to create motion stabilized image series')
        xy_translation.set_value('feature_description', ['x_motion', 'y_motion'])

        mc_mod = self.create_module(module_name)
        mc_interf = mc_mod.create_interface("MotionCorrection")
        mc_interf.add_corrected_image(interface_name, orig=original_timeseries_path, xy_translation=xy_translation,
                                      corrected=corrected)

        if mean_projection is not None:
            mc_interf.set_value('mean_projection', mean_projection)

        if max_projection is not None:
            mc_interf.set_value('max_projection', max_projection)

        mc_interf.finalize()
        mc_mod.finalize()

    def add_muliple_dataset_to_motion_correction_module(self, input_parameters, module_name='motion_correction',
                                                        temporal_downsample_rate=1):
        """
        add multiple motion corrected datasets into a motion correction module. Designed for adding multiplane
        imaging datasets at once. The motion correction module will contain multiple interfaces each corresponding
        to one imaging plane.

        :param input_parameters: list of dictionaries, each dictionary in the list represents one imaging plane
                                 the dictionary should contain the following keys:
                                 'field_name': str, name of the hdf5 group for the motion correction information
                                 'original_timeseries_path': str, the path to the timeseries of the original images
                                 'corrected_file_path': str, the full file system path to the hdf5 file
                                                        containing the corrected image data
                                 'corrected_dataset_path': str, the path within the hdf5 file pointing to the motion
                                                           corrected data. the object should have at least 3
                                                           attributes: 'conversion', resolution and unit
                                 'xy_translation_offsets': 2d array with two columns
                                 'mean_projection': optional, 2d array, mean_projection of corrected image,
                                                    if not existing, no dataset will be created
                                 'max_projection': optional, 2d array, max_projection of corrected image,
                                                    if not existing, no dataset will be created
                                 'description': optional, str, if not existing, it will be set as ''
                                 'comments': optional, str, if not existing, it will be set as ''
                                 'source': optional, str, if not existing, it will be set as ''
        :param module_name: str, module name to be created
        :param temporal_downsample_rate: int, >0, in case the movie was motion corrected before temporal downsample,
                                         use only a subset of offsets.
        """

        mc_mod = self.create_module(module_name)
        mc_interf = mc_mod.create_interface('MotionCorrection')

        for mov_dict in input_parameters:
            if 'description' not in mov_dict.keys():
                mov_dict['description'] = ''

            if 'comments' not in mov_dict.keys():
                mov_dict['comment'] = ''

            if 'source' not in mov_dict.keys():
                mov_dict['source'] = ''

            orig = self.file_pointer[mov_dict['original_timeseries_path']]
            timestamps = orig['timestamps'][()]
            # print(timestamps.shape)

            img_file = h5py.File(mov_dict['corrected_file_path'], 'r')
            img_data = img_file[mov_dict['corrected_dataset_path']]
            # print(img_data.shape)
            if timestamps.shape[0] != img_data.shape[0]:
                raise ValueError('Number of frames does not equal to the length of timestamps!')

            offsets = mov_dict['xy_translation_offsets']
            offsets = offsets[::temporal_downsample_rate, :]
            # print(offsets.shape)
            if offsets.shape[0] != timestamps.shape[0]:
                raise ValueError('Number of offsets does not equal to the length of timestamps!')

            corrected = self.create_timeseries(ts_type='ImageSeries', name='corrected', modality='other')
            corrected.set_data_as_remote_link(mov_dict['corrected_file_path'],
                                              mov_dict['corrected_dataset_path'])
            corrected.set_time_as_link(mov_dict['original_timeseries_path'])
            corrected.set_description(mov_dict['description'])
            corrected.set_comments(mov_dict['comments'])
            corrected.set_source(mov_dict['source'])

            if 'mean_projection' in mov_dict.keys() and mov_dict['mean_projection'] is not None:
                corrected.set_value('mean_projection', mov_dict['mean_projection'])

            if 'max_projection' in mov_dict.keys() and mov_dict['max_projection'] is not None:
                corrected.set_value('max_projection', mov_dict['max_projection'])

            for value_n in orig.keys():
                if value_n not in ['image_data_path_within_file', 'image_file_path', 'data', 'timestamps']:
                    if isinstance(orig[value_n][()], bytes):
                        corrected.set_value(value_n, orig[value_n][()].decode())
                    else:
                        corrected.set_value(value_n, orig[value_n][()])

            xy_translation = self.create_timeseries(ts_type='TimeSeries', name='xy_translation', modality='other')
            xy_translation.set_data(offsets, unit='pixel', conversion=np.nan,
                                    resolution=np.nan)
            xy_translation.set_time_as_link(mov_dict['original_timeseries_path'])
            xy_translation.set_value('num_samples', offsets.shape[0])
            xy_translation.set_description('Time series of x, y shifts applied to create motion '
                                           'stabilized image series')
            xy_translation.set_value('feature_description', [b'x_motion', b'y_motion'])

            mc_interf.add_corrected_image(mov_dict['field_name'], orig=mov_dict['original_timeseries_path'],
                                          xy_translation=xy_translation,
                                          corrected=corrected)

        mc_interf.finalize()
        mc_mod.finalize()

    # ===========================2p movie related=======================================================================


    # ===========================camstim visual stimuli related=========================================================
    def add_display_frame_ts_camstim(self, pkl_dict, max_mismatch=0.1, verbose=True, refresh_rate=60.,
                                     allowed_jitter=0.01):

        ts_pd_fall = self.file_pointer['acquisition/timeseries/digital_photodiode_fall/timestamps'][()]
        ts_display_rise = self.file_pointer['acquisition/timeseries/digital_vsync_visual_rise/timestamps'][()]

        ts_display_real, display_lag = ct.align_visual_display_time(pkl_dict=pkl_dict, ts_pd_fall=ts_pd_fall,
                                                                    ts_display_rise=ts_display_rise,
                                                                    max_mismatch=max_mismatch,
                                                                    verbose=verbose, refresh_rate=refresh_rate,
                                                                    allowed_jitter=allowed_jitter)

        frame_ts = self.create_timeseries('TimeSeries', 'FrameTimestamps', modality='other')
        frame_ts.set_time(ts_display_rise)
        frame_ts.set_data([], unit='', conversion=np.nan, resolution=np.nan)
        frame_ts.set_description('onset timestamps of each display frames after correction for display lag. '
                                 'Used NeuroAnalysisTools.HighLevel.align_visual_display_time() function to '
                                 'calculate display lag.')
        frame_ts.set_path('/processing/visual_display')
        frame_ts.set_value('max_mismatch_sec', max_mismatch)
        frame_ts.set_value('refresh_rate_hz', refresh_rate)
        frame_ts.set_value('allowed_jitter_sec', allowed_jitter)
        frame_ts.finalize()

        display_lag_ts = self.create_timeseries('TimeSeries', 'DisplayLag', modality='other')
        display_lag_ts.set_time(display_lag[:, 0])
        display_lag_ts.set_data(display_lag[:, 1], unit='second', conversion=np.nan, resolution=np.nan)
        display_lag_ts.set_path('/processing/visual_display')
        display_lag_ts.set_value('mean_display_lag_sec', np.mean(display_lag[:, 1]))
        display_lag_ts.finalize()

    def _add_drifting_grating_stimulation_camstim(self, stim_dict):

        dgts = self.create_timeseries(ts_type='TimeSeries',
                                      name=stim_dict['stim_name'],
                                      modality='stimulus')

        dgts.set_time(stim_dict['sweep_onset_frames'])
        dgts.set_data(stim_dict['sweeps'], unit='', conversion=np.nan, resolution=np.nan)
        dgts.set_source(stim_dict['source'])
        dgts.set_comments(stim_dict['comments'])
        dgts.set_description(stim_dict['description'])
        for fn, fv in stim_dict.items():
            if fn not in ['sweep_onset_frames', 'sweeps', 'sources', 'comments', 'description']:
                dgts.set_value(fn, fv)
        dgts.finalize()

    def _add_locally_sparse_noise_stimulation_camstim(self, stim_dict):

        lsnts = self.create_timeseries(ts_type='TimeSeries',
                                       name=stim_dict['stim_name'],
                                       modality='stimulus')
        lsnts.set_time(stim_dict['global_frame_ind'])
        lsnts.set_data(stim_dict['probes'], unit='', conversion=np.nan, resolution=np.nan)
        lsnts.set_source(stim_dict['source'])
        lsnts.set_comments(stim_dict['comments'])
        lsnts.set_description(stim_dict['description'])
        for fn, fv in stim_dict.items():
            if fn not in ['probes', 'global_frame_ind', 'sources', 'comments', 'description']:
                lsnts.set_value(fn, fv)
        lsnts.finalize()

    def add_visual_stimuli_camstim(self, stim_dict_lst):

        for stim_dict in stim_dict_lst:
            if stim_dict['stim_type'] == 'drifting_grating_camstim':
                print('adding stimulus: {} to nwb.'.format(stim_dict['stim_name']))
                self._add_drifting_grating_stimulation_camstim(stim_dict=stim_dict)
            elif stim_dict['stim_type'] == 'locally_sparse_noise_camstim':
                print('adding stimulus: {} to nwb.'.format(stim_dict['stim_name']))
                self._add_locally_sparse_noise_stimulation_camstim(stim_dict=stim_dict)
            else:
                pass

    # ===========================camstim visual stimuli related=========================================================


    # ===========================retinotopic_mapping visual stimuli related (indexed display)===========================
    def add_visual_display_log_retinotopic_mapping(self, stim_log):
        """
        add visual display log into nwb.

        :param stim_log: retinotopic_mapping.DisplayLogAnalysis.DisplayLogAnalyzer instance
        :return: None
        """

        stim_dict = stim_log.get_stim_dict()
        stim_ns = list(stim_dict.keys())
        stim_ns.sort()
        for stim_n in stim_ns:
            curr_stim_dict = stim_dict[stim_n]

            print('\nadding {} to nwb ...'.format(stim_n))

            if stim_n[-35:] == 'StimulusSeparatorRetinotopicMapping':
                self._add_stimulus_separator_retinotopic_mapping(curr_stim_dict)
            elif stim_n[-33:] == 'UniformContrastRetinotopicMapping':
                self._add_uniform_contrast_retinotopic_mapping(curr_stim_dict)
            elif stim_n[-32:] == 'FlashingCircleRetinotopicMapping':
                self._add_flashing_circle_retinotopic_mapping(curr_stim_dict)
            elif stim_n[-39:] == 'DriftingGratingCircleRetinotopicMapping':
                self._add_drifting_grating_circle_retinotopic_mapping(curr_stim_dict)
            elif stim_n[-37:] == 'StaticGratingCircleRetinotopicMapping':
                self._add_static_grating_circle_retinotopic_mapping(curr_stim_dict)
            elif stim_n[-30:] == '_SparseNoiseRetinotopicMapping':
                self._add_sparse_noise_retinotopic_mapping(curr_stim_dict)
            elif stim_n[-36:] == 'LocallySparseNoiseRetinotopicMapping':
                self._add_locally_sparse_noise_retinotopic_mapping(curr_stim_dict)
            elif stim_n[-30:] == 'StaticImagesRetinotopicMapping':
                self._add_static_images_retinotopic_mapping(curr_stim_dict)
            elif stim_n[-37:] == 'SinusoidalLuminanceRetinotopicMapping':
                self._add_sinusoidal_luminance_retinotopic_mapping(curr_stim_dict)
            else:
                raise ValueError('Do not understand stimulus name: {}.'.format(stim_n))

    def get_display_delay_retinotopic_mapping(self, stim_log, indicator_color_thr=0.5, ccg_t_range=(0., 0.1),
                                              ccg_bins=100, is_plot=True, pd_onset_ts_path=None,
                                              vsync_frame_ts_path=None):
        """

        :param stim_log: retinotopic_mapping.DisplayLogAnalysis.DisplayLogAnalyzer instance
        :param indicator_color_thr: float, [-1., 1.]
        :param ccg_t_range:
        :param ccg_bins:
        :param is_plot:
        :param pd_onset_ts_path: str, path to the timeseries of photodiode onsets in seconds
        :return:
        """

        # get photodiode onset timestamps (after display)
        if pd_onset_ts_path is None:
            if 'acquisition/timeseries/digital_photodiode_rise' in self.file_pointer:
                pd_ts_pd = self.file_pointer['acquisition/timeseries/digital_photodiode_rise/timestamps'][()]
            elif 'analysis/PhotodiodeOnsets' in self.file_pointer:
                 pd_ts_pd = self.file_pointer['analysis/PhotodiodeOnsets/timestamps'][()]
            else:
                raise LookupError('Cannot find photodiode onset timeseries.')
        else:
            pd_ts_pd = self.file_pointer[pd_onset_ts_path + '/timestamps'][()]

        # get vsync TTL timestamps for displayed frames
        if vsync_frame_ts_path is None:
            if 'acquisition/timeseries/digital_vsync_stim_rise' in self.file_pointer:
                vsync_ts = self.file_pointer['acquisition/timeseries/digital_vsync_stim_rise/timestamps'][()]
            elif 'acquisition/timeseries/digital_vsync_visual_rise' in self.file_pointer:
                vsync_ts = self.file_pointer['acquisition/timeseries/digital_vsync_visual_rise/timestamps'][()]
            else:
                raise LookupError('Cannot find vsync TTL signal for displayed frames.')
        else:
            vsync_ts = self.file_pointer[vsync_frame_ts_path + '/timestamps'][()]

        # check vsync_stim number and total frame number
        print('\nnumber of total frames in log file: {}'.format(stim_log.num_frame_tot))
        print('number of vsync_stim TTL rise events: {}'.format(len(vsync_ts)))
        # if stim_log.num_frame_tot != len(vsync_ts):
        #     raise ValueError('number of vsync_stim TTL rise events does not equal number of total frames in log file!')
        warnings.warn('number of vsync_stim TTL rise events does not equal number of total frames in log file!')

        # get photodiode onset timestamps from vsync_stim (before display)
        stim_dict = stim_log.get_stim_dict()
        pd_onsets_seq = stim_log.analyze_photodiode_onsets_sequential(stim_dict=stim_dict, pd_thr=indicator_color_thr)

        pd_ts_vsync = []
        for pd_onset in pd_onsets_seq:

            if pd_onset['global_frame_ind'] < len(vsync_ts):
                pd_ts_vsync.append(vsync_ts[pd_onset['global_frame_ind']])


        # calculate display delay as the weighted average of pd_ccg
        print('Total number of detected photodiode onsets: {}'.format(len(pd_ts_pd)))
        print('calculating photodiode cross-correlogram ...')
        pd_ccg = ta.discrete_cross_correlation(pd_ts_vsync, pd_ts_pd, t_range=ccg_t_range, bins=ccg_bins,
                                               isPlot=is_plot)
        if is_plot:
            plt.show()

        display_delay = np.sum(pd_ccg[0] * pd_ccg[1]) / np.sum(pd_ccg[1])
        print('calculated display delay: {} second.'.format(display_delay))
        self.file_pointer['analysis/visual_display_delay_sec'] = display_delay

        return display_delay

    def add_photodiode_onsets_combined_retinotopic_mapping(self, pd_onsets_com, display_delay,
                                       vsync_frame_path='acquisition/timeseries/digital_vsync_stim_rise'):
        """
        add combined photodiode onsets to self, currently the field is 'analysis/photodiode_onsets'

        :param pd_onsets_com: dictionary, product of
                              retinotopic_mapping.DisplayLogAnalysis.DisplayLogAnalyzer.analyze_photodiode_onsets_combined()
                              function
        :param display_delay: float, display delay in seconds
        :param vsync_frame_path: str, hdf5 path to digital timeseries of digital_vsync_frame_rise
        :return: None
        """

        vsync_stim_ts = self.file_pointer[vsync_frame_path]['timestamps'][()] + display_delay

        stim_ns = list(pd_onsets_com.keys())
        stim_ns.sort()

        pd_grp = self.file_pointer['analysis'].create_group('photodiode_onsets')
        for stim_n in stim_ns:
            stim_grp = pd_grp.create_group(stim_n)
            pd_onset_ns = list(pd_onsets_com[stim_n].keys())
            pd_onset_ns.sort()
            for pd_onset_n in pd_onset_ns:
                pd_onset_grp = stim_grp.create_group(pd_onset_n)
                pd_onset_grp['global_pd_onset_ind'] = pd_onsets_com[stim_n][pd_onset_n]['global_pd_onset_ind']
                pd_onset_grp['global_frame_ind'] = pd_onsets_com[stim_n][pd_onset_n]['global_frame_ind']

                try:
                   pd_onset_grp['pd_onset_ts_sec'] = vsync_stim_ts[pd_onsets_com[stim_n][pd_onset_n]['global_frame_ind']]
                except IndexError:
                    pd_onset_ts_sec = []
                    for gfi in pd_onsets_com[stim_n][pd_onset_n]['global_frame_ind']:
                        if gfi < len(vsync_stim_ts):
                            pd_onset_ts_sec.append(vsync_stim_ts[gfi])
                    pd_onset_grp['pd_onset_ts_sec'] = pd_onset_ts_sec


    def get_drifting_grating_response_table_retinotopic_mapping(self, stim_name, time_window=(-1, 2.5)):

        def get_sta(arr, arr_ts, trigger_ts, frame_start, frame_end):

            sta_arr = []

            for trig in trigger_ts:
                trig_ind = ta.find_nearest(arr_ts, trig)

                if trig_ind + frame_end < arr.shape[1]:
                    curr_sta = arr[:, (trig_ind + frame_start): (trig_ind + frame_end)]
                    # print(curr_sta.shape)
                    sta_arr.append(curr_sta.reshape((curr_sta.shape[0], 1, curr_sta.shape[1])))

            sta_arr = np.concatenate(sta_arr, axis=1)
            return sta_arr

        if time_window[0] >= time_window[1]:
            raise ValueError('time window should be from early time to late time.')

        grating_onsets_path = 'analysis/photodiode_onsets/{}'.format(stim_name)
        grating_ns = list(self.file_pointer[grating_onsets_path].keys())
        grating_ns.sort()
        # print('\n'.join(grating_ns))

        rois_and_traces_names = self.file_pointer['processing'].keys()
        rois_and_traces_names = [n for n in rois_and_traces_names if n[0:15] == 'rois_and_traces']
        rois_and_traces_names.sort()
        # print('\n'.join(rois_and_traces_paths))

        res_grp = self.file_pointer['analysis'].create_group('response_table_{}'.format(stim_name))
        for curr_trace_name in rois_and_traces_names:

            print('\nadding drifting grating response table for {} ...'.format(curr_trace_name))

            curr_plane_n = curr_trace_name[16:]

            res_grp_plane = res_grp.create_group(curr_plane_n)

            # get trace time stamps
            trace_ts = self.file_pointer['processing/motion_correction/MotionCorrection' \
                                          '/{}/corrected/timestamps'.format(curr_plane_n)]
            # get traces
            traces = {}
            if 'processing/{}/DfOverF/dff_center'.format(curr_trace_name) in self.file_pointer:
                traces['global_dff_center'] = self.file_pointer[
                    'processing/{}/DfOverF/dff_center/data'.format(curr_trace_name)][()]
            if 'processing/{}/Fluorescence'.format(curr_trace_name) in self.file_pointer:
                f_types = self.file_pointer['processing/{}/Fluorescence'.format(curr_trace_name)].keys()
                for f_type in f_types:
                    traces[f_type] = self.file_pointer['processing/{}/Fluorescence/{}/data'
                        .format(curr_trace_name, f_type)][()]
            # print(traces.keys())

            # frame_dur = np.mean(np.diff(trace_ts))
            # frame_start = int(time_window[0] // frame_dur)
            # frame_end = int(time_window[1] // frame_dur)
            # t_axis = np.arange(frame_end - frame_start) * frame_dur + time_window[0]

            frame_dur = np.mean(np.diff(trace_ts))
            frame_start = int(np.floor(time_window[0] / frame_dur))
            frame_end = int(np.ceil(time_window[1] / frame_dur))

            t_axis = np.arange(frame_end - frame_start) * frame_dur + (frame_start * frame_dur)
            res_grp_plane.attrs['sta_timestamps'] = t_axis

            for grating_n in grating_ns:

                onsets_grating_grp = self.file_pointer['{}/{}'.format(grating_onsets_path, grating_n)]

                curr_grating_grp = res_grp_plane.create_group(grating_n)

                grating_onsets = onsets_grating_grp['pd_onset_ts_sec'][()]

                curr_grating_grp.attrs['global_trigger_timestamps'] = grating_onsets
                curr_grating_grp.attrs['sta_traces_dimenstion'] = 'roi x trial x timepoint'

                for trace_n, trace in traces.items():
                    sta = get_sta(arr=trace, arr_ts=trace_ts, trigger_ts=grating_onsets, frame_start=frame_start,
                                  frame_end=frame_end)
                    curr_grating_grp.create_dataset('sta_' + trace_n, data=sta, compression='lzf')

    def get_spatial_temporal_receptive_field_retinotopic_mapping(self, stim_name, time_window=(-0.5, 2.),
                                                                 verbose=True):

        def get_sta(arr, arr_ts, trigger_ts, frame_start, frame_end):

            sta_arr = []

            for trig in trigger_ts:
                trig_ind = ta.find_nearest(arr_ts, trig)

                if trig_ind + frame_end < arr.shape[1]:
                    curr_sta = arr[:, (trig_ind + frame_start): (trig_ind + frame_end)]
                    # print(curr_sta.shape)
                    sta_arr.append(curr_sta.reshape((curr_sta.shape[0], 1, curr_sta.shape[1])))

            sta_arr = np.concatenate(sta_arr, axis=1)
            return sta_arr

        if time_window[0] >= time_window[1]:
            raise ValueError('time window should be from early time to late time.')

        probe_onsets_path = 'analysis/photodiode_onsets/{}'.format(stim_name)
        probe_ns = list(self.file_pointer[probe_onsets_path].keys())
        probe_ns.sort()
        # print('\n'.join(probe_ns))

        rois_and_traces_names = self.file_pointer['processing'].keys()
        rois_and_traces_names = [n for n in rois_and_traces_names if n[0:15] == 'rois_and_traces']
        rois_and_traces_names.sort()
        # print('\n'.join(rois_and_traces_paths))

        strf_grp = self.file_pointer['analysis'].create_group('strf_{}'.format(stim_name))
        for curr_trace_name in rois_and_traces_names:

            if verbose:
                print('\nadding strfs for {} ...'.format(curr_trace_name))

            curr_plane_n = curr_trace_name[16:]

            strf_grp_plane = strf_grp.create_group(curr_plane_n)

            # get trace time stamps
            trace_ts = self.file_pointer['processing/motion_correction/MotionCorrection' \
                                          '/{}/corrected/timestamps'.format(curr_plane_n)]
            # get traces
            traces = {}
            if 'processing/{}/DfOverF/dff_center'.format(curr_trace_name) in self.file_pointer:
                traces['global_dff_center'] = self.file_pointer[
                    'processing/{}/DfOverF/dff_center/data'.format(curr_trace_name)][()]
            if 'processing/{}/Fluorescence'.format(curr_trace_name) in self.file_pointer:
                f_types = self.file_pointer['processing/{}/Fluorescence'.format(curr_trace_name)].keys()
                for f_type in f_types:
                    traces[f_type] = self.file_pointer['processing/{}/Fluorescence/{}/data'
                        .format(curr_trace_name, f_type)][()]
            # print(traces.keys())

            frame_dur = np.mean(np.diff(trace_ts))
            frame_start = int(np.floor(time_window[0] / frame_dur))
            frame_end = int(np.ceil(time_window[1] / frame_dur))
            t_axis = np.arange(frame_end - frame_start) * frame_dur + (frame_start * frame_dur)
            # t_axis = np.arange(frame_end - frame_start) * frame_dur + time_window[0]

            strf_grp_plane.attrs['sta_timestamps'] = t_axis

            for probe_i, probe_n in enumerate(probe_ns):

                if verbose and probe_i % 100 == 0:
                    print('\tprocessing probe {} / {}'.format(probe_i+1, len(probe_ns)))

                onsets_probe_grp = self.file_pointer['{}/{}'.format(probe_onsets_path, probe_n)]

                curr_probe_grp = strf_grp_plane.create_group(probe_n)

                probe_onsets = onsets_probe_grp['pd_onset_ts_sec'][()]

                curr_probe_grp['global_trigger_timestamps'] = h5py.SoftLink('/{}/{}/pd_onset_ts_sec'
                                                                            .format(probe_onsets_path, probe_n))
                curr_probe_grp.attrs['sta_traces_dimenstion'] = 'roi x trial x timepoint'

                for trace_n, trace in traces.items():
                    sta = get_sta(arr=trace, arr_ts=trace_ts, trigger_ts=probe_onsets, frame_start=frame_start,
                                  frame_end=frame_end)
                    curr_probe_grp.create_dataset('sta_' + trace_n, data=sta, compression='lzf')

    def _add_stimulus_separator_retinotopic_mapping(self, ss_dict):

        stim_name = ss_dict['stim_name']

        if stim_name[-35:] != 'StimulusSeparatorRetinotopicMapping':
            raise ValueError('stimulus should be "StimulusSeparatorRetinotopicMapping" (StimulusSeparator from '
                             'retinotopic_mapping package). ')

        # add template
        template_ts = self.create_timeseries('TimeSeries', stim_name, 'template')
        template_ts.set_data(ss_dict['frames_unique'], unit='', conversion=np.nan, resolution=np.nan)
        template_ts.set_value('num_samples', len(ss_dict['frames_unique']))
        template_ts.set_source(ss_dict['source'])
        template_ts.finalize()

        # add stimulus
        stim_ts = self.create_timeseries('IndexSeries', stim_name, 'stimulus')
        stim_ts.set_time(ss_dict['timestamps'], dtype='u8')
        stim_ts.set_data(ss_dict['index_to_display'], unit='frame', conversion=1, resolution=1, dtype='u4')
        stim_ts.set_value_as_link('indexed_timeseries', '/stimulus/templates/{}'.format(stim_name))
        stim_ts.set_comments('The "timestamps" of this TimeSeries are indices (64-bit unsigned integer, hacked the '
                             'original ainwb code) referencing the entire display sequence. It should match hardware '
                             'vsync TTL (see "/acquisition/timeseries/digital_vsync_stim/rise"). The "data" of this '
                             'TimeSeries are indices referencing the frames template saved in the "indexed_timeseries" '
                             'field.')
        stim_ts.set_description('stimulus separator displayed by retinotopic_mapping package')
        stim_ts.set_source(ss_dict['source'])
        for key in ['stim_name', 'pregap_dur', 'postgap_dur', 'coordinate', 'background']:
            stim_ts.set_value(key, ss_dict[key])
        for key in ['frame_config']:
            stim_ts.set_value(key, [s.encode('utf-8') for s in ss_dict[key]])
        stim_ts.set_value('indicator_on_frame_num', ss_dict['indicator_on_frame_num'])
        stim_ts.set_value('indicator_off_frame_num', ss_dict['indicator_off_frame_num'])
        stim_ts.set_value('cycle_num', ss_dict['cycle_num'])

        stim_ts.finalize()

    def _add_uniform_contrast_retinotopic_mapping(self, uc_dict):
        stim_name = uc_dict['stim_name']

        if stim_name[-33:] != 'UniformContrastRetinotopicMapping':
            raise ValueError('stimulus should be "UniformContrastRetinotopicMapping" (UniformContrast from '
                             'retinotopic_mapping package). ')

        # add template
        template_ts = self.create_timeseries('TimeSeries', stim_name, 'template')
        template_ts.set_data(uc_dict['frames_unique'], unit='', conversion=np.nan, resolution=np.nan)
        template_ts.set_value('num_samples', len(uc_dict['frames_unique']))
        template_ts.set_source(uc_dict['source'])
        template_ts.finalize()

        # add stimulus
        stim_ts = self.create_timeseries('IndexSeries', stim_name, 'stimulus')
        stim_ts.set_time(uc_dict['timestamps'], dtype='u8')
        stim_ts.set_data(uc_dict['index_to_display'], unit='frame', conversion=1, resolution=1, dtype='u4')
        stim_ts.set_value_as_link('indexed_timeseries', '/stimulus/templates/{}'.format(stim_name))
        stim_ts.set_comments('The "timestamps" of this TimeSeries are indices (64-bit unsigned integer, hacked the '
                             'original ainwb code) referencing the entire display sequence. It should match hardware '
                             'vsync TTL (see "/acquisition/timeseries/digital_vsync_stim/rise"). The "data" of this '
                             'TimeSeries are indices referencing the frames template saved in the "indexed_timeseries" '
                             'field.')
        stim_ts.set_description('uniform contrast displayed by retinotopic_mapping package')
        stim_ts.set_source(uc_dict['source'])
        for key in ['stim_name', 'pregap_dur', 'postgap_dur', 'coordinate', 'background']:
            stim_ts.set_value(key, uc_dict[key])
        for key in ['frame_config']:
            stim_ts.set_value(key, [s.encode('utf-8') for s in uc_dict[key]])
        stim_ts.set_value('duration', uc_dict['duration'])
        stim_ts.set_value('color', uc_dict['color'])
        stim_ts.finalize()

    def _add_flashing_circle_retinotopic_mapping(self, fc_dict):
        stim_name = fc_dict['stim_name']

        if stim_name[-32:] != 'FlashingCircleRetinotopicMapping':
            raise ValueError('stimulus should be "FlashingCircleRetinotopicMapping" (FlashingCircle from '
                             'retinotopic_mapping package). ')

        # add template
        template_ts = self.create_timeseries('TimeSeries', stim_name, 'template')
        template_ts.set_data(fc_dict['frames_unique'], unit='', conversion=np.nan, resolution=np.nan)
        template_ts.set_value('num_samples', len(fc_dict['frames_unique']))
        template_ts.set_source(fc_dict['source'])
        template_ts.finalize()

        # add stimulus
        stim_ts = self.create_timeseries('IndexSeries', stim_name, 'stimulus')
        stim_ts.set_time(fc_dict['timestamps'], dtype='u8')
        stim_ts.set_data(fc_dict['index_to_display'], unit='frame', conversion=1, resolution=1, dtype='u4')
        stim_ts.set_value_as_link('indexed_timeseries', '/stimulus/templates/{}'.format(stim_name))
        stim_ts.set_comments('The "timestamps" of this TimeSeries are indices (64-bit unsigned integer, hacked the '
                             'original ainwb code) referencing the entire display sequence. It should match hardware '
                             'vsync TTL (see "/acquisition/timeseries/digital_vsync_stim/rise"). The "data" of this '
                             'TimeSeries are indices referencing the frames template saved in the "indexed_timeseries" '
                             'field.')
        stim_ts.set_description('flashing circle displayed by retinotopic_mapping package')
        stim_ts.set_source(fc_dict['source'])
        for key in ['stim_name', 'pregap_dur', 'postgap_dur', 'coordinate', 'background']:
            stim_ts.set_value(key, fc_dict[key])
        for key in ['frame_config']:
            stim_ts.set_value(key, [s.encode('utf-8') for s in fc_dict[key]])
        stim_ts.set_value('is_smooth_edge', fc_dict['is_smooth_edge'])
        stim_ts.set_value('smooth_width_ratio', fc_dict['smooth_width_ratio'])
        stim_ts.set_value('center', fc_dict['center'])
        stim_ts.set_value('radius', fc_dict['radius'])
        stim_ts.set_value('flash_frame_num', fc_dict['flash_frame_num'])
        stim_ts.set_value('midgap_dur', fc_dict['midgap_dur'])
        stim_ts.set_value('iteration', fc_dict['iteration'])
        stim_ts.set_value('color', fc_dict['color'])
        stim_ts.finalize()

    def _add_drifting_grating_circle_retinotopic_mapping(self, dgc_dict):
        stim_name = dgc_dict['stim_name']

        if stim_name[-39:] != 'DriftingGratingCircleRetinotopicMapping':
            raise ValueError('stimulus should be "DriftingGratingCircleRetinotopicMapping" (DriftingGratingCircle from '
                             'retinotopic_mapping package). ')

        # add template
        template_ts = self.create_timeseries('TimeSeries', stim_name, 'template')
        frames_unique = dgc_dict['frames_unique']
        frames_template = []
        for frame in frames_unique:

            # temporally fix a bug
            if frame == (1, 1, 0., 0., 0., 0., 0., 1.):
                frame = (1, 1, 0., 0., 0., 0., 0., 0., 1.)

            if frame == (1, 1, 0., 0., 0., 0., 0., 0.):
                frame = (1, 1, 0., 0., 0., 0., 0., 0., 0.)

            curr_frame = np.array(frame)
            # print(curr_frame)
            curr_frame[curr_frame == None] = np.nan
            frames_template.append(np.array(curr_frame, dtype=np.float32))
        frames_template = np.array(frames_template)
        template_ts.set_data(frames_template, unit='', conversion=np.nan, resolution=np.nan)
        template_ts.set_value('num_samples', frames_template.shape[0])
        template_ts.set_source(dgc_dict['source'])
        template_ts.finalize()

        # add stimulus
        stim_ts = self.create_timeseries('IndexSeries', stim_name, 'stimulus')
        stim_ts.set_time(dgc_dict['timestamps'], dtype='u8')
        stim_ts.set_data(dgc_dict['index_to_display'], unit='frame', conversion=1, resolution=1, dtype='u4')
        stim_ts.set_value_as_link('indexed_timeseries', '/stimulus/templates/{}'.format(stim_name))
        stim_ts.set_comments('The "timestamps" of this TimeSeries are indices (64-bit unsigned integer, hacked the '
                             'original ainwb code) referencing the entire display sequence. It should match hardware '
                             'vsync TTL (see "/acquisition/timeseries/digital_vsync_stim/rise"). The "data" of this '
                             'TimeSeries are indices referencing the frames template saved in the "indexed_timeseries" '
                             'field.')
        stim_ts.set_description('drifting grating circle displayed by retinotopic_mapping package')
        stim_ts.set_source(dgc_dict['source'])
        for key in ['stim_name', 'pregap_dur', 'postgap_dur', 'coordinate', 'background']:
            stim_ts.set_value(key, dgc_dict[key])
        for key in ['frame_config']:
            stim_ts.set_value(key, [s.encode('utf-8') for s in dgc_dict[key]])
        stim_ts.set_value('is_smooth_edge', dgc_dict['is_smooth_edge'])
        stim_ts.set_value('smooth_width_ratio', dgc_dict['smooth_width_ratio'])
        stim_ts.set_value('center', dgc_dict['center'])
        stim_ts.set_value('iteration', dgc_dict['iteration'])
        stim_ts.set_value('dire_list', dgc_dict['dire_list'])
        stim_ts.set_value('radius_list', dgc_dict['radius_list'])
        stim_ts.set_value('con_list', dgc_dict['con_list'])
        stim_ts.set_value('sf_list', dgc_dict['sf_list'])
        stim_ts.set_value('tf_list', dgc_dict['tf_list'])
        stim_ts.set_value('block_dur', dgc_dict['block_dur'])
        stim_ts.set_value('midgap_dur', dgc_dict['midgap_dur'])
        stim_ts.finalize()

    def _add_static_grating_circle_retinotopic_mapping(self, sgc_dict):
        stim_name = sgc_dict['stim_name']

        if stim_name[-37:] != 'StaticGratingCircleRetinotopicMapping':
            raise ValueError('stimulus should be "StaticGratingCircleRetinotopicMapping" (StaticGratingCircle from '
                             'retinotopic_mapping package). ')

        # add template
        template_ts = self.create_timeseries('TimeSeries', stim_name, 'template')
        frames_unique = sgc_dict['frames_unique']
        frames_template = []
        for frame in frames_unique:
            curr_frame = np.array(frame)
            curr_frame[curr_frame == None] = np.nan
            frames_template.append(np.array(curr_frame, dtype=np.float32))
        frames_template = np.array(frames_template)
        template_ts.set_data(frames_template, unit='', conversion=np.nan, resolution=np.nan)
        template_ts.set_value('num_samples', frames_template.shape[0])
        template_ts.set_source(sgc_dict['source'])
        template_ts.finalize()

        # add stimulus
        stim_ts = self.create_timeseries('IndexSeries', stim_name, 'stimulus')
        stim_ts.set_time(sgc_dict['timestamps'], dtype='u8')
        stim_ts.set_data(sgc_dict['index_to_display'], unit='frame', conversion=1, resolution=1, dtype='u4')
        stim_ts.set_value_as_link('indexed_timeseries', '/stimulus/templates/{}'.format(stim_name))
        stim_ts.set_comments('The "timestamps" of this TimeSeries are indices (64-bit unsigned integer, hacked the '
                             'original ainwb code) referencing the entire display sequence. It should match hardware '
                             'vsync TTL (see "/acquisition/timeseries/digital_vsync_stim/rise"). The "data" of this '
                             'TimeSeries are indices referencing the frames template saved in the "indexed_timeseries" '
                             'field.')
        stim_ts.set_description('static grating circle displayed by retinotopic_mapping package')
        stim_ts.set_source(sgc_dict['source'])
        for key in ['stim_name', 'pregap_dur', 'postgap_dur', 'coordinate', 'background']:
            stim_ts.set_value(key, sgc_dict[key])
        for key in ['frame_config']:
            stim_ts.set_value(key, [s.encode('utf-8') for s in sgc_dict[key]])
        stim_ts.set_value('is_smooth_edge', sgc_dict['is_smooth_edge'])
        stim_ts.set_value('smooth_width_ratio', sgc_dict['smooth_width_ratio'])
        stim_ts.set_value('center', sgc_dict['center'])
        stim_ts.set_value('iteration', sgc_dict['iteration'])
        stim_ts.set_value('ori_list', sgc_dict['ori_list'])
        stim_ts.set_value('radius_list', sgc_dict['radius_list'])
        stim_ts.set_value('con_list', sgc_dict['con_list'])
        stim_ts.set_value('sf_list', sgc_dict['sf_list'])
        stim_ts.set_value('phase_list', sgc_dict['phase_list'])
        stim_ts.set_value('display_dur', sgc_dict['display_dur'])
        stim_ts.set_value('midgap_dur', sgc_dict['midgap_dur'])
        stim_ts.finalize()

    def _add_sparse_noise_retinotopic_mapping(self, sn_dict):
        stim_name = sn_dict['stim_name']

        if stim_name[-30:] != '_SparseNoiseRetinotopicMapping':
            raise ValueError('stimulus should be "SparseNoiseRetinotopicMapping" (SparseNoise from '
                             'retinotopic_mapping package). ')

        # add template
        template_ts = self.create_timeseries('TimeSeries', stim_name, 'template')
        frames_template = []
        probes = []
        for frame in sn_dict['frames_unique']:

            frames_template.append(np.array([frame[0], frame[3]], dtype=np.float32))

            if frame[1] is None:
                probes.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
            else:
                # print([frame[1][0], frame[1][1], frame[2]])
                probes.append(np.array([frame[1][0], frame[1][1], frame[2]], dtype=np.float32))

        frames_template = np.array(frames_template)
        probes = np.array(probes)
        template_ts.set_data(frames_template, unit='', conversion=np.nan, resolution=np.nan)
        template_ts.set_value('probes', probes)
        template_ts.set_value('num_samples', frames_template.shape[0])
        template_ts.set_source(sn_dict['source'])
        template_ts.set_description('The "data" field saved modified frame configuration: '
                                    '[is_display, indicator color)]. While the "probe" field saved modified probe '
                                    'configuration: [altitude, azimuth, polarity]. These two fields have one-to-one '
                                    'relationship. Together they define an unique display frame of sparse noise '
                                    'stimulus. The order of these two fields should not be changed.')
        template_ts.finalize()

        # add stimulus
        stim_ts = self.create_timeseries('IndexSeries', stim_name, 'stimulus')
        stim_ts.set_time(sn_dict['timestamps'], dtype='u8')
        stim_ts.set_data(sn_dict['index_to_display'], unit='frame', conversion=1, resolution=1, dtype='u4')
        stim_ts.set_value_as_link('indexed_timeseries', '/stimulus/templates/{}'.format(stim_name))
        stim_ts.set_comments('The "timestamps" of this TimeSeries are indices (64-bit unsigned integer, hacked the '
                             'original ainwb code) referencing the entire display sequence. It should match hardware '
                             'vsync TTL (see "/acquisition/timeseries/digital_vsync_stim/rise"). The "data" of this '
                             'TimeSeries are indices referencing the frames template saved in the "indexed_timeseries" '
                             'field.')
        stim_ts.set_description('sparse noise displayed by retinotopic_mapping package')
        stim_ts.set_source(sn_dict['source'])
        for key in ['stim_name', 'pregap_dur', 'postgap_dur', 'coordinate', 'background']:
            stim_ts.set_value(key, sn_dict[key])
        stim_ts.set_value('frame_config', ['is_display'.encode('utf-8'),
                                           'indicator color[-1., 1.]'.encode('utf-8')]) # modified frame config
        stim_ts.set_value('probe_config', ['altitude (deg)'.encode('utf-8'),
                                           'azimuth (deg)'.encode('utf-8'),
                                           'polarity'.encode('utf-8')]) # modified probe config
        stim_ts.set_value('is_include_edge', sn_dict['is_include_edge'])
        stim_ts.set_value('probe_frame_num', sn_dict['probe_frame_num'])
        stim_ts.set_value('subregion', sn_dict['subregion'])
        stim_ts.set_value('iteration', sn_dict['iteration'])
        stim_ts.set_value('grid_space', sn_dict['grid_space'])
        stim_ts.set_value('probe_orientation', sn_dict['probe_orientation'])
        stim_ts.set_value('sign', sn_dict['sign'])
        stim_ts.set_value('probe_size', sn_dict['probe_size'])
        stim_ts.finalize()

    def _add_locally_sparse_noise_retinotopic_mapping(self, lsn_dict):
        stim_name = lsn_dict['stim_name']

        if stim_name[-36:] != 'LocallySparseNoiseRetinotopicMapping':
            raise ValueError('stimulus should be "LocallySparseNoiseRetinotopicMapping" (LocallySparseNoise from '
                             'retinotopic_mapping package). ')

        # add template
        template_ts = self.create_timeseries('TimeSeries', stim_name, 'template')

        max_probe_num = 0 # get max probe number in a single frame
        for frame in lsn_dict['frames_unique']:
            if frame[1] is not None:
                max_probe_num = max([max_probe_num, len(frame[1])])

        frames_template = np.empty((len(lsn_dict['frames_unique']), 2), dtype=np.float32)
        probes = np.empty((len(lsn_dict['frames_unique']), max_probe_num, 3), dtype=np.float64)
        probes[:] = np.nan

        for frame_ind, frame in enumerate(lsn_dict['frames_unique']):

            frames_template[frame_ind] = np.array([frame[0], frame[3]], dtype=np.float32)

            if frame[1] is not None:
                for curr_probe_i, curr_probe in enumerate(frame[1]):
                    probes[frame_ind, curr_probe_i, :] = np.array([curr_probe], dtype=np.float64)

        template_ts.set_data(frames_template, unit='', conversion=np.nan, resolution=np.nan)
        template_ts.set_value('probes', probes, dtype='float64')
        template_ts.set_value('num_samples', frames_template.shape[0])
        template_ts.set_source(lsn_dict['source'])
        template_ts.set_description('The "data" field saved modified frame configuration: '
                                    '[is_display, indicator color)]. While the "probe" field saved modified probe '
                                    'configuration. It is a 3-d array with axis: [frame_num, probe_num, probe_info], '
                                    'the probe info is specified as: [altitude, azimuth, polarity]. The frame '
                                    'dimension of these two fields have one-to-one relationship. '
                                    'Together they define an unique display frame of locally sparse noise stimulus. '
                                    'The order of these two fields should not be changed.')
        template_ts.finalize()

        # add stimulus
        stim_ts = self.create_timeseries('IndexSeries', stim_name, 'stimulus')
        stim_ts.set_time(lsn_dict['timestamps'], dtype='u8')
        stim_ts.set_data(lsn_dict['index_to_display'], unit='frame', conversion=1, resolution=1, dtype='u4')
        stim_ts.set_value_as_link('indexed_timeseries', '/stimulus/templates/{}'.format(stim_name))
        stim_ts.set_comments('The "timestamps" of this TimeSeries are indices (64-bit unsigned integer, hacked the '
                             'original ainwb code) referencing the entire display sequence. It should match hardware '
                             'vsync TTL (see "/acquisition/timeseries/digital_vsync_stim/rise"). The "data" of this '
                             'TimeSeries are indices referencing the frames template saved in the "indexed_timeseries" '
                             'field.')
        stim_ts.set_description('locally sparse noise displayed by retinotopic_mapping package')
        stim_ts.set_source(lsn_dict['source'])
        for key in ['stim_name', 'pregap_dur', 'postgap_dur', 'coordinate', 'background']:
            stim_ts.set_value(key, lsn_dict[key])
        stim_ts.set_value('frame_config', ['is_display'.encode('utf-8'),
                                           'indicator color[-1., 1.]'.encode('utf-8')])  # modified frame config
        stim_ts.set_value('probe_config', 'frame_num x probe_num x probe_info '
                                          '(altitude_deg, azimuth_deg, polarity)'.encode('utf-8'))  # modified probe config
        stim_ts.set_value('is_include_edge', lsn_dict['is_include_edge'])
        stim_ts.set_value('probe_frame_num', lsn_dict['probe_frame_num'])
        stim_ts.set_value('subregion', lsn_dict['subregion'])
        stim_ts.set_value('iteration', lsn_dict['iteration'])
        stim_ts.set_value('grid_space', lsn_dict['grid_space'])
        stim_ts.set_value('probe_orientation', lsn_dict['probe_orientation'])
        stim_ts.set_value('sign', lsn_dict['sign'])
        stim_ts.set_value('probe_size', lsn_dict['probe_size'])
        stim_ts.set_value('min_distance', lsn_dict['min_distance'])
        stim_ts.set_value('repeat', lsn_dict['repeat'])
        stim_ts.finalize()

    def _add_static_images_retinotopic_mapping(self, si_dict):
        stim_name = si_dict['stim_name']

        if stim_name[-30:] != 'StaticImagesRetinotopicMapping':
            raise ValueError('stimulus should be "StaticImagesRetinotopicMapping" (StaticImages from '
                             'retinotopic_mapping package). ')

        # add template
        template_ts = self.create_timeseries('TimeSeries', stim_name, 'template')
        frames_unique = si_dict['frames_unique']
        frames_template = []
        for frame in frames_unique:
            curr_frame = np.array(frame)
            curr_frame[curr_frame == None] = np.nan
            frames_template.append(np.array(curr_frame, dtype=np.float32))
        frames_template = np.array(frames_template)
        template_ts.set_data(frames_template, unit='', conversion=np.nan, resolution=np.nan)
        template_ts.set_value('num_samples', frames_template.shape[0])
        template_ts.set_value('images_wrapped', si_dict['images_wrapped'])
        template_ts.set_value('images_dewrapped', si_dict['images_dewrapped'])
        template_ts.set_source(si_dict['source'])
        template_ts.finalize()

        # add stimulus
        stim_ts = self.create_timeseries('IndexSeries', stim_name, 'stimulus')
        stim_ts.set_time(si_dict['timestamps'], dtype='u8')
        stim_ts.set_data(si_dict['index_to_display'], unit='frame', conversion=1, resolution=1, dtype='u4')
        stim_ts.set_value_as_link('indexed_timeseries', '/stimulus/templates/{}'.format(stim_name))
        stim_ts.set_comments('The "timestamps" of this TimeSeries are indices (64-bit unsigned integer, hacked the '
                             'original ainwb code) referencing the entire display sequence. It should match hardware '
                             'vsync TTL (see "/acquisition/timeseries/digital_vsync_stim/rise"). The "data" of this '
                             'TimeSeries are indices referencing the frames template saved in the "indexed_timeseries" '
                             'field.')
        stim_ts.set_description('static images displayed by retinotopic_mapping package')
        stim_ts.set_source(si_dict['source'])
        for key in ['stim_name', 'pregap_dur', 'postgap_dur', 'coordinate', 'background']:
            stim_ts.set_value(key, si_dict[key])
        for key in ['frame_config']:
            stim_ts.set_value(key, [s.encode('utf-8') for s in si_dict[key]])
        stim_ts.set_value('altitude_dewrapped', si_dict['altitude_dewrapped'])
        stim_ts.set_value('azimuth_dewrapped', si_dict['azimuth_dewrapped'])
        stim_ts.set_value('img_center', si_dict['img_center'])
        stim_ts.set_value('midgap_dur', si_dict['midgap_dur'])
        stim_ts.set_value('display_dur', si_dict['display_dur'])
        stim_ts.set_value('iteration', si_dict['iteration'])
        stim_ts.set_value('deg_per_pixel_azi', si_dict['deg_per_pixel_azi'])
        stim_ts.set_value('deg_per_pixel_alt', si_dict['deg_per_pixel_alt'])
        stim_ts.finalize()

    def _add_sinusoidal_luminance_retinotopic_mapping(self, sl_dict):
        stim_name = sl_dict['stim_name']

        if stim_name[-37:] != 'SinusoidalLuminanceRetinotopicMapping':
            raise ValueError('stimulus should be "SinusoidalLuminanceRetinotopicMapping" (StaticImages from '
                             'retinotopic_mapping package). ')

        # add template
        template_ts = self.create_timeseries('TimeSeries', stim_name, 'template')
        frames_unique = sl_dict['frames_unique']
        frames_template = []
        for frame in frames_unique:
            curr_frame = np.array(frame)
            curr_frame[curr_frame == None] = np.nan
            frames_template.append(np.array(curr_frame, dtype=np.float32))
        frames_template = np.array(frames_template)
        template_ts.set_data(frames_template, unit='', conversion=np.nan, resolution=np.nan)
        template_ts.set_value('num_samples', frames_template.shape[0])
        template_ts.set_source(sl_dict['source'])
        template_ts.finalize()

        # add stimulus
        stim_ts = self.create_timeseries('IndexSeries', stim_name, 'stimulus')
        stim_ts.set_time(sl_dict['timestamps'], dtype='u8')
        stim_ts.set_data(sl_dict['index_to_display'], unit='frame', conversion=1, resolution=1, dtype='u4')
        stim_ts.set_value_as_link('indexed_timeseries', '/stimulus/templates/{}'.format(stim_name))
        stim_ts.set_comments('The "timestamps" of this TimeSeries are indices (64-bit unsigned integer, hacked the '
                             'original ainwb code) referencing the entire display sequence. It should match hardware '
                             'vsync TTL (see "/acquisition/timeseries/digital_vsync_stim/rise"). The "data" of this '
                             'TimeSeries are indices referencing the frames template saved in the "indexed_timeseries" '
                             'field.')
        stim_ts.set_description('sinusoidal luminance displayed by retinotopic_mapping package')
        stim_ts.set_source(sl_dict['source'])
        for key in ['stim_name', 'pregap_dur', 'postgap_dur', 'coordinate', 'background']:
            stim_ts.set_value(key, sl_dict[key])
        for key in ['frame_config']:
            stim_ts.set_value(key, [s.encode('utf-8') for s in sl_dict[key]])
        stim_ts.set_value('cycle_num', sl_dict['cycle_num'])
        stim_ts.set_value('max_level', sl_dict['max_level'])
        stim_ts.set_value('start_phase', sl_dict['start_phase'])
        stim_ts.set_value('midgap_dur', sl_dict['midgap_dur'])
        stim_ts.set_value('frequency', sl_dict['frequency'])
        stim_ts.finalize()
    # ===========================retinotopic_mapping visual stimuli related (indexed display)===========================


    # # ===========================corticalmapping visual stimuli related (non-indexed display)===========================
    # def add_visual_stimulus_corticalmapping(self, log_path, display_order=0):
    #     """
    #     load visual stimulation given saved display log pickle file
    #     :param log_path: the path to the display log generated by corticalmapping.VisualStim
    #     :param display_order: int, in case there is more than one visual display in the file.
    #                           This value records the order of the displays
    #     :return:
    #     """
    #     self._check_display_order(display_order)

    #     log_dict = ft.loadFile(log_path)

    #     stim_name = log_dict['stimulation']['stimName']

    #     display_frames = log_dict['presentation']['displayFrames']
    #     time_stamps = log_dict['presentation']['timeStamp']

    #     if len(display_frames) != len(time_stamps):
    #         print ('\nWarning: {}'.format(log_path))
    #         print('Unequal number of displayFrames ({}) and timeStamps ({}).'.format(len(display_frames),
    #                                                                                  len(time_stamps)))

    #     if stim_name == 'SparseNoise':
    #         self._add_sparse_noise_stimulus_corticalmapping(log_dict, display_order=display_order)
    #     elif stim_name == 'FlashingCircle':
    #         self._add_flashing_circle_stimulus_corticalmapping(log_dict, display_order=display_order)
    #     elif stim_name == 'UniformContrast':
    #         self._add_uniform_contrast_stimulus_corticalmapping(log_dict, display_order=display_order)
    #     elif stim_name == 'DriftingGratingCircle':
    #         self._add_drifting_grating_circle_stimulus_corticalmapping(log_dict, display_order=display_order)
    #     elif stim_name == 'KSstimAllDir':
    #         self._add_drifting_checker_board_stimulus_corticalmapping(log_dict, display_order=display_order)
    #     else:
    #         raise ValueError('stimulation name {} unrecognizable!'.format(stim_name))

    # def add_visual_stimuli_corticalmapping(self, log_paths):

    #     exist_stimuli = self.file_pointer['stimulus/presentation'].keys()

    #     for i, log_path in enumerate(log_paths):
    #         self.add_visual_stimulus_corticalmapping(log_path, i + len(exist_stimuli))

    # def analyze_visual_stimuli_corticalmapping(self, onsets_ts=None):
    #     """
    #     add stimuli onset timestamps of all saved stimulus presentations to 'processing/stimulus_onsets' module

    #     :param onsets_ts: 1-d array, timestamps of stimuli onsets. if None, it will look for
    #                       ['processing/photodiode/timestemps'] as onset_ts
    #     """

    #     if onsets_ts is None:
    #         print('input onsets_ts is None, try to use photodiode onsets as onsets_ts.')
    #         onsets_ts = self.file_pointer['processing/PhotodiodeOnsets/photodiode_onsets/timestamps'].value

    #     stim_ns = self.file_pointer['stimulus/presentation'].keys()
    #     stim_ns.sort()

    #     total_onsets = 0

    #     for stim_ind, stim_n in enumerate(stim_ns):

    #         if int(stim_n[0: 2]) != stim_ind:
    #             raise ValueError('Stimulus name: {} does not follow the order: {}'.format(stim_n, stim_ind))

    #         curr_stim_grp = self.file_pointer['stimulus/presentation'][stim_n]
    #         if curr_stim_grp['stim_name'].value == 'SparseNoise':
    #             _ = self._analyze_sparse_noise_frames_corticalmapping(curr_stim_grp)
    #         elif curr_stim_grp['stim_name'].value == 'FlashingCircle':
    #             _ = self._analyze_flashing_circle_frames_corticalmapping(curr_stim_grp)
    #         elif curr_stim_grp['stim_name'].value == 'DriftingGratingCircle':
    #             _ = self._analyze_driftig_grating_frames_corticalmapping(curr_stim_grp)
    #         elif curr_stim_grp['stim_name'].value == 'UniformContrast':
    #             _ = self._analyze_uniform_contrast_frames_corticalmapping(curr_stim_grp)
    #         else:
    #             raise LookupError('Do not understand stimulus type: {}.'.format(stim_n))

    #         curr_onset_arr, curr_data_format, curr_description, pooled_onsets = _

    #         total_onsets += curr_onset_arr.shape[0]

    #     if total_onsets != len(onsets_ts):
    #         raise ValueError('Number of stimuli onsets ({}) do not match the number of given onsets_ts ({}).'
    #                          .format(total_onsets, len(onsets_ts)))

    #     curr_onset_start_ind = 0

    #     for stim_ind, stim_n in enumerate(stim_ns):
    #         curr_stim_grp = self.file_pointer['stimulus/presentation'][stim_n]

    #         if curr_stim_grp['stim_name'].value == 'SparseNoise':
    #             _ = self._analyze_sparse_noise_frames_corticalmapping(curr_stim_grp)
    #         elif curr_stim_grp['stim_name'].value == 'FlashingCircle':
    #             _ = self._analyze_flashing_circle_frames_corticalmapping(curr_stim_grp)
    #         elif curr_stim_grp['stim_name'].value == 'DriftingGratingCircle':
    #             _ = self._analyze_driftig_grating_frames_corticalmapping(curr_stim_grp)
    #         elif curr_stim_grp['stim_name'].value == 'UniformContrast':
    #             _ = self._analyze_uniform_contrast_frames_corticalmapping(curr_stim_grp)
    #         else:
    #             raise LookupError('Do not understand stimulus type: {}.'.format(stim_n))

    #         curr_onset_arr, curr_data_format, curr_description, pooled_onsets = _

    #         curr_onset_ts = onsets_ts[curr_onset_start_ind: curr_onset_start_ind + curr_onset_arr.shape[0]]

    #         curr_onset = self.create_timeseries('TimeSeries', stim_n, modality='other')
    #         curr_onset.set_data(curr_onset_arr, unit='', conversion=np.nan, resolution=np.nan)
    #         curr_onset.set_time(curr_onset_ts)
    #         curr_onset.set_description(curr_description)
    #         curr_onset.set_value('data_format', curr_data_format)
    #         curr_onset.set_path('processing/StimulusOnsets')
    #         curr_onset.set_value('stim_name', curr_stim_grp['stim_name'].value)
    #         curr_onset.set_value('background_color', curr_stim_grp['background_color'].value)
    #         curr_onset.set_value('pre_gap_dur_sec', curr_stim_grp['pre_gap_dur_sec'].value)
    #         curr_onset.set_value('post_gap_dur_sec', curr_stim_grp['post_gap_dur_sec'].value)

    #         if curr_stim_grp['stim_name'].value == 'UniformContrast':
    #             curr_onset.set_value('color', curr_stim_grp['color'].value)
    #             curr_onset.finalize()

    #         elif curr_stim_grp['stim_name'].value == 'FlashingCircle':
    #             curr_onset.set_value('iteration', curr_stim_grp['iteration'].value)
    #             curr_onset.finalize()

    #         elif curr_stim_grp['stim_name'].value == 'SparseNoise':
    #             curr_onset.set_value('grid_space', curr_stim_grp['grid_space'].value)
    #             curr_onset.set_value('iteration', curr_stim_grp['iteration'].value)
    #             curr_onset.set_value('probe_frame_num', curr_stim_grp['probe_frame_num'].value)
    #             curr_onset.set_value('probe_height_deg', curr_stim_grp['probe_height_deg'].value)
    #             curr_onset.set_value('probe_orientation', curr_stim_grp['probe_orientation'].value)
    #             curr_onset.set_value('probe_width_deg', curr_stim_grp['probe_width_deg'].value)
    #             curr_onset.set_value('sign', curr_stim_grp['sign'].value)
    #             curr_onset.set_value('subregion_deg', curr_stim_grp['subregion_deg'].value)
    #             curr_onset.finalize()
    #             for curr_sn, curr_sd in pooled_onsets.items():
    #                 curr_s_ts = self.create_timeseries('TimeSeries', curr_sn, modality='other')
    #                 curr_s_ts.set_data([], unit='', conversion=np.nan, resolution=np.nan)
    #                 curr_s_ts.set_time(curr_onset_ts[curr_sd['onset_ind']])
    #                 curr_s_ts.set_value('azimuth_deg', curr_sd['azi'])
    #                 curr_s_ts.set_value('altitude_deg', curr_sd['alt'])
    #                 curr_s_ts.set_value('sign', curr_sd['sign'])
    #                 curr_s_ts.set_path('processing/StimulusOnsets/' + stim_n + '/square_timestamps')
    #                 curr_s_ts.finalize()

    #         elif curr_stim_grp['stim_name'].value == 'DriftingGratingCircle':
    #             curr_onset.set_value('iteration', curr_stim_grp['iteration'].value)
    #             curr_onset.set_value('mid_gap_dur_sec', curr_stim_grp['mid_gap_dur_sec'].value)
    #             curr_onset.set_value('center_altitude_deg', curr_stim_grp['center_altitude_deg'].value)
    #             curr_onset.set_value('center_azimuth_deg', curr_stim_grp['center_azimuth_deg'].value)
    #             curr_onset.set_value('contrast_list', curr_stim_grp['contrast_list'].value)
    #             curr_onset.set_value('direction_list', curr_stim_grp['direction_list'].value)
    #             curr_onset.set_value('radius_list', curr_stim_grp['radius_list'].value)
    #             curr_onset.set_value('spatial_frequency_list', curr_stim_grp['spatial_frequency_list'].value)
    #             curr_onset.set_value('temporal_frequency_list', curr_stim_grp['temporal_frequency_list'].value)
    #             curr_onset.finalize()
    #             for curr_gn, curr_gd in pooled_onsets.items():
    #                 curr_g_ts = self.create_timeseries('TimeSeries', curr_gn, modality='other')
    #                 curr_g_ts.set_data([], unit='', conversion=np.nan, resolution=np.nan)
    #                 curr_g_ts.set_time(curr_onset_ts[curr_gd['onset_ind']])
    #                 curr_g_ts.set_value('sf_cyc_per_deg', curr_gd['sf'])
    #                 curr_g_ts.set_value('tf_hz', curr_gd['tf'])
    #                 curr_g_ts.set_value('direction_arc', curr_gd['dir'])
    #                 curr_g_ts.set_value('contrast', curr_gd['con'])
    #                 curr_g_ts.set_value('radius_deg', curr_gd['r'])
    #                 curr_g_ts.set_path('processing/StimulusOnsets/' + stim_n + '/grating_timestamps')
    #                 curr_g_ts.finalize()

    #         curr_onset_start_ind = curr_onset_start_ind + curr_onset_arr.shape[0]

    # @staticmethod
    # def _analyze_sparse_noise_frames_corticalmapping(sn_grp):
    #     """
    #     analyze sparse noise display frames saved in '/stimulus/presentation', extract information about onset of
    #     each displayed square:

    #     return: all_squares: 2-d array, each line is a displayed square in sparse noise, each column is a feature of
    #                          a particular square, squares follow display order
    #             data_format: str, description of the column structure of each square
    #             description: str,
    #             pooled_squares: dict, squares with same location and sign are pooled together.
    #                             keys: 'square_00000', 'square_00001', 'square_00002' ... each represents a unique
    #                                   square.
    #                             values: dict, {
    #                                            'azi': <azimuth of the square>,
    #                                            'alt': <altitude of the square>,
    #                                            'sign': <sign of the square>,
    #                                            'onset_ind': list of indices of the appearances of current square in
    #                                                         in "all_squares", to be aligned with to photodiode onset
    #                                                         timestamps
    #                                            }
    #     """

    #     if sn_grp['stim_name'].value != 'SparseNoise':
    #         raise NameError('The input stimulus should be "SparseNoise".')

    #     frames = sn_grp['data'].value
    #     frames = [tuple(x) for x in frames]
    #     dtype = [('isDisplay', int), ('azimuth', float), ('altitude', float), ('sign', int), ('isOnset', int)]
    #     frames = np.array(frames, dtype=dtype)

    #     all_squares = []
    #     for i in range(len(frames)):
    #         if frames[i]['isDisplay'] == 1 and \
    #                 (i == 0 or (frames[i - 1]['isOnset'] == -1 and frames[i]['isOnset'] == 1)):
    #             all_squares.append(np.array((i, frames[i]['azimuth'], frames[i]['altitude'], frames[i]['sign']),
    #                                         dtype=np.float32))

    #     all_squares = np.array(all_squares)

    #     pooled_squares = {}
    #     unique_squares = list(set([tuple(x[1:]) for x in all_squares]))
    #     for i, unique_square in enumerate(unique_squares):
    #         curr_square_n = 'square_' + ft.int2str(i, 5)
    #         curr_azi = unique_square[0]
    #         curr_alt = unique_square[1]
    #         curr_sign = unique_square[2]
    #         curr_onset_ind = []
    #         for j, give_square in enumerate(all_squares):
    #             if np.array_equal(give_square[1:], unique_square):
    #                 curr_onset_ind.append(j)
    #         pooled_squares.update({curr_square_n: {'azi': curr_azi,
    #                                                'alt': curr_alt,
    #                                                'sign': curr_sign,
    #                                                'onset_ind': curr_onset_ind}})
    #     all_squares = np.array(all_squares)
    #     data_format = ['display frame indices for the onset of each square', 'azimuth of each square',
    #                    'altitude of each square', 'sign of each square']
    #     description = 'TimeSeries of sparse noise square onsets. Stimulus generated by ' \
    #                   'corticalmapping.VisualStim.SparseNoise class.'
    #     return all_squares, data_format, description, pooled_squares

    # @staticmethod
    # def _analyze_driftig_grating_frames_corticalmapping(dg_grp):
    #     """
    #     analyze drifting grating display frames saved in '/stimulus/presentation', extract information about onset of
    #     each displayed grating:

    #     return: all_gratings: 2-d array, each line is a displayed square in sparse noise, each column is a feature of
    #                          a particular square, squares follow display order
    #             data_format: str, description of the column structure of each grating
    #             description: str,
    #             pooled_squares: dict, gratings with same parameters are pooled together.
    #                             keys: 'grating_00000', 'grating_00001', 'grating_00002' ... each represents a unique
    #                                   grating.
    #                             values: dict, {
    #                                            'sf': <spatial frequency of the grating>,
    #                                            'tf': <temporal frequency of the grating>,
    #                                            'direction': <moving direction of the grating>,
    #                                            'contrast': <contrast of the grating>,
    #                                            'radius': <radius of the grating>,
    #                                            'azi': <azimuth of the grating center>
    #                                            'alt': <altitude of the grating center>
    #                                            'onset_ind': list of indices of the appearances of current square in
    #                                                         in "all_squares", to be aligned with to photodiode onset
    #                                                         timestamps
    #                                            }
    #     """
    #     if dg_grp['stim_name'].value != 'DriftingGratingCircle':
    #         raise NameError('The input stimulus should be "DriftingGratingCircle".')

    #     frames = dg_grp['data'].value

    #     all_gratings = []
    #     for i in range(len(frames)):
    #         if frames[i][8] == 1 and (i == 0 or (frames[i - 1][8] == -1)):
    #             all_gratings.append(np.array((i, frames[i][2], frames[i][3], frames[i][4], frames[i][5], frames[i][6]),
    #                                          dtype=np.float32))

    #     all_gratings = np.array(all_gratings)

    #     pooled_gratings = {}
    #     unique_gratings = list(set([tuple(x[1:]) for x in all_gratings]))
    #     for i, unique_grating in enumerate(unique_gratings):
    #         curr_grating_n = 'grating_' + ft.int2str(i, 5)
    #         curr_sf = unique_grating[0]
    #         curr_tf = unique_grating[1]
    #         curr_dir = unique_grating[2]
    #         curr_con = unique_grating[3]
    #         curr_r = unique_grating[4]
    #         curr_onset_ind = []
    #         for j, given_grating in enumerate(all_gratings):
    #             if np.array_equal(given_grating[1:], unique_grating):
    #                 curr_onset_ind.append(j)
    #         pooled_gratings.update({curr_grating_n: {'sf': curr_sf,
    #                                                  'tf': curr_tf,
    #                                                  'dir': curr_dir,
    #                                                  'con': curr_con,
    #                                                  'r': curr_r,
    #                                                  'onset_ind': curr_onset_ind}})
    #     data_format = ['display frame indices for the onset of each square', 'spatial frequency (cyc/deg)',
    #                    'temporal frequency (Hz)', 'moving direction (arc)', 'contrast (%)', 'radius (deg)']
    #     description = 'TimeSeries of drifting grating circle onsets. Stimulus generated by ' \
    #                   'corticalmapping.VisualStim.SparseNoise class.'
    #     return all_gratings, data_format, description, pooled_gratings

    # @staticmethod
    # def _analyze_flashing_circle_frames_corticalmapping(fc_grp):
    #     """
    #     analyze flashing circle display frames saved in '/stimulus/presentation', extract information about onset of
    #     each displayed circle:

    #     return: all_circles: 2-d array, each line is the onset of displayed circle, each column is a feature of
    #                          that circle, circles follow the display order
    #             data_format: str, description of the column structure of each circle
    #             description: str,
    #             pooled_circles: None
    #     """

    #     if fc_grp['stim_name'].value != 'FlashingCircle':
    #         raise NameError('The input stimulus should be "FlashingCircle".')

    #     frames = fc_grp['data'][:, 0]
    #     azi = fc_grp['center_azimuth_deg'].value
    #     alt = fc_grp['center_altitude_deg'].value
    #     color_c = fc_grp['center_color'].value
    #     color_b = fc_grp['background_color'].value
    #     radius = fc_grp['radius_deg'].value

    #     all_cirlces = []
    #     for i in range(len(frames)):
    #         if frames[i] == 1 and (i == 0 or (frames[i - 1] == 0)):
    #             all_cirlces.append(np.array((i, azi, alt, color_c, color_b, radius), dtype=np.float32))

    #     all_cirlces = np.array(all_cirlces)
    #     data_format = ['display frame indices for the onset of each circle', 'center_azimuth_deg',
    #                    'center_altitude_deg', 'center_color', 'background_color', 'radius_deg']
    #     description = 'TimeSeries of flashing circle onsets. Stimulus generated by ' \
    #                   'corticalmapping.VisualStim.SparseNoise class.'
    #     return all_cirlces, data_format, description, None

    # @staticmethod
    # def _analyze_uniform_contrast_frames_corticalmapping(uc_grp):

    #     if uc_grp['stim_name'].value != 'UniformContrast':
    #         raise NameError('The input stimulus should be "UniformContrast".')

    #     onset_array = np.array([])
    #     data_format = ''
    #     description = 'TimeSeries of uniform contrast stimulus. No onset information. Stimulus generated by ' \
    #                   'corticalmapping.VisualStim.UniformContrast class.'
    #     return onset_array, data_format, description, {}

    # def _add_sparse_noise_stimulus_corticalmapping(self, log_dict, display_order):

    #     stim_name = log_dict['stimulation']['stimName']

    #     if stim_name != 'SparseNoise':
    #         raise ValueError('stimulus was not sparse noise.')

    #     display_frames = log_dict['presentation']['displayFrames']
    #     time_stamps = log_dict['presentation']['timeStamp']

    #     frame_array = np.empty((len(display_frames), 5), dtype=np.float32)
    #     for i, frame in enumerate(display_frames):
    #         if frame[0] == 0:
    #             frame_array[i] = np.array([0, np.nan, np.nan, np.nan, frame[3]])
    #         elif frame[0] == 1:
    #             frame_array[i] = np.array([1, frame[1][0], frame[1][1], frame[2], frame[3]])
    #         else:
    #             raise ValueError('The first value of ' + str(i) + 'th display frame: ' + str(frame) + ' should' + \
    #                              ' be only 0 or 1.')
    #     stim = self.create_timeseries('TimeSeries', ft.int2str(display_order, 2) + '_' + stim_name,
    #                                   'stimulus')
    #     stim.set_time(time_stamps)
    #     stim.set_data(frame_array, unit='', conversion=np.nan, resolution=np.nan)
    #     stim.set_comments('the timestamps of displayed frames (saved in data) are referenced to the start of'
    #                       'this particular display, not the master time clock. For more useful timestamps, check'
    #                       '/processing for aligned photodiode onset timestamps.')
    #     stim.set_description('data formatting: [isDisplay (0:gap; 1:display), azimuth (deg), altitude (deg), '
    #                          'polarity (from -1 to 1), indicatorColor (for photodiode, from -1 to 1)]')
    #     stim.set_value('data_formatting', ['isDisplay', 'azimuth', 'altitude', 'polarity', 'indicatorColor'])
    #     stim.set_value('background_color', log_dict['stimulation']['background'])
    #     stim.set_value('pre_gap_dur_sec', log_dict['stimulation']['preGapDur'])
    #     stim.set_value('post_gap_dur_sec', log_dict['stimulation']['postGapDur'])
    #     stim.set_value('iteration', log_dict['stimulation']['iteration'])
    #     stim.set_value('sign', log_dict['stimulation']['sign'])
    #     stim.set_value('probe_width_deg', log_dict['stimulation']['probeSize'][0])
    #     stim.set_value('probe_height_deg', log_dict['stimulation']['probeSize'][1])
    #     stim.set_value('subregion_deg', log_dict['stimulation']['subregion'])
    #     try:
    #         stim.set_value('probe_orientation', log_dict['stimulation']['probeOrientation'])
    #     except KeyError:
    #         stim.set_value('probe_orientation', log_dict['stimulation']['probeOrientationt'])
    #     stim.set_value('stim_name', log_dict['stimulation']['stimName'])
    #     stim.set_value('grid_space', log_dict['stimulation']['gridSpace'])
    #     stim.set_value('probe_frame_num', log_dict['stimulation']['probeFrameNum'])
    #     stim.set_source('corticalmapping.VisualStim.SparseNoise for stimulus; '
    #                     'corticalmapping.VisualStim.DisplaySequence for display')
    #     stim.finalize()

    # def _add_flashing_circle_stimulus_corticalmapping(self, log_dict, display_order):

    #     stim_name = log_dict['stimulation']['stimName']

    #     if stim_name != 'FlashingCircle':
    #         raise ValueError('stimulus should be flashing circle.')

    #     display_frames = log_dict['presentation']['displayFrames']
    #     time_stamps = log_dict['presentation']['timeStamp']

    #     frame_array = np.empty((len(display_frames), 2), dtype=np.int8)
    #     for i, frame in enumerate(display_frames):
    #         if frame[0] == 0 or frame[0] == 1:
    #             frame_array[i] = np.array([frame[0], frame[3]])
    #         else:
    #             raise ValueError('The first value of ' + str(i) + 'th display frame: ' + str(frame) + ' should' + \
    #                              ' be only 0 or 1.')
    #     stim = self.create_timeseries('TimeSeries', ft.int2str(display_order, 2) + '_' + stim_name,
    #                                   'stimulus')
    #     stim.set_time(time_stamps)
    #     stim.set_data(frame_array, unit='', conversion=np.nan, resolution=np.nan)
    #     stim.set_comments('the timestamps of displayed frames (saved in data) are referenced to the start of'
    #                       'this particular display, not the master time clock. For more useful timestamps, check'
    #                       '/processing for aligned photodiode onset timestamps.')
    #     stim.set_description('data formatting: [isDisplay (0:gap; 1:display), '
    #                          'indicatorColor (for photodiode, from -1 to 1)]')
    #     stim.set_value('data_formatting', ['isDisplay', 'indicatorColor'])
    #     stim.set_source('corticalmapping.VisualStim.FlashingCircle for stimulus; '
    #                     'corticalmapping.VisualStim.DisplaySequence for display')
    #     stim.set_value('radius_deg', log_dict['stimulation']['radius'])
    #     stim.set_value('center_azimuth_deg', log_dict['stimulation']['center'][0])
    #     stim.set_value('center_altitude_deg', log_dict['stimulation']['center'][1])
    #     stim.set_value('center_color', log_dict['stimulation']['color'])
    #     stim.set_value('background_color', log_dict['stimulation']['background'])
    #     stim.set_value('stim_name', log_dict['stimulation']['stimName'])
    #     stim.set_value('pre_gap_dur_sec', log_dict['stimulation']['preGapDur'])
    #     stim.set_value('post_gap_dur_sec', log_dict['stimulation']['postGapDur'])
    #     stim.set_value('iteration', log_dict['stimulation']['iteration'])
    #     stim.finalize()

    # def _add_uniform_contrast_stimulus_corticalmapping(self, log_dict, display_order):

    #     stim_name = log_dict['stimulation']['stimName']

    #     if stim_name != 'UniformContrast':
    #         raise ValueError('stimulus should be uniform contrast.')

    #     display_frames = log_dict['presentation']['displayFrames']
    #     time_stamps = log_dict['presentation']['timeStamp']

    #     frame_array = np.array(display_frames, dtype=np.int8)

    #     stim = self.create_timeseries('TimeSeries', ft.int2str(display_order, 2) + '_' + stim_name,
    #                                   'stimulus')
    #     stim.set_time(time_stamps)
    #     stim.set_data(frame_array, unit='', conversion=np.nan, resolution=np.nan)
    #     stim.set_comments('the timestamps of displayed frames (saved in data) are referenced to the start of'
    #                       'this particular display, not the master time clock. For more useful timestamps, check'
    #                       '/processing for aligned photodiode onset timestamps.')
    #     stim.set_description('data formatting: [isDisplay (0:gap; 1:display), '
    #                          'indicatorColor (for photodiode, from -1 to 1)]')
    #     stim.set_value('data_formatting', ['isDisplay', 'indicatorColor'])
    #     stim.set_source('corticalmapping.VisualStim.UniformContrast for stimulus; '
    #                     'corticalmapping.VisualStim.DisplaySequence for display')
    #     stim.set_value('color', log_dict['stimulation']['color'])
    #     stim.set_value('background_color', log_dict['stimulation']['background'])
    #     stim.set_value('pre_gap_dur_sec', log_dict['stimulation']['preGapDur'])
    #     stim.set_value('post_gap_dur_sec', log_dict['stimulation']['postGapDur'])
    #     stim.set_value('stim_name', log_dict['stimulation']['stimName'])
    #     stim.finalize()

    # def _add_drifting_grating_circle_stimulus_corticalmapping(self, log_dict, display_order):

    #     stim_name = log_dict['stimulation']['stimName']

    #     if stim_name != 'DriftingGratingCircle':
    #         raise ValueError('stimulus should be drifting grating circle.')

    #     display_frames = log_dict['presentation']['displayFrames']
    #     time_stamps = log_dict['presentation']['timeStamp']

    #     frame_array = np.array(display_frames)
    #     frame_array[np.equal(frame_array, None)] = np.nan
    #     frame_array = frame_array.astype(np.float32)

    #     stim = self.create_timeseries('TimeSeries', ft.int2str(display_order, 2) + '_' + stim_name,
    #                                   'stimulus')
    #     stim.set_time(time_stamps)
    #     stim.set_data(frame_array, unit='', conversion=np.nan, resolution=np.nan)
    #     stim.set_comments('the timestamps of displayed frames (saved in data) are referenced to the start of '
    #                       'this particular display, not the master time clock. For more useful timestamps, check '
    #                       '"/processing" for aligned photodiode onset timestamps.')
    #     stim.set_description('data formatting: [isDisplay (0:gap; 1:display), '
    #                          'firstFrameInCycle (first frame in cycle:1, rest display frames: 0), '
    #                          'spatialFrequency (cyc/deg), '
    #                          'temporalFrequency (Hz), '
    #                          'direction ([0, 2*pi)), '
    #                          'contrast ([0, 1]), '
    #                          'radius (deg), '
    #                          'phase ([0, 2*pi)'
    #                          'indicatorColor (for photodiode, from -1 to 1)]. '
    #                          'for gap frames, the 2ed to 8th elements should be np.nan.')
    #     stim.set_value('data_formatting', ['isDisplay', 'firstFrameInCycle', 'spatialFrequency', 'temporalFrequency',
    #                                        'direction', 'contrast', 'radius', 'phase', 'indicatorColor'])
    #     stim.set_value('spatial_frequency_list', log_dict['stimulation']['sf_list'])
    #     stim.set_value('temporal_frequency_list', log_dict['stimulation']['tf_list'])
    #     stim.set_value('direction_list', log_dict['stimulation']['dire_list'])
    #     stim.set_value('contrast_list', log_dict['stimulation']['con_list'])
    #     stim.set_value('radius_list', log_dict['stimulation']['size_list'])
    #     stim.set_value('center_azimuth_deg', log_dict['stimulation']['center'][0])
    #     stim.set_value('center_altitude_deg', log_dict['stimulation']['center'][1])
    #     stim.set_value('pre_gap_dur_sec', log_dict['stimulation']['preGapDur'])
    #     stim.set_value('post_gap_dur_sec', log_dict['stimulation']['postGapDur'])
    #     stim.set_value('mid_gap_dur_sec', log_dict['stimulation']['midGapDur'])
    #     stim.set_value('background_color', log_dict['stimulation']['background'])
    #     stim.set_value('iteration', log_dict['stimulation']['iteration'])
    #     stim.set_value('stim_name', log_dict['stimulation']['stimName'])
    #     stim.set_value('sweep_dur_sec', log_dict['stimulation']['blockDur'])
    #     stim.set_source('corticalmapping.VisualStim.DriftingGratingCircle for stimulus; '
    #                     'corticalmapping.VisualStim.DisplaySequence for display')
    #     stim.set_value('background_color', log_dict['stimulation']['background'])
    #     stim.finalize()

    # def _add_drifting_checker_board_stimulus_corticalmapping(self, log_dict, display_order):

    #     stim_name = log_dict['stimulation']['stimName']

    #     if stim_name != 'KSstimAllDir':
    #         raise ValueError('stimulus should be drifting checker board all directions.')

    #     display_frames = log_dict['presentation']['displayFrames']
    #     time_stamps = log_dict['presentation']['timeStamp']

    #     display_frames = [list(f) for f in display_frames]

    #     for i in range(len(display_frames)):
    #         if display_frames[i][4] == 'B2U':
    #             display_frames[i][4] = 0
    #         elif display_frames[i][4] == 'U2B':
    #             display_frames[i][4] = 1
    #         elif display_frames[i][4] == 'L2R':
    #             display_frames[i][4] = 2
    #         elif display_frames[i][4] == 'R2L':
    #             display_frames[i][4] = 3

    #     frame_array = np.array(display_frames)
    #     frame_array[np.equal(frame_array, None)] = np.nan
    #     frame_array = frame_array.astype(np.float32)

    #     stim = self.create_timeseries('TimeSeries', ft.int2str(display_order, 2) + '_' + stim_name,
    #                                   'stimulus')
    #     stim.set_time(time_stamps)
    #     stim.set_data(frame_array, unit='', conversion=np.nan, resolution=np.nan)
    #     stim.set_comments('the timestamps of displayed frames (saved in data) are referenced to the start of'
    #                       'this particular display, not the master time clock. For more useful timestamps, check'
    #                       '/processing for aligned photodiode onset timestamps.')
    #     stim.set_description('data formatting: [isDisplay (0:gap; 1:display), '
    #                          'square polarity (1: not reversed; -1: reversed), '
    #                          'sweeps, ind, index in sweep table, '
    #                          'indicatorColor (for photodiode, from -1 to 1)]. '
    #                          'direction (B2U: 0, U2B: 1, L2R: 2, R2L: 3), '
    #                          'for gap frames, the 2ed to 3th elements should be np.nan.')
    #     stim.set_value('data_formatting',
    #                    ['isDisplay', 'squarePolarity', 'sweepIndex', 'indicatorColor', 'sweepDirection'])
    #     stim.set_source('corticalmapping.VisualStim.KSstimAllDir for stimulus; '
    #                     'corticalmapping.VisualStim.DisplaySequence for display')
    #     stim.set_value('background_color', log_dict['stimulation']['background'])
    #     stim.set_value('pre_gap_dur_sec', log_dict['stimulation']['preGapDur'])
    #     stim.set_value('post_gap_dur_sec', log_dict['stimulation']['postGapDur'])
    #     stim.set_value('iteration', log_dict['stimulation']['iteration'])
    #     stim.set_value('stim_name', log_dict['stimulation']['stimName'])
    #     stim.finalize()

    #     display_info = hl.analysisMappingDisplayLog(display_log=log_dict)
    #     display_grp = self.file_pointer['processing'].create_group('mapping_display_info')
    #     display_grp.attrs['description'] = 'This group saves the useful infomation about the retiotopic mapping visual' \
    #                                        'stimulation (drifting checker board sweeps in all directions). Generated ' \
    #                                        'by the corticalmapping.HighLevel.analysisMappingDisplayLog() function.'
    #     for direction, value in display_info.items():
    #         dir_grp = display_grp.create_group(direction)
    #         dir_grp.attrs['description'] = 'group containing the relative information about all sweeps in a particular' \
    #                                        'sweep direction. B: bottom, U: up, L: nasal, R: temporal (for stimulus to' \
    #                                        'the right eye)'
    #         ind_dset = dir_grp.create_dataset('onset_index', data=value['ind'])
    #         ind_dset.attrs['description'] = 'indices of sweeps of current direction in the whole experiment'
    #         st_dset = dir_grp.create_dataset('start_time', data=value['startTime'])
    #         st_dset.attrs['description'] = 'sweep start time relative to stimulus onset (second)'
    #         sd_dset = dir_grp.create_dataset('sweep_duration', data=value['sweepDur'])
    #         sd_dset.attrs['description'] = 'sweep duration (second)'
    #         equ_dset = dir_grp.create_dataset('phase_retinotopy_equation', data=[value['slope'], value['intercept']])
    #         equ_dset.attrs['description'] = 'the linear equation to transform fft phase into retinotopy visual degrees.' \
    #                                         'degree = phase * slope + intercept'
    #         equ_dset.attrs['data_format'] = ['slope', 'intercept']

    # # ===========================corticalmapping visual stimuli related (non-indexed display)===========================


    # ============================================eye tracking related==================================================
    def add_eyetracking_eyetracker3(self, ts_path='', pupil_x=None, pupil_y=None, pupil_area=None, module_name='eye_tracking',
                                    unit='unknown', side='leftright_unknown', comments='', description='', source='',
                                    pupil_shape=None, pupil_shape_meta=None):
        """
        add eyetrackin data as a module named 'eye_tracking'
        :param ts_path: str, timestamp path in the nwb file
        :param pupil_x: 1-d array, horizontal position of pupil center
        :param pupil_y: 1-d array, vertical position of pupil center
        :param pupil_area: 1-d array, area of detected pupil
        :param module_name: str, module name to be created
        :param unit: str, the unit of pupil_x, pupil_y, the unit of pupil_area should be <unit>^2
        :param side: str, side of the eye, 'left' or 'right'
        :param comments: str
        :param description: str
        :param source: str
        :return:
        """

        if ts_path not in list(self.file_pointer['acquisition/timeseries'].keys()):
            print('Cannot find field "{}" in "acquisition/timeseries".'.format(ts_path))
            return
        else:
            ts = self.file_pointer['acquisition/timeseries'][ts_path]['timestamps'][()]

        ts_num = len(ts)
        print('number of eyet racking timestamps: {}'.format(ts.shape))

        ts_num_min = ts_num

        if pupil_x is not None:
            if pupil_x.shape[0] != ts_num:
                print('length of pupil_x ({}) is different from the number'
                      ' of timestamps ({}).'.format(pupil_x.shape[0], ts_num))
                ts_num_min = min([ts_num_min, pupil_x.shape[0]])

        if pupil_y is not None:
            if pupil_y.shape[0] != ts_num:
                print('length of pupil_y ({}) is different from the number'
                      ' of timestamps ({}).'.format(pupil_area.shape[0], ts_num))
                ts_num_min = min([ts_num_min, pupil_y.shape[0]])

        if pupil_area is not None:
            if pupil_area.shape[0] != ts_num:
                print('length of pupil_area ({}) is different from the number'
                      ' of timestamps ({}).'.format(pupil_area.shape[0], ts_num))
                ts_num_min = min([ts_num_min, pupil_area.shape[0]])

        ts_to_add = ts[0:ts_num_min]

        pupil_series = self.create_timeseries('TimeSeries', name='eyetracking', modality='other')
        pupil_series.set_data([], unit='', conversion=np.nan, resolution=np.nan)
        pupil_series.set_time(ts_to_add)

        if pupil_x is not None:
            pupil_series.set_value('pupil_x', pupil_x[0:ts_num_min])

        if pupil_y is not None:
            pupil_series.set_value('pupil_y', pupil_y[0:ts_num_min])

        if pupil_area is not None:
            pupil_series.set_value('pupil_area', pupil_area[0:ts_num_min])

        if pupil_shape is not None:
            pupil_series.set_value('pupil_shape', pupil_shape[0:ts_num_min, :])

        if pupil_shape_meta is not None:
            pupil_series.set_value('pupil_shape_meta', pupil_shape_meta)

        pupil_series.set_value('unit', 'pupil_x: {}; pupil_y: {}; pupil_area: {} ^ 2'.format(unit, unit, unit))
        pupil_series.set_value('side', side)
        pupil_series.set_comments(comments)
        pupil_series.set_description(description)
        pupil_series.set_source(source)

        et_mod = self.create_module('{}_{}'.format(module_name, side))
        et_interf = et_mod.create_interface("PupilTracking")
        et_interf.add_timeseries(pupil_series)
        pupil_series.finalize()
        et_interf.finalize()
        et_mod.finalize()

    def add_eyetracking_general(self, ts_path='', data=None, module_name='eye_tracking',
                                side='leftright_unknown', comments='', description='', source=''):
        """

        :param ts_path:
        :param data:
        :param module_name:
        :param side:
        :param comments:
        :param description:
        :param source:
        :return:
        """

        if data is None:
            return

        ts = self.file_pointer['acquisition/timeseries'][ts_path]['timestamps'][()]
        ts_num_min = min([len(ts), data.shape[0]])
        ts_to_add = ts[0:ts_num_min]

        et_series = self.create_timeseries('TimeSeries', name='eyetracking', modality='other')
        et_series.set_data(data, unit='', conversion=np.nan, resolution=np.nan)
        et_series.set_time(ts_to_add)
        et_series.set_comments(comments)
        et_series.set_description(description)
        et_series.set_source(source)
        et_series.set_value('side', side)

        et_mod = self.create_module('{}_{}'.format(module_name, side))
        et_interf = et_mod.create_interface("PupilTracking")
        et_interf.add_timeseries(et_series)
        et_series.finalize()
        et_interf.finalize()
        et_mod.finalize()

    # ============================================eye tracking related==================================================


if __name__ == '__main__':
    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # open_ephys_folder = r"E:\data\2016-07-19-160719-M256896\100_spontaneous_2016-07-19_09-45-06_Jun"
    # rf = RecordedFile(tmp_path, identifier='', description='')
    # rf.add_open_ephys_data(open_ephys_folder, '100', ['wf_read', 'wf_trigger', 'visual_frame'])
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # rf = RecordedFile(tmp_path)
    # rf.add_general()
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # rf = RecordedFile(tmp_path)
    # rf.add_acquisition_image('surface_vas_map', np.zeros((10, 10)), description='surface vasculature map')
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # data_path = r"E:\data\2016-07-25-160722-M256896\processed_1"
    # rf = RecordedFile(tmp_path)
    # rf.add_phy_template_clusters(folder=data_path, module_name='LGN')
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # data_path = r"E:\data\2016-07-25-160722-M256896\processed_1"
    # rf = RecordedFile(tmp_path)
    # rf.add_kilosort_clusters(folder=data_path, module_name='LGN_kilosort')
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # log_path = r"E:\data\2016-06-29-160610-M240652-Ephys\101_160610172256-SparseNoise-M240652-Jun-0-" \
    #            r"notTriggered-complete.pkl"
    # rf = RecordedFile(tmp_path)
    # rf.add_visual_stimulation_corticalmapping(log_path)
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # log_path = r"\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData\161017-M274376-FlashingCircle" \
    #            r"\161017162026-FlashingCircle-M274376-Sahar-101-Triggered-complete.pkl"
    # rf = RecordedFile(tmp_path)
    # rf.add_visual_stimulation_corticalmapping(log_path, display_order=1)
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # log_paths = [r"\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData\161017-M274376-FlashingCircle\161017162026-FlashingCircle-M274376-Sahar-101-Triggered-complete.pkl",
    #              r"E:\data\2016-06-29-160610-M240652-Ephys\101_160610172256-SparseNoise-M240652-Jun-0-notTriggered-complete.pkl",]
    # rf = RecordedFile(tmp_path)
    # rf.add_visual_stimuli_corticalmapping(log_paths)
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # log_paths = [r"C:\data\sequence_display_log\161018164347-UniformContrast-MTest-Jun-255-notTriggered-complete.pkl"]
    # rf = RecordedFile(tmp_path)
    # rf.add_visual_stimuli_corticalmapping(log_paths)
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # # log_paths = [r"C:\data\sequence_display_log\160205131514-ObliqueKSstimAllDir-MTest-Jun-255-notTriggered-incomplete.pkl"]
    # log_paths = [r"C:\data\sequence_display_log\161018174812-DriftingGratingCircle-MTest-Jun-255-notTriggered-complete.pkl"]
    # rf = RecordedFile(tmp_path)
    # rf.add_visual_stimuli_corticalmapping(log_paths)
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # img_data_path = r"E:\data\python_temp_folder\img_data.hdf5"
    # img_data = h5py.File(img_data_path)
    # dset = img_data.create_dataset('data', data=np.random.rand(1000, 1000, 100))
    # dset.attrs['conversion'] = np.nan
    # dset.attrs['resolution'] = np.nan
    # dset.attrs['unit'] = ''
    # img_data.close()

    # ts = np.random.rand(1000)
    #
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # rf = RecordedFile(tmp_path)
    # rf.add_acquired_image_series_as_remote_link('test_img', image_file_path=img_data_path, dataset_path='/data',
    #                                             timestamps=ts)
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # rf = RecordedFile(r"D:\data2\thalamocortical_project\method_development\2017-02-25-ephys-software-development"
    #                   r"\test_folder\170302_M292070_100_SparseNoise.nwb")
    # unit = 'unit_00065'
    # wfs = rf.file_pointer['processing/tetrode/UnitTimes'][unit]['template'].value
    # stds = rf.file_pointer['processing/tetrode/UnitTimes'][unit]['template_std'].value
    # x_pos = rf.file_pointer['processing/tetrode/channel_xpos'].value
    # y_pos = rf.file_pointer['processing/tetrode/channel_ypos'].value
    # rf.close()
    # plot_waveforms(wfs, zip(x_pos, y_pos), stds, axes_size=(0.3, 0.3))
    # plt.show()
    # =========================================================================================================

    print('for debug ...')
