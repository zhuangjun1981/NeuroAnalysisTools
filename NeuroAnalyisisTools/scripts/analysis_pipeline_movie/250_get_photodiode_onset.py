import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import corticalmapping.NwbTools as nt
import corticalmapping.HighLevel as hl

# photodiode
digitizeThr = 0.15 # deepscope: 0.15 or 0.055, sutter: -0.15
filterSize = 0.01 # deepscope: 0.01, sutter: 0.01
segmentThr = 0.02 # deepscope: 0.02, sutter: 0.01
smallestInterval = 0.05 # deepscope: 0.05, sutter: 0.05

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]

nwb_f = nt.RecordedFile(nwb_fn)
pd, pd_t = nwb_f.get_analog_data(ch_n='analog_photodiode')
fs = 1. / np.mean(np.diff(pd_t))
# print fs

pd_onsets = hl.segmentPhotodiodeSignal(pd, digitizeThr=digitizeThr, filterSize=filterSize,
                                       segmentThr=segmentThr, Fs=fs, smallestInterval=smallestInterval)

raw_input('press enter to continue ...')

pdo_ts = nwb_f.create_timeseries('TimeSeries', 'digital_photodiode_rise', modality='other')
pdo_ts.set_time(pd_onsets)
pdo_ts.set_data([], unit='', conversion=np.nan, resolution=np.nan)
pdo_ts.set_value('digitize_threshold', digitizeThr)
pdo_ts.set_value('filter_size', filterSize)
pdo_ts.set_value('segment_threshold', segmentThr)
pdo_ts.set_value('smallest_interval', smallestInterval)
pdo_ts.set_description('Real Timestamps (master acquisition clock) of photodiode onset. '
                       'Extracted from analog photodiode signal by the function:'
                       'corticalmapping.HighLevel.segmentPhotodiodeSignal() using parameters saved in the'
                       'current timeseries.')
pdo_ts.set_path('/analysis')
pdo_ts.finalize()

nwb_f.close()


