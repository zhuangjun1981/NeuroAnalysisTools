import os
import numpy as np
import matplotlib.pyplot as plt
import corticalmapping.NwbTools as nt
import corticalmapping.core.TimingAnalysis as ta
import corticalmapping.SingleCellAnalysis as sca
import corticalmapping.core.FileTools as ft
import corticalmapping.core.ImageAnalysis as ia
from matplotlib.backends.backend_pdf import PdfPages

stim_name = '001_LocallySparseNoiseRetinotopicMapping'
trace_source = 'f_center_subtracted'
start_time = -0.5
end_time = 2.

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = nt.RecordedFile(nwb_fn)

# deleting existing strf group
if 'STRFs' in nwb_f.file_pointer['analysis']:
    del nwb_f.file_pointer['analysis/STRFs']

probe_grp = nwb_f.file_pointer['analysis/photodiode_onsets/' + stim_name]
probe_ns = probe_grp.keys()
probe_ns.sort()

probe_locations = [[float(pn[3: 9]), float(pn[13: 19])] for pn in probe_ns]
probe_signs = [float(pn[-2:]) for pn in probe_ns]
# print(probe_locations)

plane_ns = nwb_f.file_pointer['processing'].keys()
plane_ns = [pn.split('_')[-1] for pn in plane_ns if 'rois_and_traces_plane' in pn]
plane_ns.sort()
print('\n'.join(plane_ns))

strf_grp = nwb_f.file_pointer['analysis'].create_group('STRFs')

for plane_n in plane_ns:
    print('\ngetting STRFs for {} ...'.format(plane_n))

    roi_ns = nwb_f.file_pointer['processing/rois_and_traces_' + plane_n +
                                '/ImageSegmentation/imaging_plane/roi_list'].value
    roi_ns = [rn for rn in roi_ns if rn[0: 4] == 'roi_']
    roi_ns.sort()
    roi_num = len(roi_ns)

    plane_strf_grp = strf_grp.create_group(plane_n)
    plane_traces = nwb_f.file_pointer['processing/rois_and_traces_' + plane_n + '/Fluorescence/' +
                                      trace_source + '/data'].value
    plane_trace_ts = nwb_f.file_pointer['processing/rois_and_traces_' + plane_n + '/Fluorescence/' +
                                        trace_source + '/timestamps'].value

    plane_mean_frame_dur = np.mean(np.diff(plane_trace_ts))
    plane_chunk_frame_dur = int(np.ceil((end_time - start_time) / plane_mean_frame_dur))
    plane_chunk_frame_start = int(np.floor(start_time / plane_mean_frame_dur))
    plane_t = (np.arange(plane_chunk_frame_dur) + plane_chunk_frame_start) * plane_mean_frame_dur
    print '{}: STRF time axis: \n{}'.format(plane_n, plane_t)

    plane_roi_traces = []
    trigger_ts_lst = []

    for probe_ind, probe_n in enumerate(probe_ns):

        probe_ts = probe_grp[probe_n]['pd_onset_ts_sec'].value
        trigger_ts_lst.append(probe_ts)
        probe_traces = []
        for curr_probe_ts in probe_ts:
            curr_frame_start = ta.find_nearest(plane_trace_ts, curr_probe_ts) + plane_chunk_frame_start
            curr_frame_end = curr_frame_start + plane_chunk_frame_dur
            if curr_frame_start >= 0 and curr_frame_end <= len(plane_trace_ts):
                probe_traces.append(plane_traces[:, curr_frame_start: curr_frame_end])

        plane_roi_traces.append(np.array(probe_traces))
        print('probe: {} / {}; shape: {}'.format(probe_ind + 1, len(probe_ns), np.array(probe_traces).shape))

    # plane_roi_traces = np.array(plane_roi_traces)

    print('saving ...')
    for roi_ind in range(roi_num):

        print "roi: {} / {}".format(roi_ind + 1, roi_num)
        curr_unit_traces = [pt[:, roi_ind, :] for pt in plane_roi_traces]
        curr_unit_traces = [list(t) for t in curr_unit_traces]
        curr_strf = sca.SpatialTemporalReceptiveField(locations=probe_locations,
                                                      signs=probe_signs,
                                                      traces=curr_unit_traces,
                                                      time=plane_t,
                                                      # trigger_ts=trigger_ts_lst,
                                                      trigger_ts=None,
                                                      name='roi_{:04d}'.format(roi_ind),
                                                      trace_data_type=trace_source)

        curr_strf_grp = plane_strf_grp.create_group('strf_roi_{:04d}'.format(roi_ind))
        curr_strf.to_h5_group(curr_strf_grp)

nwb_f.close()


