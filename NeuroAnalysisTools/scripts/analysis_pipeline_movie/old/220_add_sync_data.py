import os
import NeuroAnalysisTools.NwbTools as nt

record_date = '190503'
mouse_id = '439939'
session_id = '110'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

nwb_fn = record_date + '_M' + mouse_id + '_' + session_id + '.nwb'

sync_fn = [f for f in os.listdir(curr_folder) if f[-3:] == '.h5' and record_date in f and 'M' + mouse_id in f]
if len(sync_fn) == 0:
    raise LookupError('Did not find sync .h5 file.')
elif len(sync_fn) > 1:
    raise LookupError('More than one sync .h5 files found.')
else:
    sync_fn = sync_fn[0]

nwb_f = nt.RecordedFile(nwb_fn)
nwb_f.add_sync_data(sync_fn)
nwb_f.close()
