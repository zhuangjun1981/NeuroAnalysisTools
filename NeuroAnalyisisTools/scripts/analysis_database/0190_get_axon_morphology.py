import os
import h5py
import pandas as pd
import corticalmapping.DatabaseTools as dt

nwb_folder = r"G:\bulk_LGN_database\nwbs"
clu_folder = r"intermediate_results\bouton_clustering\AllStimuli_DistanceThr_1.30"

dfa_fn = "dataframe_190530171338_axon_AllStimuli_DistanceThr_1.30.csv"

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

print('reading csv file ...')
dfa = pd.read_csv(dfa_fn)
print('csv file loaded.')

df_plane = dfa[['date', 'mouse_id', 'plane_n']].drop_duplicates().reset_index()
print(len(df_plane))

axon_morph_dict = {}

ind = 0
for plane_i, plane_row in df_plane.iterrows():
    date = int(plane_row['date'])
    mid = plane_row['mouse_id']
    plane_n = plane_row['plane_n']

    print('{}_{}_{}, {}/{}'.format(date, mid, plane_n, plane_i + 1, len(df_plane)))

    nwb_f = h5py.File(os.path.join(nwb_folder, '{}_{}_110_repacked.nwb'.format(date, mid)), 'r')
    clu_f = h5py.File(os.path.join(clu_folder, '{}_{}_{}_axon_grouping.hdf5'.format(date, mid, plane_n)), 'r')

    curr_dfa = dfa[(dfa['date'] == date) &
                   (dfa['mouse_id'] == mid) &
                   (dfa['plane_n'] == plane_n)].reset_index()

    for axon_i, axon_row in curr_dfa.iterrows():

        axon_morph = dt.get_axon_morphology(clu_f=clu_f, nwb_f=nwb_f, plane_n=plane_n, axon_n=axon_row['roi_n'])
        axon_morph.update({'date': date,
                           'mouse_id': mid,
                           'plane_n': plane_n,
                           'roi_n': axon_row['roi_n']})

        axon_morph_dict[ind] = axon_morph
        ind = ind + 1

dfa_morph = pd.DataFrame.from_dict(axon_morph_dict, orient='index')
print(dfa_morph)

strs = os.path.splitext(dfa_fn)[0].split('_')
save_n = '{}_{}_axon_morphology_{}_{}.csv'.format(strs[0], strs[1], strs[-2], strs[-1])
dfa_morph.to_csv(save_n)
