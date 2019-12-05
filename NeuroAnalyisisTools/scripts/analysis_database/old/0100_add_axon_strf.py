import os
import h5py
import corticalmapping.DatabaseTools as dt

nwb_folder = "nwbs"
clu_folder = r"intermediate_results\bouton_clustering\AllStimuli_DistanceThr_1.30"
strf_t_win = [-0.5, 2.]

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

clu_fns = [f for f in os.listdir(clu_folder) if f[-5:] == '.hdf5']
clu_fns.sort()
print('total number of planes: {}'.format(len(clu_fns)))

for clu_fi, clu_fn in enumerate(clu_fns):

    date, mid, plane_n, _, _ = clu_fn.split('_')

    print('processing {}_{}_{}, {} / {}'.format(date, mid, plane_n, clu_fi + 1, len(clu_fns)))

    nwb_fn = '{}_{}_110_repacked.nwb'.format(date, mid)
    nwb_f = h5py.File(os.path.join(nwb_folder, nwb_fn), 'r')

    clu_f = h5py.File(os.path.join(clu_folder, clu_fn))

    bc = dt.BoutonClassifier()
    bc.add_axon_strf(nwb_f=nwb_f, clu_f=clu_f, plane_n=plane_n, t_win=strf_t_win, verbose=False)

    nwb_f.close()
    clu_f.close()