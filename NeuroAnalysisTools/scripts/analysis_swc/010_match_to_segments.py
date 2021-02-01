import os
import matplotlib.pyplot as plt
import NeuroAnalysisTools.core.FileTools as ft
import NeuroAnalysisTools.SwcTools as st

seg_folder = 'segments'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

# get all reconstruction swc files
recon_fns = ft.look_for_file_list(source=curr_folder,
                                  identifiers=['reconstruction', 'original'],
                                  file_type='swc',
                                  is_full_path=False)
# print(recon_fns)

recon_dict = {}
for recon_fn in recon_fns:
    recon_dict[os.path.splitext(recon_fn)[0]] = st.read_swc(recon_fn)

# get all segment swc files
seg_fns = ft.look_for_file_list(source=seg_folder,
                                identifiers=[],
                                file_type='swc',
                                is_full_path=False)
seg_fns.sort()
# print(seg_fns)

save_folder = os.path.join(curr_folder, 'matches')
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

for seg_fn in seg_fns:
    curr_seg = st.read_swc(os.path.join(seg_folder, seg_fn))

    f = plt.figure(figsize=(5, 5))
    ax = f.add_subplot(111)
    ax.set_title(seg_fn)

    for recon_n, recon_swc in recon_dict.items():
        min_diss = recon_swc.get_min_distances(curr_seg)
        ax.hist(min_diss, range=[0, 200], bins=20, label=recon_n, alpha=0.5)

    ax.legend()
    plt.show()

    f.savefig(os.path.join(save_folder, f'{seg_fn}_match.pdf'))
    del f
