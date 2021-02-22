import os
import matplotlib.pyplot as plt
import NeuroAnalysisTools.SwcTools as st

# source : target pairs for all reconstructions
fn_dict = {'M541172_green_stamp_2021_01_30_20_54.ano.eswc' : "M541172_reconstruction001",
           'M541172_red_stamp_2021_01_30_20_55.ano.eswc'   : "M541172_reconstruction002"}

pia_pix = 18
rot_deg = 110
rot_dir = "CW"

vox_size_x = 0.414
vox_size_y = 0.414
vox_size_z = 0.5


curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

for src_fn, trg_fn in fn_dict.items():

    print(f'\nprocession: {src_fn} ...')

    ff = st.read_swc(src_fn)

    ff.save_swc(f'{trg_fn}_original.swc')

    f = plt.figure(figsize=(10, 5))

    # scale structure in to microns
    ff.scale(scale_x=vox_size_x,
             scale_y=vox_size_y,
             scale_z=vox_size_z, unit='um')

    # set structure type to axon
    ff.type = 2

    # set pia surface to be z=0
    ff.z = ff.z - (pia_pix * vox_size_z)

    # center xy
    cen = ff.get_center()
    ff.move_to_origin([cen[0], cen[1], 0])

    # save
    ff.save_swc(f'{trg_fn}_centered.swc')
    axo = f.add_subplot(121)
    ff.plot_xy_mpl(ax=axo)
    axo.set_title(f'{trg_fn}, before rotation')

    # rotate to standard orientation
    ff.rotate_xy(angle=rot_deg, is_rad=False, rot_center=(0.0, 0.0), direction=rot_dir)

    # save
    ff.save_swc(f'{trg_fn}_rotated.swc')
    axr = f.add_subplot(122)
    ff.plot_xy_mpl(ax=axr)
    axr.set_title(f'{trg_fn}, after rotation')

    plt.show()

    f.savefig(f'{trg_fn}_xy.pdf')
