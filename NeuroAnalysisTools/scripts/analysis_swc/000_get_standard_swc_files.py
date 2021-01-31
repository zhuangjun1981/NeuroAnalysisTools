import os
import NeuroAnalysisTools.SwcTools as st

swc_path = 'M541183.ano_green.swc'
vox_size_x = 0.414
vox_size_y = 0.414
vox_size_z = 0.5
rot_ang = 50

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

swc_f = st.read_swc(swc_path)
swc_f.type = 2 # change all nodes to be axon
swc_f.radius = 1 # set radius of all nodes to be 1
swc_f.y = -swc_f.y # match vaa3d orientation at "zero" rotation
cen = swc_f.get_center()
swc_f.move_to_origin([cen[0], cen[1], 0]) # center x and y


