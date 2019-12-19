import os
import numpy as np
import tifffile as tf
import NeuroAnalysisTools.core.ImageAnalysis as ia

base_name = '363669_1_01'
save_name = '363669_1_section02.tif'
thumbnail_region = [100, 1125, 1673, 3176]

channels = ['mRuby', 'GCaMP', 'DAPI', 'NeuN']
d_rate = 0.5

thumbnail_d_rate = 0.1 # down sample rate of thumbnail

curr_folder = os.path.realpath(os.path.dirname(__file__))
os.chdir(curr_folder)

big_region = (np.array(thumbnail_region) / thumbnail_d_rate).astype(np.uint64)
print('region in big image: {}'.format(big_region))

thumbnail_fns = [f for f in os.listdir(curr_folder) if base_name in f and f[-4:] == '.tif']
ch_lst = []
for chn in channels:
    curr_chi = [int(f.split('_')[-2]) for f in thumbnail_fns if chn in f]
    if len(curr_chi) != 1:
        raise LookupError
    ch_lst.append(curr_chi[0])

print('channel index list: {}'.format(ch_lst))

big_img = tf.imread(base_name + '.btf')
print('reading the big image: {}.btf ...'.format(base_name))

section_img = []

for ch_i in ch_lst:
    curr_img = big_img[ch_i][::-1, :][big_region[0]: big_region[1], big_region[2]: big_region[3]]
    curr_img = ia.rigid_transform_cv2(curr_img, zoom=d_rate).astype(np.uint16)
    section_img.append(curr_img)

section_img = np.array(section_img)

print('saving {} ...'.format(save_name))
tf.imsave(save_name, section_img)