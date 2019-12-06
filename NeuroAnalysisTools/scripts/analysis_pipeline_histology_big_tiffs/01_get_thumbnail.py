import os
import numpy as np
import tifffile as tf
import corticalmapping.core.ImageAnalysis as ia

channels = ['DAPI', 'GCaMP', 'mRuby', 'NeuN']
downsample_rate = 0.1

curr_folder = os.path.realpath(os.path.dirname(__file__))
os.chdir(curr_folder)

fns = [f for f in os.listdir(curr_folder) if f[-4:] == '.btf']
fns.sort()
print('\n'.join(fns))

for fn in fns:
    print('\nprocessing {} ...'.format(fn))
    big_img = tf.imread(fn)
    fname = os.path.splitext(fn)[0]
    print('shape: {}'.format(big_img.shape))
    print('dtype: {}'.format(big_img.dtype))

    # comb_img = []

    for chi, chn in enumerate(channels):
        print('\tchannel: {}'.format(chn))
        down_img_ch = ia.rigid_transform_cv2(big_img[chi], zoom=downsample_rate).astype(np.uint16)[::-1, :]
        tf.imsave('thumbnail_{}_{:02d}_{}.tif'.format(fname, chi, chn), down_img_ch)
        # comb_img.append(down_img_ch)

    # comb_img = np.array(comb_img)
    # tf.imsave('{}_downsampled.tif'.format(fname), comb_img)
