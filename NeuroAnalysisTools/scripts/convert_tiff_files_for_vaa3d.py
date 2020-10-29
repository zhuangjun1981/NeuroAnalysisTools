import os
import tifffile as tf
from libtiff import TIFFimage

src_p = r"Z:\rabies_tracing_project\M533921" \
        r"\2020-10-12-align-sections\step02_align_sections" \
        r"\201026-M533921-merged-GCaMP.tif"

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

save_folder, save_name = os.path.split(src_p)
save_name = f'{os.path.splitext(save_name)[0]}_libtiff.tif'
save_path = os.path.join(save_folder, save_name)

print(f'reading {src_p} ...')
img = tf.imread(src_p)

print(f'\timage datatype: {img.dtype}.')
print(f'\timage shape: {img.shape}.')

if len(img.shape) > 3:
    raise NotImplementedError(f'pylibtiff cannot convert image with more than 3 dimensions. '
                              f'Current number of dimensions: {len(img.shape)}.')

print(f'\tsaving {save_path} ...')
tiff = TIFFimage(img, description='')
tiff.write_file(save_name, compression='none') # or 'lzw'
del tiff

print('\tdone.')