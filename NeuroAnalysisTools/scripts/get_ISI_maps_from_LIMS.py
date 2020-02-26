import os
import shutil
import tifffile as tf
import numpy as np
import NeuroAnalysisTools.core.ImageAnalysis as ia
import NeuroAnalysisTools.core.PlottingTools as pt
import matplotlib.pyplot as plt

data_folder = r"\\allen\programs\braintv\production\neuralcoding\prod54" \
              r"\specimen_1006364737\isi_experiment_1010158049"

save_prefix = '2020-02-25-M513381'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

def save_png(arr, save_path, figsize=(5, 5), **kwargs):
    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111)
    ax.imshow(arr, **kwargs)
    pt.save_figure_without_borders(f=f, savePath=save_path)


def transform_for_deepscope(arr):

    arr_r = arr.transpose(2, 0, 1)
    arr_r = ia.rigid_transform(arr_r[:, :, ::-1], rotation=-145).astype(np.uint8)
    arr_r = arr_r.transpose(1, 2, 0)
    return arr_r

exp_id = data_folder.split('_')[-1]
overlay_fn = '{}_isi_overlay.tif'.format(exp_id)
tm_fn = '{}_target_map.tif'.format(exp_id)

shutil.copyfile(os.path.join(data_folder, overlay_fn),
                os.path.join(curr_folder, overlay_fn))

shutil.copyfile(os.path.join(data_folder, tm_fn),
                os.path.join(curr_folder, tm_fn))

overlay = tf.imread(overlay_fn)
save_png(overlay, figsize=(5, 5), save_path='{}-ISI.png'.format(save_prefix))

overlay_r = transform_for_deepscope(overlay)
save_png(overlay_r, figsize=(8, 8), save_path='{}-ISI-forDeepscope.png'.format(save_prefix))

target = tf.imread(tm_fn)
save_png(target, figsize=(5, 5), save_path='{}-TargetMap.png'.format(save_prefix))

target_r = transform_for_deepscope(target)
save_png(target_r, figsize=(8, 8), save_path='{}-TargetMap-forDeepscope.png'.format(save_prefix))

