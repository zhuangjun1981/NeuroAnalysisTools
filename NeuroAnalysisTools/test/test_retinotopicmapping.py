import os
import tifffile as tf
import matplotlib.pyplot as plt

# from ..core import FileTools as ft
# from .. import RetinotopicMapping as rm

import NeuroAnalysisTools.core.FileTools as ft
import NeuroAnalysisTools.RetinotopicMapping as rm

curr_folder = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(curr_folder, 'data')
os.chdir(curr_folder)

vasculature_map = tf.imread(os.path.join(data_folder, 'example_vasculature_map.tif'))
altitude_map = tf.imread(os.path.join(data_folder, 'example_altitude_map.tif'))
azimuth_map = tf.imread(os.path.join(data_folder, 'example_azimuth_map.tif'))
altitude_power_map = tf.imread(os.path.join(data_folder, 'example_altitude_power_map.tif'))
azimuth_power_map = tf.imread(os.path.join(data_folder, 'example_azimuth_power_map.tif'))

params = {
          'phaseMapFilterSigma': 0.5,
          'signMapFilterSigma': 8.,
          'signMapThr': 0.4,
          'eccMapFilterSigma': 15.0,
          'splitLocalMinCutStep': 5.,
          'closeIter': 3,
          'openIter': 3,
          'dilationIter': 15,
          'borderWidth': 1,
          'smallPatchThr': 100,
          'visualSpacePixelSize': 0.5,
          'visualSpaceCloseIter': 15,
          'splitOverlapThr': 1.1,
          'mergeOverlapThr': 0.1
          }

trial = rm.RetinotopicMappingTrial(altPosMap=altitude_map,
                                   aziPosMap=azimuth_map,
                                   altPowerMap=altitude_power_map,
                                   aziPowerMap=azimuth_power_map,
                                   vasculatureMap=vasculature_map,
                                   mouseID='test',
                                   dateRecorded='160612',
                                   comments='This is an example.',
                                   params=params)

trial.processTrial(isPlot=True)
plt.show()

_ = trial.plotFinalPatchBorders2()
plt.show()

names = [
    ['patch01', 'V1'],
    ['patch02', 'PM'],
    ['patch03', 'RL'],
    ['patch04', 'P'],
    ['patch05', 'LM'],
    ['patch06', 'AM'],
    ['patch07', 'LI'],
    ['patch08', 'MMA'],
    ['patch09', 'AL'],
    ['patch10', 'RLL'],
    ['patch11', 'LLA'],
    ['patch13', 'MMP']
]

finalPatchesMarked = dict(trial.finalPatches)

for i, namePair in enumerate(names):
    currPatch = finalPatchesMarked.pop(namePair[0])
    newPatchDict = {namePair[1]: currPatch}
    finalPatchesMarked.update(newPatchDict)

trial.finalPatchesMarked = finalPatchesMarked

_ = trial.plotFinalPatchBorders2()
plt.show()

trialDict = trial.generateTrialDict()

save_path = os.path.join(data_folder, 'example_save_folder.pkl')
ft.saveFile(save_path, trialDict)

os.remove(save_path)
