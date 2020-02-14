"""
this is a script version of the jupyter notebook for Allen Institute pipeline eyetracking analysis
https://github.com/AllenInstitute/dlc-eye-tracking/blob/master/DLC%20Eye%20Tracking%20and%20Ellipse%20Fitting.ipynb

packages:
https://github.com/AlexEMG/DeepLabCut

https://github.com/AllenInstitute/dlc-eye-tracking/
by Peter Ledochowitsch, MAT, Allen Institute

This notebook demonstrates the necessary steps to use DeepLabCut for Eye Tracking.
It assumes that DeepLabCut is installed.

This notebook illustrates how to:

do eye and pupil tracking
fit ellipses
generate labeled video
This uses a pretrained model! Make sure that the GPU is available or the kernel might die!


***
this script is not part of the NeuroAnalysisTools analysis scripts
please use a environment with DeepLabCut and TensorFlow installed to run
"""


import os
import tensorflow as tf
import deeplabcut

print('tensorflow version: {}'.format(tf.__version__))
print('deeplabcut version: {}'.format(deeplabcut.__version__))


path_config = r"Z:\rabies_tracing_project\ZZZ_deeplabcut_models" \
              r"\fromPeter\universal_eye_tracking-peterl-2019-07-10\config.yaml"

path_mov = "test_mov_dlc.avi"

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

video_file_path = 'full path to video file'
deeplabcut.analyze_videos(path_config,[path_mov])  #can accept list of video files
deeplabcut.create_labeled_video(path_config,[path_mov]) #can accept list of video files
