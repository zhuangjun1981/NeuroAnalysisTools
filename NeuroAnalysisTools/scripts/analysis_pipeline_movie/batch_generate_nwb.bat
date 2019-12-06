call activate bigmess

set PYTHONPATH=%PYTHONPATH%;E:\data\python_packages\corticalmapping;E:\data\python_packages\allensdk_internal;E:\data\python_packages\ainwb\ainwb;E:\data\github_packages\retinotopic_mapping;

python 200_generate_nwb.py
python 210_add_vasmap.py
python 220_add_sync_data.py
python 230_add_image_data.py
python 240_add_motion_correction_module.py
python 250_get_photodiode_onset.py
python 260_add_visual_stimuli_retinotopic_mapping.py
python 270_analyze_photodiode_onsets.py
python 275_add_eyetracking.py
python 280_add_rois_and_traces_caiman_segmentation.py

PAUSE