call activate analysis

:: set PYTHONPATH=%PYTHONPATH%;E:\data\github_packages\NeuroAnalysisTools

Echo Launch dir: "%~dp0"
Echo Current dir: "%CD%"

python "%~dp0\120_get_cells_file_soma.py"
python "%~dp0\130_refine_cells_soma.py"
python "%~dp0\140_get_weighted_rois_and_surrounds.py"
python "%~dp0\150_get_raw_center_and_surround_traces.py"
python "%~dp0\160_get_neuropil_subtracted_traces.py"
python "%~dp0\135_generate_marked_avi.py"