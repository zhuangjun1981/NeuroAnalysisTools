call activate bigmess

set PYTHONPATH=%PYTHONPATH%;E:\data\python_packages\corticalmapping;E:\data\python_packages\allensdk_internal;E:\data\python_packages\ainwb\ainwb

Echo Launch dir: "%~dp0"
Echo Current dir: "%CD%"

python "%~dp0\360_plot_dgc_tuning_curves.py"
python "%~dp0\330_plot_RF_contours.py"
python "%~dp0\320_plot_zscore_RFs.py"
python "%~dp0\350_plot_dgc_response_mean.py"
python "%~dp0\340_plot_dgc_response_all.py"
python "%~dp0\310_plot_STRFs.py"