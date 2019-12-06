import os

base_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data\181112-M424454-2p\zstack"
channels = ['green', 'red']
is_remove_img = False

for ch in channels:
    print('processing channel: {} ...'.format(ch))
    ch_folder = os.path.join(base_folder, ch)

    step_fns = [f for f in os.listdir(ch_folder) if f.split('_')[-2] == 'step']
    step_fns.sort()
    print('\n'.join(step_fns))

    for step_fn in step_fns:

        print('\n' + step_fn)
        step_folder = os.path.join(ch_folder, step_fn)
        fns = os.listdir(step_folder)


        if is_remove_img:
            if 'corrected_max_projection.tif' in fns:
                print('removing corrected_max_projection.tif')
                os.remove(os.path.join(step_folder, 'corrected_max_projection.tif'))

            if 'corrected_max_projections.tif' in fns:
                print('removing corrected_max_projections.tif')
                os.remove(os.path.join(step_folder, 'corrected_max_projections.tif'))

            if 'corrected_mean_projection.tif' in fns:
                print('removing corrected_mean_projection.tif')
                os.remove(os.path.join(step_folder, 'corrected_mean_projection.tif'))

            if 'corrected_mean_projections.tif' in fns:
                print('removing corrected_mean_projections.tif')
                os.remove(os.path.join(step_folder, 'corrected_mean_projections.tif'))

        if 'correction_offsets.hdf5' in fns:
            print('removing correction_offsets.hdf5')
            os.remove(os.path.join(step_folder, 'correction_offsets.hdf5'))

        fn_cor = [f for f in fns if f[-14:] == '_corrected.tif']
        if len(fn_cor) == 1:
            print('removing ' + fn_cor[0])
            os.remove(os.path.join(step_folder, fn_cor[0]))