import os

base_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180619-M386444-2p\zstack_zoom2\zstack_zoom2"

channels = ['green', 'red']

for ch_n in channels:
    print('remove uncorrected files for channle: {}'.format(ch_n))

    step_fns = [f for f in os.listdir(os.path.join(base_folder, ch_n)) if f.split('_')[-2] == 'step']
    step_fns.sort()
    print('\n'.join(step_fns))

    for step_fn in step_fns:

        print('\n' + step_fn)
        step_folder = os.path.join(base_folder, ch_n, step_fn)

        fns = os.listdir(step_folder)

        if step_fn + '.tif' in fns:
            os.remove(os.path.join(step_folder, step_fn + '.tif'))
        else:
            print('Cannot find uncorrected file. Skip.')
        