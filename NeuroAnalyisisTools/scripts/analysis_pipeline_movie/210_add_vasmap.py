import os
import corticalmapping.NwbTools as nt
import corticalmapping.core.ImageAnalysis as ia
import matplotlib.pyplot as plt
import tifffile as tf


import os
import corticalmapping.NwbTools as nt
import corticalmapping.core.ImageAnalysis as ia
import matplotlib.pyplot as plt
import tifffile as tf

is_plot = False
vasmap_dict = {
                'vasmap_wf': 'wide field surface vasculature map through cranial window original',
                'vasmap_wf_rotated': 'wide field surface vasculature map through cranial window rotated',
                'vasmap_2p_green': '2p surface vasculature map through cranial window green original, zoom1',
                'vasmap_2p_green_rotated': '2p surface vasculature map through cranial window green rotated, zoom1',
                'vasmap_2p_red': '2p surface vasculature map through cranial window red original, zoom1',
                'vasmap_2p_red_rotated': '2p surface vasculature map through cranial window red rotated, zoom1'
                }

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

nwb_fn = [f for f in os.listdir(curr_folder) if f[-4:] == '.nwb'][0]
nwb_f = nt.RecordedFile(nwb_fn)

for mn, des in vasmap_dict.items():
    try:
        curr_m = ia.array_nor(tf.imread(mn + '.tif'))

        if is_plot:
            f = plt.figure(figsize=(10, 10))
            ax = f.add_subplot(111)
            ax.imshow(curr_m, vmin=0., vmax=1., cmap='gray', interpolation='nearest')
            ax.set_axis_off()
            ax.set_title(mn)
            plt.show()

        print('adding {} to nwb file.'.format(mn))
        nwb_f.add_acquisition_image(mn, curr_m, description=des)

    except Exception as e:
        print(e)

nwb_f.close()