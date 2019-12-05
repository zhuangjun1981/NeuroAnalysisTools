import os
import numpy as np
import corticalmapping.core.FileTools as ft
import corticalmapping.core.ImageAnalysis as ia
import tifffile as tf

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project\180816-M376019-zstack"

vas_map_fns= ["180816JCamF100",]

saveFolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(saveFolder)

vas_map_paths = [os.path.join(data_folder, f) for f in vas_map_fns]

vas_maps = []

for vas_map_path in vas_map_paths:
    
    # if not work, try: tailerLength=218
    vas_map_focused, _, _ = ft.importRawJCamF(vas_map_path, column=1024, row=1024, headerLength = 116,
                                              tailerLength=452) 
    vas_map_focused = vas_map_focused[2:]
    vas_map_focused = vas_map_focused[:, ::-1, :]
    vas_map_focused[vas_map_focused > 50000] = 400
    vas_map_focused = np.mean(vas_map_focused, axis=0)
    vas_maps.append(ia.array_nor(vas_map_focused))

vas_map = ia.array_nor(np.mean(vas_maps, axis=0))

tf.imsave('vas_map_focused_wf_green.tif', vas_map.astype(np.float32))