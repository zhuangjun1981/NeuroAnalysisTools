import os
import numpy as np
import scipy.ndimage as ni

def load_data(plane_folder):
    traces = np.load(os.path.join(plane_folder, 'F.npy'))
    traces_neu = np.load(os.path.join(plane_folder, 'Fneu.npy'))
    iscell = np.load(os.path.join(plane_folder, 'iscell.npy'))
    ops = np.load(os.path.join(plane_folder, 'ops.npy'), allow_pickle=True).item()
    spks = np.load(os.path.join(plane_folder, 'spks.npy'))
    stat = np.load(os.path.join(plane_folder, 'stat.npy'), allow_pickle=True)

    return traces, traces_neu, iscell, ops, spks, stat

def get_masks_from_dict(roi_dict, ly, lx):

    ypix = roi_dict['ypix'][~roi_dict['overlap']]
    xpix = roi_dict['xpix'][~roi_dict['overlap']]
    weight = roi_dict['lam'][~roi_dict['overlap']]
    mask = np.zeros((ly, lx))
    mask[ypix, xpix] = weight


    if 'neuropil_mask' in roi_dict.keys():
        mask_neuropil = np.zeros((ly, lx))
        ypix_neuropil = roi_dict['neuropil_mask'] // ly
        xpix_neuropil = roi_dict['neuropil_mask'] % ly
        mask_neuropil[ypix_neuropil, xpix_neuropil] = 1
    else:
        mask_binary = np.zeros((ly, lx))
        mask_binary[ypix, xpix] = 1
        mask_neuropil = np.logical_xor(ni.binary_dilation(mask_binary, iterations=1),
                                       ni.binary_dilation(mask_binary, iterations=8))

    return mask, mask_neuropil




if __name__ == '__main__':

    plane_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\temp" \
                   r"\forBryan\test_add_suite2p\m80_vol2_DGM\plane0\suite2p\plane0"

    traces, traces_neu, iscell, ops, spks, stat = load_data(plane_folder)

    mask, mask_neu = get_masks_from_dict(stat[10], ly=ops['Ly'], lx=ops['Lx'])

    print(stat[0].keys())

    import matplotlib.pyplot as plt
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axs[0].imshow(mask)
    axs[1].imshow(mask_neu)
    plt.show()

