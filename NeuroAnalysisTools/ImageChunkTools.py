import numpy as np
import scipy.ndimage as ni
import NeuroAnalysisTools.core.ImageAnalysis as ia
import NeuroAnalysisTools.core.TimingAnalysis as ta


def downsample_planes(img, d_rate):
    """
    downsample each z plane of a image chunk

    :param img: 3d array, ZXY
    :param d_rate: int, downsample rate
    :return imgd: 3d array, downsampled image chunk
    """

    if len(img.shape) != 3:
        raise ValueError('input array should be 3d.')

    dtype = img.dtype
    z, y, x = img.shape

    yd = img.shape[1] // d_rate
    xd = img.shape[2] // d_rate

    img = img[:,
              :yd * d_rate,
              :xd * d_rate]

    # imgd = ia.rigid_transform_cv2(img=img, zoom=1 / d_rate).astype(dtype)

    imgd = np.zeros((z, yd, xd, d_rate, d_rate), dtype=np.float64)

    for yi in range(d_rate):
        for xi in range(d_rate):
            imgd[:, :, :, yi, xi] = img[:, yi::d_rate, xi::d_rate]

    imgd = np.mean(imgd, axis=-1)
    imgd = np.mean(imgd, axis=-1)
    imgd = imgd.astype(dtype)

    return imgd


def threshold(img, std_thr):
    """
    :param img:
    :param std_thr: float, how many std to threshold
    :return imgt:
    """
    v_median = np.median(img.flat)
    v_std = np.std(img.flat)

    thr_low = v_median - std_thr * v_std
    if thr_low < 0:
        thr_low = 0

    thr_high = v_median + std_thr * v_std

    img[img < thr_low] = thr_low
    img[img > thr_high] = thr_high
    return img


def filter_planes(img, vox_size_x, vox_size_y,sigma_size):
    """
    2d gaussian filter for each plane

    :param img: 3d array, ZYX
    :param vox_size_x: float, um
    :param vox_size_y: float, um
    :param sigma_size: float, um
    :return imgf: 3d array, filtered image chunk
    """

    if len(img.shape) != 3:
        raise ValueError('input array should be 3d.')

    imgf = ni.gaussian_filter(img, sigma=(1,
                                          sigma_size / vox_size_y,
                                          sigma_size / vox_size_x))
    return imgf


def find_surface(img, surface_thr):
    """

    :param img: 3d array, ZYX, assume small z = top; large z = bottom
    :param surface_thr: [0, 1], threshold for detecting surface
    :return top: 2d array, same size as each plane in img, z index of top surface
    :return bot: 2d array, same size as each plane in img, z index of bottom surface
    """

    if len(img.shape) != 3:
        raise ValueError('input array should be 3d.')

    z, y, x = img.shape

    top = np.zeros((y, x), dtype=np.int)
    bot = np.ones((y, x), dtype=np.int) * z

    for yi in range(y):
        for xi in range(x):
            curr_t = img[:, yi, xi]

            if curr_t.max() != curr_t.min():
                curr_t = (curr_t - curr_t.min()) / (curr_t.max() - curr_t.min())

                if curr_t[0] < surface_thr:
                    curr_top = ta.up_crossings(curr_t, surface_thr)
                    if len(curr_top) != 0:
                        top[yi, xi] = curr_top[0]

                if curr_t[-1] < surface_thr:
                    curr_bot = ta.down_crossings(curr_t, surface_thr)
                    if len(curr_bot) != 0:
                        bot[yi, xi] = curr_bot[-1]

    return top, bot


def flatten_top(img, top):
    """
    shift the height of each pixel to align the top of the section
    :param img: 3d array
    :param top: 2d array, int, indices of top surface
    :return imgft: 3d img, same size as img,
    """

    if len(img.shape) != 3:
        raise ValueError('input array should be 3d.')

    if top.shape != (img.shape[1], img.shape[2]):
        raise ValueError('the shape of top should be the same size as each plane in img.')

    imgt = np.zeros(img.shape, dtype=img.dtype)

    z, y, x = img.shape

    for yi in range(y):
        for xi in range(x):
            t = top[yi, xi]
            col = img[t:, yi, xi]
            imgt[:len(col), yi, xi] = col

    return imgt


def flatten_bottom(img, bottom):
    """
    shift the height of each pixel to align the bottom of the section
    :param img: 3d array
    :param bottom: 2d array, int, indices of bottom surface
    :return imgb: 3d img, same size as img,
    """

    if len(img.shape) != 3:
        raise ValueError('input array should be 3d.')

    if bottom.shape != (img.shape[1], img.shape[2]):
        raise ValueError('the shape of top should be the same size as each plane in img.')

    imgb = np.zeros(img.shape, dtype=img.dtype)

    z, y, x = img.shape

    for yi in range(y):
        for xi in range(x):
            b = bottom[yi, xi]
            col = img[:b, yi, xi]
            imgb[-len(col):, yi, xi] = col

    return imgb


def flatten_both_sides(img, top, bottom):
    """
    flatten both sides by interpolation
    :param img: 3d array
    :param top: 2d array, int, indices of top surface
    :param bottom: 2d array, int, indices of bottom surface
    :return imgtb: 3d img
    """

    if len(img.shape) != 3:
        raise ValueError('input array should be 3d.')

    if bottom.shape != (img.shape[1], img.shape[2]):
        raise ValueError('the shape of top should be the same size as each plane in img.')

    if top.shape != (img.shape[1], img.shape[2]):
        raise ValueError('the shape of top should be the same size as each plane in img.')

    z, y, x = img.shape

    depths = bottom - top
    import matplotlib.pyplot as plt
    # plt.imshow(depths)
    # plt.show()
    depth = int(np.median(depths.flat))
    # print(depth, z)

    imgtb = np.zeros((depth, y, x), dtype=img.dtype)

    colz_tb = np.arange(depth)

    for yi in range(y):
        for xi in range(x):
            col = img[top[yi, xi]:bottom[yi, xi], yi, xi]
            colz = np.arange(len(col))
            imgtb[:, yi, xi] = np.interp(colz_tb, colz, col)

    return imgtb





