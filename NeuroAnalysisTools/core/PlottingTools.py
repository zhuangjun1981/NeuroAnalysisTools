# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:07:20 2014

@author: junz
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import matplotlib.gridspec as gridspec
import colorsys
import matplotlib.colors as col
import scipy.ndimage as ni
import scipy.stats as stats
from . import ImageAnalysis as ia

try:
    import tifffile as tf
except ImportError:
    import skimage.external.tifffile as tf

try:
    import cv2
except ImportError as e:
    print('can not import OpenCV. ' + str(e))


def get_rgb(colorStr):
    """
    get R,G,B int value from a hex color string
    """
    return int(colorStr[1:3], 16), int(colorStr[3:5], 16), int(colorStr[5:7], 16)


def get_color_str(R, G, B):
    """
    get hex color string from R,G,B value (integer with uint8 format)
    """
    if not (isinstance(R, int) and isinstance(G, int) and isinstance(G, int)):
        raise TypeError('Input R, G and B should be integer!')

    if not ((0 <= R <= 255) and (0 <= G <= 255) and (
            0 <= B <= 255)):
        raise ValueError('Input R, G and B should between 0 and 255!')

    # ================== old =========================
    # return '#' + ''.join(map(chr, (R, G, B))).encode('hex')
    # ================================================

    cstrs = [R, G, B]
    cstrs = ['{:02x}'.format(x) for x in cstrs]
    return '#' + ''.join(cstrs)


def binary_2_rgba(img, foregroundColor='#ff0000', backgroundColor='#000000', foregroundAlpha=255, backgroundAlpha=0):
    """
    generate display image in (RGBA).(np.uint8) format which can be displayed by imshow
    :param img: input image, should be a binary array (np.bool, or np.(u)int
    :param foregroundColor: color for 1 in the array, RGB str, i.e. '#ff0000'
    :param backgroundColor: color for 0 in the array, RGB str, i.e. '#ff00ff'
    :param foregroundAlpha: alpha for 1 in the array, int, 0-255
    :param backgroundAlpha: alpha for 1 in the array, int, 0-255
    :return: displayImg, (RGBA).(np.uint8) format, ready for imshow
    """

    if img.dtype == np.bool:
        pass
    elif issubclass(img.dtype.type, np.integer):
        if np.amin(img) < 0 or np.amax(img) > 1: raise ValueError('Values of input image should be either 0 or 1.')
    else:
        raise TypeError('Data type of input image should be either np.bool or integer.')

    if isinstance(foregroundAlpha, int):
        if foregroundAlpha < 0 or foregroundAlpha > 255: raise ValueError('Value of foreGroundAlpha should be between 0 and 255.')
    else:
        raise TypeError('Data type of foreGroundAlpha should be integer.')

    if isinstance(backgroundAlpha, int):
        if backgroundAlpha < 0 or backgroundAlpha > 255: raise ValueError('Value of backGroundAlpha should be between 0 and 255.')
    else:
        raise TypeError('Data type of backGroundAlpha should be integer.')

    fR, fG, fB = get_rgb(foregroundColor)
    bR, bG, bB = get_rgb(backgroundColor)

    displayImg = np.zeros((img.shape[0], img.shape[1], 4)).astype(np.uint8)
    displayImg[img == 1] = np.array([fR, fG, fB, foregroundAlpha]).astype(np.uint8)
    displayImg[img == 0] = np.array([bR, bG, bB, backgroundAlpha]).astype(np.uint8)

    return displayImg


def scalar_2_rgba(img, color='#ff0000'):
    """
    generate display a image in (RGBA).(np.uint8) format which can be displayed by imshow
    alpha is defined by values in the img
    :param img: input image
    :param alphaMatrix: matrix of alpha
    :param foreGroundColor: color for 1 in the array, RGB str, i.e. '#ff0000'
    :return: displayImg, (RGBA).(np.uint8) format, ready for imshow
    """

    R, G, B = get_rgb(color)

    RMatrix = (R * ia.array_nor(img.astype(np.float32))).astype(np.uint8)
    GMatrix = (G * ia.array_nor(img.astype(np.float32))).astype(np.uint8)
    BMatrix = (B * ia.array_nor(img.astype(np.float32))).astype(np.uint8)

    alphaMatrix = (ia.array_nor(img.astype(np.float32)) * 255).astype(np.uint8)

    displayImg = np.zeros((img.shape[0], img.shape[1], 4)).astype(np.uint8)
    displayImg[:, :, 0] = RMatrix;
    displayImg[:, :, 1] = GMatrix;
    displayImg[:, :, 2] = BMatrix;
    displayImg[:, :, 3] = alphaMatrix

    return displayImg


def bar_graph(left,
              height,
              error=None,
              errorDir='both',  # 'both', 'positive' or 'negative'
              width=0.1,
              plotAxis=None,
              lw=3,
              errorColor='#000000',
              faceColor='none',
              edgeColor='#000000',
              capSize=10,
              label=None,
              **kwargs):
    """
    plot a single bar with error bar
    """

    if not plotAxis:
        f = plt.figure()
        plotAxis = f.add_subplot(111)

    if error is not None:
        if errorDir == 'both':
            yerr = error
        elif errorDir == 'positive':
            yerr = [[0], [error]]
        elif errorDir == 'negative':
            yerr = [[error], [0]]
        else:
            raise (ValueError, '"errorDir" should be one of the following: "both", "positive" of "negative".')



        plotAxis.errorbar(left + width / 2,
                          height,
                          yerr=yerr,
                          lw=lw,
                          capsize=capSize,
                          capthick=lw,
                          color=errorColor)

    plotAxis.bar(left,
                 height,
                 width=width,
                 color=faceColor,
                 edgecolor=edgeColor,
                 lw=lw,
                 label=label,
                 align='edge',
                 **kwargs)

    return plotAxis


def random_color(numOfColor=10):
    """
    generate as list of random colors
    """
    numOfColor = int(numOfColor)

    colors = []

    Cmatrix = (np.random.rand(numOfColor, 3) * 255).astype(np.uint8)

    for i in range(numOfColor):

        r = hex(Cmatrix[i][0]).split('x')[1]
        if len(r) == 1:
            r = '0' + r

        g = hex(Cmatrix[i][1]).split('x')[1]
        if len(g) == 1:
            g = '0' + g

        b = hex(Cmatrix[i][2]).split('x')[1]
        if len(b) == 1:
            b = '0' + b

        colors.append('#' + r + g + b)

    return colors


def show_movie(path,  # tif file path or numpy arrary of the movie
               mode='raw',  # 'raw', 'dF' or 'dFoverF'
               baselinePic=None,  # picuture of baseline
               baselineType='mean',  # way to calculate baseline
               cmap='gray'):
    """
    plot tf movie in the way defined by mode
    """

    if isinstance(path, str):
        rawMov = tf.imread(path)
    elif isinstance(path, np.ndarray):
        rawMov = path

    if mode == 'raw':
        mov = rawMov
    else:
        _, dFMov, dFoverFMov = ia.normalize_movie(rawMov,
                                                  baselinePic=baselinePic,
                                                  baselineType=baselineType)
        if mode == 'dF':
            mov = dFMov
        elif mode == 'dFoverF':
            mov = dFoverFMov
        else:
            raise LookupError('The "mode" should be "raw", "dF" or "dFoverF"!')

    if isinstance(path, str):
        tf.imshow(mov,
                  cmap=cmap,
                  vmax=np.amax(mov),
                  vmin=np.amin(mov),
                  title=mode + ' movie of ' + path)
    elif isinstance(path, np.ndarray):
        tf.imshow(mov,
                  cmap=cmap,
                  vmax=np.amax(mov),
                  vmin=np.amin(mov),
                  title=mode + ' Movie')

    return mov


def standalone_color_bar(vmin, vmax, cmap, sectionNum=10):
    """
    plot a stand alone color bar.
    """

    a = np.array([[vmin, vmax]])

    f = plt.figure(figsize=(0.1, 9))
    ax = f.add_subplot(111)
    fig = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_visible(False)
    cbar = f.colorbar(fig)
    cbar.set_ticks(np.linspace(vmin, vmax, num=sectionNum + 1))
    return f


def alpha_blending(image, alphaData, vmin, vmax, cmap='Paired', sectionNum=10, background=-1, interpolation='nearest',
                   isSave=False, savePath=None):
    """
    Generate image with transparency weighted by another matrix.

    Plot numpy array 'image' with colormap 'cmap'. And define the tranparency
    of each pixel by the value in another numpy array alphaData.

    All the elements in alphaData should be non-negative.
    """

    if image.shape != alphaData.shape:
        raise LookupError('"image" and "alphaData" should have same shape!!')

    if np.amin(alphaData) < 0:
        raise ValueError('All the elements in alphaData should be bigger than zero.')

    # normalize image
    image[image > vmax] = vmax
    image[image < vmin] = vmin

    image = (image - vmin) / (vmax - vmin)

    # get colored image of image
    exec ('colorImage = cm.' + cmap + '(image)')

    # normalize alphadata
    alphaDataNor = alphaData / np.amax(alphaData)
    alphaDataNor = np.sqrt(alphaDataNor)

    colorImage[:, :, 3] = alphaDataNor

    # plt.figure()
    # plot dummy figure for colorbar
    a = np.array([[vmin, vmax]])
    plt.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0)
    # plt.gca().set_visible(False)
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(vmin, vmax, num=sectionNum + 1))
    cbar.set_alpha(1)
    cbar.draw_all()

    # generate black background
    b = np.array(colorImage)
    b[:] = background
    b[:, :, 3] = 1
    plt.imshow(b, cmap='gray')

    # plot map
    plt.imshow(colorImage, interpolation=interpolation)

    return colorImage


def plot_mask(mask, plotAxis=None, color='#ff0000', zoom=1, borderWidth=None, closingIteration=None):
    """
    plot mask borders in a given color
    """

    if not plotAxis:
        f = plt.figure()
        plotAxis = f.add_subplot(111)

    cmap1 = col.ListedColormap(color, 'temp')
    cm.register_cmap(cmap=cmap1)

    if zoom != 1:
        mask = ni.interpolation.zoom(mask, zoom, order=0)

    mask2 = mask.astype(np.float32)
    mask2[np.invert(np.isnan(mask2))] = 1.
    mask2[np.isnan(mask2)] = 0.

    struc = ni.generate_binary_structure(2, 2)
    if borderWidth:
        border = mask2 - ni.binary_erosion(mask2, struc, iterations=borderWidth).astype(np.float32)
    else:
        border = mask2 - ni.binary_erosion(mask2, struc).astype(np.float32)

    if closingIteration:
        border = ni.binary_closing(border, iterations=closingIteration).astype(np.float32)

    border[border == 0] = np.nan

    currfig = plotAxis.imshow(border, cmap='temp', interpolation='nearest')

    return currfig


def plot_mask_borders(mask, plotAxis=None, color='#ff0000', zoom=1, borderWidth=2, closingIteration=None,
                      is_filled=False, **kwargs):
    """
    plot mask (ROI) borders by using pyplot.contour function. all the 0s and Nans in the input mask will be considered
    as background, and non-zero, non-nan pixel will be considered in ROI.
    """
    if not plotAxis:
        f = plt.figure()
        plotAxis = f.add_subplot(111)

    plotingMask = np.ones(mask.shape, dtype=np.uint8)

    plotingMask[np.logical_or(np.isnan(mask), mask == 0)] = 0

    if zoom != 1:
        plotingMask = cv2.resize(plotingMask.astype(np.float),
                                 dsize=(int(plotingMask.shape[1] * zoom), int(plotingMask.shape[0] * zoom)))
        plotingMask[plotingMask < 0.5] = 0
        plotingMask[plotingMask >= 0.5] = 1
        plotingMask = plotingMask.astype(np.uint8)

    if closingIteration is not None:
        plotingMask = ni.binary_closing(plotingMask, iterations=closingIteration).astype(np.uint8)

    if is_filled:
        currfig = plotAxis.contourf(plotingMask, levels=[0.5, 1], colors=color, **kwargs)
    else:
        currfig = plotAxis.contour(plotingMask, levels=[0.5], colors=color, linewidths=borderWidth, **kwargs)

    # put y axis in decreasing order
    y_lim = list(plotAxis.get_ylim())
    y_lim.sort()
    plotAxis.set_ylim(y_lim[::-1])

    plotAxis.set_aspect('equal')

    return currfig


def plot_mask2(mask, plotAxis=None, color='#ff0000', zoom=1, closingIteration=None, **kwargs):
    """
    plot mask (ROI) borders by using pyplot.contour function. all the 0s and Nans in the input mask will be considered
    as background, and non-zero, non-nan pixel will be considered in ROI.
    """
    if not plotAxis:
        f = plt.figure()
        plotAxis = f.add_subplot(111)

    plotingMask = np.ones(mask.shape, dtype=np.uint8)

    plotingMask[np.logical_or(np.isnan(mask), mask == 0)] = 0

    if zoom != 1:
        plotingMask = cv2.resize(plotingMask.astype(np.float),
                                 dsize=(int(plotingMask.shape[1] * zoom), int(plotingMask.shape[0] * zoom)))
        plotingMask[plotingMask < 0.5] = 0
        plotingMask[plotingMask >= 0.5] = 1
        plotingMask = plotingMask.astype(np.uint8)

    if closingIteration is not None:
        plotingMask = ni.binary_closing(plotingMask, iterations=closingIteration).astype(np.uint8)

    currfig = plotAxis.contourf(plotingMask, levels=[0.5, 1], colors=color, **kwargs)

    # put y axis in decreasing order
    y_lim = list(plotAxis.get_ylim())
    y_lim.sort()
    plotAxis.set_ylim(y_lim[::-1])

    plotAxis.set_aspect('equal')

    return currfig


def grid_axis(rowNum, columnNum, totalPlotNum, **kwarg):
    """
    return figure handles and axis handels for multiple subplots and figures
    """

    figureNum = totalPlotNum // (rowNum * columnNum) + 1

    figureHandles = []

    for i in range(figureNum):
        f = plt.figure(**kwarg)
        figureHandles.append(f)

    axisHandles = []
    for i in range(totalPlotNum):
        currFig = figureHandles[i // (rowNum * columnNum)]
        currIndex = i % (rowNum * columnNum)
        currAxis = currFig.add_subplot(rowNum, columnNum, currIndex + 1)
        axisHandles.append(currAxis)

    return figureHandles, axisHandles


def grid_axis2(nrows, ncols, fig=None, fig_kw=None, gridspec_kw=None, share_level=2):
    """
    return a grid of axes acorrding to the gridspec specified by input parameters

    :param nrows: positive int, number of rows
    :param ncols: positive int, number of cols
    :param fig: matplotlib.figure object
    :param fig_kw: dict, keyword arguments for matplotlib.figure() method
    :param gridspec_kw: dict, keyword arguments for matplotlib.gridspec.GridSpec class
    :param share_level: 0, the generated axes do not share axis at all;
                        1, the generated axes according to the rows and columns
                        2, the generated axes all share same x and y axis
    :return: a 2d array of matplotlib.axes objects
    """

    if fig_kw is None:
        fig_kw = {'figsize': (10, 10)}

    if fig is None:
        fig = plt.figure(**fig_kw)

    if gridspec_kw is None:
        gridspec_kw = {}

    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, **gridspec_kw)

    if share_level == 0:
        axs = [[fig.add_subplot(gs[i, j]) for j in range(ncols)] for i in range(nrows)]
    elif share_level == 1:

        axs = [[0 for j in range(ncols)] for i in range(nrows)]
        for i in range(nrows)[::-1]:
            for j in range(ncols):
                if i < nrows - 1 and j > 0:
                    axs[i][j] = fig.add_subplot(gs[i, j], sharex=axs[nrows - 1][j], sharey=axs[i][0])
                elif i < nrows - 1 and j == 0:
                    axs[i][j] = fig.add_subplot(gs[i, j], sharex=axs[nrows - 1][j])
                elif i == nrows - 1 and j > 0:
                    axs[i][j] = fig.add_subplot(gs[i, j], sharey=axs[i][0])
                else:
                    axs[i][j] = fig.add_subplot(gs[i, j])

    elif share_level == 2:
        axs = [[0 for j in range(ncols)] for i in range(nrows)]
        for i in range(nrows)[::-1]:
            for j in range(ncols):
                if i < nrows - 1 or j > 0:
                    axs[i][j] = fig.add_subplot(gs[i, j], sharex=axs[nrows - 1][0], sharey=axs[nrows - 1][0])
                else:
                    axs[i][j] = fig.add_subplot(gs[i, j])
    else:
        raise ValueError('do not understand "share_level", should be 0 or 1 or 2.')

    axs = np.array([np.array(a) for a in axs])

    return axs


def tile_axis(f, rowNum, columnNum, topDownMargin=0.05, leftRightMargin=0.01, rowSpacing=0.01, columnSpacing=0.01):
    if 2 * topDownMargin + (
        (rowNum - 1) * rowSpacing) >= 1: raise ValueError('Top down margin or row spacing are too big!')
    if 2 * leftRightMargin + (
        (columnNum - 1) * columnSpacing) >= 1: raise ValueError('Left right margin or column spacing are too big!')

    height = (1 - (2 * topDownMargin) - (rowNum - 1) * rowSpacing) / rowNum
    width = (1 - (2 * leftRightMargin) - (columnNum - 1) * columnSpacing) / columnNum

    xStarts = np.arange(leftRightMargin, 1 - leftRightMargin, (width + columnSpacing))
    yStarts = np.arange(topDownMargin, 1 - topDownMargin, (height + rowSpacing))[::-1]

    axisList = [[f.add_axes([xStart, yStart, width, height]) for xStart in xStarts] for yStart in yStarts]

    return axisList


def save_figure_without_borders(f,
                                savePath,
                                removeSuperTitle=True,
                                is_axis_off=True,
                                **kwargs):
    """
    remove borders of a figure
    """
    f.gca().get_xaxis().set_visible(False)
    f.gca().get_yaxis().set_visible(False)
    if is_axis_off:
        f.gca().set_axis_off()
    f.gca().set_title('')
    if removeSuperTitle:
        f.suptitle('')
    f.tight_layout(pad=0., h_pad=0., w_pad=0., rect=(0, 0, 1, 1))
    # f.savefig(savePath, frameon=False, **kwargs)
    # f.savefig(savePath, pad_inches=0, bbox_inches='tight', frameon=False, **kwargs)
    f.savefig(savePath, pad_inches=0, bbox_inches='tight', **kwargs)


def merge_normalized_images(imgList, isFilter=True, sigma=50, mergeMethod='mean', dtype=np.float32):
    """
    merge images in a list in to one, for each image, local intensity variability will be removed by subtraction of
    gaussian filtered image. Then all images will be collapsed by the mergeMethod in to single image
    """

    imgList2 = []

    for currImg in imgList:
        imgList2.append(ia.array_nor(currImg.astype(dtype)))

    if mergeMethod == 'mean':
        mergedImg = np.mean(np.array(imgList2), axis=0)
    elif mergeMethod == 'min':
        mergedImg = np.min(np.array(imgList2), axis=0)
    elif mergeMethod == 'max':
        mergedImg = np.max(np.array(imgList2), axis=0)
    elif mergeMethod == 'median':
        mergedImg = np.median(np.array(imgList2), axis=0)

    if isFilter:
        mergedImgf = ni.filters.gaussian_filter(mergedImg.astype(np.float), sigma=sigma)
        return ia.array_nor(mergedImg - mergedImgf).astype(dtype)
    else:
        return ia.array_nor(mergedImg).astype(dtype)


def hue_2_rgb(hue):
    """
    get the RGB value as format as hex string from the decimal ratio of hue (from 0 to 1)
    color model as described in:
    https://en.wikipedia.org/wiki/Hue
    """
    if hue < 0: hue = 0
    if hue > 1: hue = 1
    color = colorsys.hsv_to_rgb(hue, 1, 1)
    color = [int(x * 255) for x in color]
    return get_color_str(*color)


def hot_2_rgb(hot):
    """
    get the RGB value as format as hex string from the decimal ratio of hot colormap (from 0 to 1)
    """
    if hot < 0: hot = 0
    if hot > 1: hot = 1
    cmap_hot = plt.get_cmap('hot')
    color = cmap_hot(hot)[0:3];
    color = [int(x * 255) for x in color]
    return get_color_str(*color)


def cmap_2_rgb(value, cmap_string):
    """
    get the RGB value as format as hex string from the value of a given color map

    :param value: float, input value for a given color, this value should be within (0., 1.). value out of range will
                  be clipped to the limit
    :param cmap_string: str, string for a colormap
    :return: color hex string
    """

    cmap = plt.get_cmap(cmap_string)
    color = cmap(value)[0:3]
    color = [int(x * 255) for x in color]
    return get_color_str(*color)


def value_2_rgb(value, cmap):
    """
    get the RGB value as format as hex string from the decimal ratio of a given colormap (from 0 to 1)
    """
    if value < 0: value = 0
    if value > 1: value = 1
    cmap = plt.get_cmap(cmap)
    color = cmap(value)[0:3]
    color = [int(x * 255) for x in color]
    return get_color_str(*color)


def plot_event_ticks(event_tss, t_range, plot_axis=None, color='#000000', **kwargs):
    """
    plot event timestamps for multiple trials with each event as a small vertical tick. Input of this function can be
    generated by core.TimingAnalysis.event_triggered_event_trains() function

    :param event_tss: list of lists, each sublist contains timestamps (float) of event for a given trial
    :param t_range: tuple of two floats, relative start and end time of a trial
    :param plot_axis: axes object for plotting
    :param **kwargs: inputs to plot_axis.plot() function
    :return:
    """

    step_width = 1. / len(event_tss)
    centers = np.arange(step_width / 2., 1., step_width)[::-1]
    lowers = centers - (step_width * 0.9 / 2.)
    uppers = centers + (step_width * 0.9 / 2.)

    if len(centers) != len(event_tss):
        raise ValueError('number of line centers does not equal number of trials!')

    if plot_axis is None:
        f = plt.figure(figsize=(10, 6))
        plot_axis = f.add_subplot(111)
    plot_axis.set_xlim(t_range)
    plot_axis.set_ylim([0, 1])
    plot_axis.set_yticks([])
    plot_axis.set_xlabel('time (sec)')
    for i, event_ts in enumerate(event_tss):
        curr_lower = lowers[i]
        curr_upper = uppers[i]
        for event in event_ts:
            plot_axis.plot([event, event], [curr_lower, curr_upper], '-', color=color, **kwargs)


def plot_spike_waveforms(unit_ts, channels, channel_ts, fig=None, t_range=(-0.002, 0.002), channel_names=None,
                         **kwargs):
    """
    plot spike waveforms across multiple continuous channels given unit spike timestamps

    :param unit_ts: 1d array, timestamps of the plotting unit (second)
    :param channels: list of 1d arrays, value of each continuous channel, all arrays should have same length
    :param channel_ts: 1d array, timestamps of the continuous channels
    :param fig: matplotlib.figure object, if None, a fig object will be created
    :param t_range: tuple of two floats, time range to plot along spike time stamps
    :param channel_names: list of strings, name for each channel, should have same length as channels
    :param kwargs: inputs to matplotlib.axes.plot() function
    :return: fig
    """

    # print('in plotting tools.')

    if fig is None:
        fig = plt.figure(figsize=(8, 6))

    ch_num = len(channels)
    t_step = np.mean(np.diff(channel_ts))

    ind_range = [int(t_range[0] / t_step), int(t_range[1] / t_step)]
    # print('ind_range:', ind_range)

    if t_range[0] < 0:
        base_point_num = -int(t_range[0] / t_step)
    else:
        base_point_num = ind_range[1] - ind_range[0]

    # print('getting spike indices ...')
    unit_inds = np.round((unit_ts - channel_ts[0]) / t_step).astype(np.int64)
    unit_inds = np.array([ind for ind in unit_inds if (ind + ind_range[0]) >= 0 and
                          (ind + ind_range[1]) < len(channel_ts)])

    # axis direction: (channel, spike, time)
    traces = np.zeros((ch_num, len(unit_inds), ind_range[1] - ind_range[0]), dtype=np.float32)

    # print('traces shape:', traces.shape)

    # print('filling traces ...')
    for i, ch in enumerate(channels):
        # print('current channel:', i)
        for j, unit_ind in enumerate(unit_inds):
            curr_trace = ch[unit_ind + ind_range[0]: unit_ind + ind_range[1]]
            traces[i, j, :] = curr_trace - np.mean(curr_trace[0:base_point_num])

    traces_min = np.amin(traces)
    traces_max = np.amax(traces)
    mean_traces = np.mean(traces, axis=1)

    # print(traces_min)
    # print(traces_max)

    t_axis = t_range[0] + np.arange(traces.shape[2], dtype=np.float32) * t_step
    for k in range(traces.shape[0]):
        curr_ax = fig.add_subplot(1, ch_num, k + 1)
        curr_ax.set_xlim(t_range)
        curr_ax.set_ylim([traces_min, traces_max])
        if k == 0:
            curr_ax.set_yticks([traces_min, 0., traces_max])
        else:
            curr_ax.set_yticks([])
        curr_ax.locator_params(nbins=3, axis='x')
        curr_ax.set_xlabel('time (sec)')
        if channel_names is not None:
            curr_ax.set_title(channel_names[k])
        # for l in range(traces.shape[1]):
        #     curr_ax.plot(t_axis, traces[k, l, :], **kwargs)
        curr_ax.plot(t_axis, mean_traces[k, :], '-k', lw=2)

    return fig


def distributed_axes(f, axes_pos, axes_region=(0., 0., 1., 1.), margin=(0.2, 0.2), axes_size=(0.1, 0.1)):
    """
    generate a spatially distributed axes in the given figure f, the axes locations are defined by axes_pos, this
    function should capture the relative positions among axes but scale free. Designed for plot spike waveforms from
    different channels

    :param f: matplotlib figure object
    :param axes_pos: list of tuples, each element is (x_pos, y_pos) of axes center location
    :param axes_region: tuple of 4 floats, (left, bottom, width, height) to define the subregion to place these axes
    :param margin: tuple of 2 floats, (x_margin, y_margin) on both ends within and relative to axex_region
    :param axes_size: tuple, sise of subplot, (width, height), relative to axes_region
    :return: list of axes at specified locations, same order as in axes_pos
    """

    x_pos = np.array([p[0] for p in axes_pos], dtype=np.float32)
    y_pos = np.array([p[1] for p in axes_pos], dtype=np.float32)

    x_size_new = axes_size[0] * axes_region[2]
    x_margin_new = margin[0] * axes_region[2]
    x_center_start = axes_region[0] + x_margin_new + x_size_new / 2.
    x_center_range = axes_region[2] - 2 * (x_margin_new) - x_size_new

    if np.amax(x_pos) - np.amin(x_pos) == 0:
        x_pos_new = np.array([x_center_start + x_center_range / 2 for ind in x_pos])
    else:
        x_pos_new = ia.array_nor(x_pos) * x_center_range + x_center_start

    y_size_new = axes_size[1] * axes_region[3]
    y_margin_new = margin[1] * axes_region[3]
    y_center_start = axes_region[1] + y_margin_new + y_size_new / 2.
    y_center_range = axes_region[3] - 2 * (y_margin_new) - y_size_new

    if np.amax(y_pos) - np.amin(y_pos) == 0:
        y_pos_new = np.array([y_center_start + y_center_range / 2 for ind in y_pos])
    else:
        y_pos_new = ia.array_nor(y_pos) * y_center_range + y_center_start

    ax_list = []
    for i in range(len(x_pos_new)):
        curr_ax = f.add_axes([x_pos_new[i] - x_size_new / 2,
                              y_pos_new[i] - y_size_new / 2,
                              x_size_new,
                              y_size_new])
        ax_list.append(curr_ax)

    return ax_list


def plot_multiple_traces(traces, x=None, plot_axis=None, mean_kw=None, is_plot_shade=True, shade_type='std',
                         shade_kw=None, is_plot_sample=True, sample_kw=None):
    """
    plot mean and variance (std or sem) of multiple traces on the same time axis, it can plot individual trace
    as well, designed for plotting event triggered average of multiple trials.

    :param traces: 2d array, trial x time_sample
    :param x: 1d array, len should equal to traces.shape[1], if None, it will be np.arange(traces.shape[1])
    :param plot_axis: matplotlib axes object, plotting axes
    :param mean_kw: keyword arguments for plotting mean trace, follow matplotlib.axes.plot function
    :param is_plot_shade: bool, plot vairance as shaded area or not
    :param shade_type: keyword arguments for plotting shaded variance, follow matplotlib.axes.
    :param shade_kw:
    :param is_plot_sample:
    :param sample_kw:
    :return:
    """

    traces = np.array(traces, dtype=np.float64)

    if x is None:
        x = np.arange(traces.shape[1])

    if mean_kw is None:
        mean_kw = {'color':'#000088', 'lw':2}

    if shade_kw is None:
        shade_kw = {'facecolor':'#000088', 'lw':0, 'alpha':0.3}

    if sample_kw is None:
        sample_kw = {'color':'#888888', 'lw':0.5, 'alpha':0.5}

    if plot_axis is None:
        f = plt.figure(figsize=(8, 4))
        plot_axis = f.add_subplot(111)

    if is_plot_sample:
        for trace in traces:
            plot_axis.plot(x, trace, **sample_kw)

    trace_mean = np.mean(traces, axis=0)

    if is_plot_shade:
        if shade_type == 'std':
            trace_var = np.std(traces, axis=0)
        elif shade_type == 'sem':
            trace_var = np.std(traces, axis=0) / np.sqrt(float(traces.shape[0]))
        else:
            raise ValueError('Do not understand shade type. Should be "std" or "sem".')

        plot_axis.fill_between(x, trace_mean - trace_var, trace_mean + trace_var, **shade_kw)

    plot_axis.plot(x, trace_mean, **mean_kw)

    return plot_axis


def plot_dire_distribution(dires, weights=None, is_arc=False, bins=12, plot_ax=None,
                           plot_type='bar', is_density=False, denominator=None,
                           **kwargs):
    """
    plot the distribution of a list of directions in a nice way.

    :param dires: array of float. directions to be plotted.
    :param weights: array with same size as dires, weights of data
    :param is_arc: bool. If True, dires are in [0, 2*pi] scale, if False, dires are in [0, 360] scale
    :param bins: int, how many bins are there
    :param plot_ax: matplotlib.axes._subplots.PolarAxesSubplot object
    :param plot_type: str, 'bar' or 'line'
    :param kwargs: if plot_type == 'bar', key word argument to the plot_ax.bar() function;
                   if plot_type == 'line', kew word argument to the plot_ax.plot() function;
    :return:
    """

    if plot_ax is None:
        f = plt.figure(figsize=(5,5))
        plot_ax = f.add_subplot(111, projection='polar')

    if not isinstance(plot_ax, matplotlib.projections.polar.PolarAxes):
        raise TypeError('input "plot_ax" should be a "matplotlib.projections.polar.PolarAxes" or '
                        'a "matplotlib.axes._subplots.PolarAxesSubplot" object')

    plot_dires = np.array(dires, dtype=np.float64)

    if is_arc is False:
        plot_dires = plot_dires * np.pi / 180.

    plot_dires = plot_dires % (2 * np.pi)

    bin_width = np.pi * 2 / bins

    for dire_i, dire in enumerate(plot_dires):
        if dire > ((np.pi * 2) - (bin_width / 2)):
            plot_dires[dire_i] = dire - (np.pi * 2)

    # print(plot_dires)
    counts, bin_lst = np.histogram(plot_dires,
                                   weights=weights,
                                   bins=bins,
                                   density=is_density,
                                   range=[-bin_width / 2., (np.pi * 2) - (bin_width / 2)])

    if denominator is not None:
        counts = counts / denominator

    bin_lst = bin_lst[0:-1] + (bin_width / 2)

    if plot_type == 'bar':
        plot_ax.bar(bin_lst, counts, width=bin_width, align='center', **kwargs)
    elif plot_type == 'line':
        counts = list(counts)
        counts.append(counts[0])
        bin_lst = list(bin_lst)
        bin_lst.append(bin_lst[0])
        plot_ax.plot(bin_lst, counts, **kwargs)
    else:
        raise LookupError('Do not understand parameter "plot_type", should be "bar" or "line".')

    plot_ax.set_xticklabels([])

    return plot_ax, counts[:-1], bin_lst[:-1]


def plot_orie_distribution(ories, weights=None, is_arc=False, bins=12,  plot_ax=None, plot_type='bar',
                           plot_color='#888888', is_density=False, denominator=None,
                           **kwargs):
    """
    plot the distribution of a list of directions in a nice way.

    :param ories: array of float. orientations to be plotted.
    :param weights: array with same size as dires, weights of data
    :param is_arc: bool. If True, dires are in [0, 2*pi] scale, if False, dires are in [0, 360] scale
    :param bins: int, how many bins are there
    :param plot_ax: matplotlib.axes._subplots.PolarAxesSubplot object
    :param plot_type: str, 'bar' or 'line'
    :param kwargs: if plot_type == 'bar', key word argument to the plot_ax.bar() function;
                   if plot_type == 'line', kew word argument to the plot_ax.plot() function;
    :return:
    """

    if plot_ax is None:
        f = plt.figure(figsize=(5,5))
        plot_ax = f.add_subplot(111, projection='polar')

    if not isinstance(plot_ax, matplotlib.projections.polar.PolarAxes):
        raise TypeError('input "plot_ax" should be a "matplotlib.projections.polar.PolarAxes" or '
                        'a "matplotlib.axes._subplots.PolarAxesSubplot" object')

    plot_ories = np.array(ories, dtype=np.float64)

    if is_arc is False:
        plot_ories = plot_ories * np.pi / 180.

    plot_ories = plot_ories % np.pi

    bin_width = np.pi / bins

    for orie_i, orie in enumerate(plot_ories):
        if orie > (np.pi - (bin_width / 2)):
            plot_ories[orie_i] = orie - (np.pi * 2)

    # print(plot_dires)
    counts, bin_lst = np.histogram(plot_ories,
                                   weights=weights,
                                   bins=bins,
                                   density=is_density,
                                   range=[-bin_width / 2.,np.pi - (bin_width / 2)])

    if denominator is not None:
        counts = counts / denominator

    bin_lst = bin_lst[0:-1] + (bin_width / 2)

    if plot_type == 'bar':
        plot_ax.bar(bin_lst, counts, width=bin_width * 0.5, align='center', color=plot_color,
                    edgecolor=plot_color, **kwargs)
        plot_ax.bar(bin_lst + np.pi, counts, width=bin_width * 0.5, align='center', color='#ffffff',
                    edgecolor=plot_color, **kwargs)
    elif plot_type == 'line':
        counts = list(counts)
        counts.append(counts[0])
        bin_lst = list(bin_lst)
        bin_lst.append(bin_lst[-1] + bin_width)
        bin_lst = np.array(bin_lst)
        plot_ax.plot(bin_lst, counts, ls='-', color=plot_color, **kwargs)
        plot_ax.plot(bin_lst + np.pi, counts, ls='--', color=plot_color, **kwargs)
    else:
        raise LookupError('Do not understand parameter "plot_type", should be "bar" or "line".')

    plot_ax.set_xticks(np.concatenate((bin_lst, bin_lst + np.pi), axis=0))
    plot_ax.set_xticklabels([])

    return plot_ax, counts[:-1], bin_lst[:-1]

def plot_distribution(data, bin_centers, plot_ax=None, plot_type='line', is_density=True, is_cumulative=True,
                      is_plot_mean=True, color=None, label=None, **kwargs):
    """

    :param data: 1d array
    :param bin_centers: 1d array
    :param plot_ax: matplotlib.axes object
    :param plot_type: str, 'line' or 'step'
    :param is_density: bool
    :param is_cumulative: bool
    :param is_plot_mean: bool
    :param kwargs: more parameters to matplotlib plot/step functions
    :return: plot_ax
    """

    bin_width = np.mean(np.diff(bin_centers))
    hist_range = [bin_centers[0] - (bin_width / 2.), bin_centers[-1] + (bin_width / 2.)]
    hist, bine = np.histogram(data, bins=len(bin_centers), range=hist_range)

    if color is None:
        color = random_color(1)[0]

    if is_density:
        hist = hist / len(data)

    if is_cumulative:
        hist = np.cumsum(hist)

    if plot_ax is None:
        f = plt.figure(figsize=(5, 5))
        plot_ax = f.add_subplot(111)

    if is_plot_mean:
        plot_ax.axvline(x=np.mean(data), color=color, **kwargs)

    if plot_type == 'line':
        plot_ax.plot(bin_centers, hist, color=color, label=label, **kwargs)

    if plot_type == 'step':
        hist = np.concatenate(([hist[0]], hist))
        plot_ax.step(bine, hist, where='pre', color=color, label=label, **kwargs)

    return plot_ax


def density_scatter_plot(x, y, ax=None, is_log_x=False, is_log_y=False,
                         xmin=1e-10, ymin=1e-10, **kwargs):
    """
    :param x: 1d array, same size as y
    :param y: 1d array, same size as x
    :param ax: plt.axes object
    :param is_log_x: bool, if True, x will be plotted on log scale
    :param is_log_y: bool, if True, y will be plotted on log scale
    :param xmin: float, >0, minimum value used for x, this will be applied
                 only if is_log_x == True, to avoid log(0) error
    :param ymin: float, >0, minimum value used for y, this will be applied
                 only if is_log_y == True, to avoid log(0) error
    :param kwargs: input to plt.scatter function
    :return: ax
    """

    if len(x.shape) != 1 or len(y.shape) != 1:
        raise ValueError("input x and y should be 1d array.")

    if x.shape != y.shape:
        raise ValueError("input x and y should have same shape.")

    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)

    if is_log_x:
        y = y[x >= xmin]
        x = x[x >= xmin]
        x = np.log10(x)

    if is_log_y:
        x = x[y >= ymin]
        y = y[y >= ymin]
        y = np.log10(y)

    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    ax.scatter(x, y, c=z, **kwargs)

    return ax





if __name__ == '__main__':
    plt.ioff()

    # ----------------------------------------------------
    # dires = [0,0,0,90,90,90,90,90,90,180,180]
    # plot_dire_distribution(dires=dires, is_arc=False)
    # plt.show()
    # ----------------------------------------------------

    # ----------------------------------------------------
    # bg = np.random.rand(100,100)
    # maskBin=np.zeros((100,100),dtype=np.uint8)
    # maskBin[20:30,50:60]=1
    # maskNan=np.zeros((100,100),dtype=np.float32)
    # maskNan[20:30,50:60]=1
    # f=plt.figure(); ax=f.add_subplot(111)
    # ax.imshow(bg,cmap='gray')
    # _ = plot_mask_borders(maskNan, plotAxis=ax, color='#0000ff', zoom=1, closingIteration=20)
    # plt.show()
    # ----------------------------------------------------

    # ----------------------------------------------------
    # bg = np.random.rand(100,100)
    # maskBin=np.zeros((100,100),dtype=np.uint8)
    # maskBin[20:30,50:60]=1
    # maskNan=np.zeros((100,100),dtype=np.float32)
    # maskNan[20:30,50:60]=1
    # f=plt.figure(); ax=f.add_subplot(111)
    # ax.imshow(bg,cmap='gray', interpolation='nearest')
    # _ = plot_mask2(maskBin, plotAxis=ax, color='#0000ff', zoom=1, closingIteration=None)
    # plt.show()
    # ----------------------------------------------------

    # ----------------------------------------------------
    # ax = bar_graph(0.5,1,error=0.1,label='xx')
    # ax.legend()
    # plt.show()
    # ----------------------------------------------------

    # ----------------------------------------------------
    # figures, axises = grid_axis(2,3,20)
    # for i, ax in enumerate(axises):
    #     ax.imshow(np.random.rand(5,5))
    # plt.show()
    # ----------------------------------------------------


    # ----------------------------------------------------
    # mask = np.zeros((100,100))
    # mask[30:50,20:60]=1
    # mask[mask==0]=np.nan
    #
    # plot_mask(mask)
    # plt.show()
    # ----------------------------------------------------

    # ----------------------------------------------------
    # aa=np.random.rand(20,20)
    # mask = np.zeros((20,20),dtype=np.bool)
    # mask[4:7,13:16]=True
    # displayMask = binary_2_rgba(mask)
    # plt.figure()
    # plt.imshow(aa)
    # plt.imshow(displayMask,interpolation='nearest')
    # plt.show()
    # ----------------------------------------------------

    # ----------------------------------------------------
    # b=np.random.rand(5,5)
    # displayImg = scalar_2_rgba(b)
    # plt.imshow(displayImg,interpolation='nearest')
    # plt.show()
    # ----------------------------------------------------

    # ----------------------------------------------------
    # print hue2RGB((2./3.))
    # assert hue2RGB((2./3.)) == '#0000ff'
    # ----------------------------------------------------

    # ----------------------------------------------------
    # f=plt.figure()
    # f.suptitle('test')
    # ax=f.add_subplot(111)
    # ax.imshow(np.random.rand(20,20))
    # save_figure_without_borders(f,r'C:\JunZhuang\labwork\data\python_temp_folder\test_title.png',removeSuperTitle=False,dpi=300)
    # save_figure_without_borders(f,r'C:\JunZhuang\labwork\data\python_temp_folder\test_notitle.png',removeSuperTitle=True,dpi=300)
    # ----------------------------------------------------

    # ----------------------------------------------------
    # f=plt.figure(figsize=(12,9))
    # axisList = tile_axis(f,4,3,0.05,0.05,0.05,0.05)
    # print np.array(axisList).shape
    # plt.show()
    # ----------------------------------------------------

    # ----------------------------------------------------
    # assert(hot_2_rgb(0.5) == value_2_rgb(0.5,'hot'))
    # ----------------------------------------------------

    # ----------------------------------------------------
    # etts = [[-0.5, -0.4, 0., 1., 3.], [-0.2, -0.2, 0., 0.5, 1.3, 2., 2.1, 2.5]]
    # t_range = (-1, 4)
    # plot_event_ticks(event_tss=etts, t_range=t_range, lw=2)
    # plt.show()
    # ----------------------------------------------------

    # ----------------------------------------------------
    # f = plt.figure(figsize=(8.5, 11))
    # ax_pos = [(0., -1.5), (0., 0.), (-0.866, -3.), (0.866, -3.)]
    # axs = distributed_axes(f, ax_pos, axes_region=[0.05, 0.75, 0.25, 0.2], margin=(0., 0.,), axes_size=(0.25, 0.25))
    # plt.show()
    # ----------------------------------------------------

    # ----------------------------------------------------
    # bg = np.random.rand(100,100)
    # maskBin=np.zeros((100,100),dtype=np.uint8)
    # maskBin[20:30,50:60]=1
    # maskNan=np.zeros((100,100),dtype=np.float32)
    # maskNan[20:30,50:60]=1
    # f=plt.figure(); ax=f.add_subplot(111)
    # ax.imshow(bg,cmap='gray', interpolation='nearest')
    # _ = plot_mask_borders(maskBin, plotAxis=ax, color='#0000ff', zoom=1, is_filled=True, closingIteration=None)
    # plt.show()
    # ----------------------------------------------------

    # ----------------------------------------------------
    # traces = np.random.rand(50, 20)
    # plot_multiple_traces(traces)
    # plt.show()
    # ----------------------------------------------------

    # ----------------------------------------------------
    # grid_axis2(nrows=4, ncols=3, share_level=1)
    # plt.show()
    # ----------------------------------------------------

    # ----------------------------------------------------
    data = [1,2,2,3,3,4,4,4,5,6,7,7,8,8,9,9,9,9,9,10]
    bin_centers = np.arange(1, 11)
    print(bin_centers)
    plot_distribution(data, bin_centers, plot_ax=None, plot_type='step',
                      is_density=False, is_cumulative=False,
                      is_plot_mean=True)
    plt.show()

    # ----------------------------------------------------

    print('for debug')
