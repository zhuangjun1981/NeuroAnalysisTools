

__author__ = 'junz'

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import scipy.ndimage as ni
import scipy.stats as stats
import skimage.morphology as sm
import skimage.measure as measure
import time
from . import FileTools as ft
from . import PlottingTools as pt

try:
    import cv2
except ImportError as e:
    print(e)

# try:
#     from toolbox.misc import BinarySlicer
# except ImportError as e:
#     print(e)


def resample(t1,y1,interval,kind='linear', isPlot = False):

    '''
    :param t1: time stamps of original data
    :param y1: value of original data (relative to t1)
    :param interval: the intervals of resampled time stamps, second
    :param kind: interpolation type, same as 'scipy.interpolate.interp1d'
    :return: t2, y2
    '''

    f = interpolate.interp1d(t1,y1,kind=kind)

    t2 = np.arange(t1[0],t1[-1], interval)

    y2 = f(t2)

    if isPlot:
        plt.figure()
        plt.plot(t1,y1)
        plt.plot(t2,y2)
        plt.legend('original','resampled')

    return t2, y2


def resample2(t1, y1, t2, kind = 'linear', isPlot=False, bounds_error=False):

    '''
    :param t1: time stamps of original data
    :param y1: value of original data (relative to t1)
    :param t2: time stamps for resample
    :param kind: interpolation type, same as 'scipy.interpolate.interp1d'
    :return: y2
    '''

    f = interpolate.interp1d(t1,y1,kind=kind, bounds_error=bounds_error)

    y2 = f(t2)

    if isPlot:
        plt.figure()
        plt.plot(t1,y1)
        plt.plot(t2,y2)
        plt.legend('original','resampled')

    return y2


def array_nor(A):
    '''
    normalize a np.array to the scale [0, 1]
    '''

    B = A.astype(np.float)

    if np.isnan(A).any():
        maxv = np.nanmax(B.flat)
        minv = np.nanmin(B.flat)
    else:
        maxv = np.max(B.flat)
        minv = np.min(B.flat)

    return (B - minv) / (maxv - minv)


def array_nor_median(A):
    '''
    normalize array by minus median, data type will be switch to np.float
    '''
    B=A.astype(np.float)
    return B-np.median(B.flatten())


def array_nor_mean(A):
    '''
    normalize array by minus mean, data type will be switch to np.float
    '''
    B=A.astype(np.float)
    return B-np.mean(B.flatten())


def array_nor_mean_std(A):
    '''
    normalize array by minus mean and then devided by standard deviation, data type will be switch to np.float
    '''
    A=A.astype(np.float)
    B=(A-np.mean(A.flatten()))/np.std(A.flatten())
    return B


def zscore(A):
    '''
    return Z score of an array.
    '''

    if np.isnan(A).any(): return (A-np.nanmean(A.flatten()))/np.nanstd(A.flatten())
    else: A = A.astype(np.float); return (A-np.mean(A.flatten()))/np.std(A.flatten())


def distance(p0, p1):
    '''
    calculate distance between two points, can be multi-dimensinal

    p0 and p1 should be a 1d array, with each element for each dimension
    '''


    #old code======================================================================
    # if (len(p0.shape) > 1) or (len(p1.shape) > 1):
    #     raise(LookupError('Both input arrays should be 1d array!!'))
    #
    # if p0.shape != p1.shape:
    #     raise LookupError('The two input arrays should have same dimensions.')
    #
    # distance = math.sqrt(np.sum(((p0.astype(np.float)-p1.astype(np.float))**2)))
    #===============================================================================

    if not isinstance(p0, np.ndarray):p0 = np.array(p0)
    if not isinstance(p1, np.ndarray):p1 = np.array(p1)
    return np.sqrt(np.mean(np.square(p0-p1).flatten()))


def array_diff(a0, a1):
    '''
    calculate the sum of pixel-wise difference between two arrays
    '''
    if not isinstance(a0, np.ndarray):a0 = np.array(a0)
    if not isinstance(a1, np.ndarray):a1 = np.array(a1)
    return np.mean(np.abs(a0-a1).flatten())


def binarize(array, threshold):
    '''
    binarize array to 0s and 1s, by cutting at threshold
    '''

    newArray = np.array(array)

    newArray[array>=threshold] = 1.

    newArray[array<threshold] = 0.

    newArray = newArray.astype(array.dtype)

    return newArray


def center_image(img,  # original image, 2d ndarray
                 centerPixel,  # the coordinates of center pixel in original image, [col, row]
                 newSize = 512,  #the size of output image
                 fill_value = 0
                 ):
    '''
    center a certain image in a new canvas

    the pixel defined by 'centerPixel' in the original image will be at the center of the output image
    the size of output image is defined by 'newSize'

    empty pixels will be filled with zeros
    '''

    x = newSize/2 - centerPixel[1]
    y = newSize/2 - centerPixel[0]

    M = np.float32([[1,0,x],[0,1,y]])

    newImg = cv2.warpAffine(img, M, (newSize,newSize), borderValue=fill_value)

    return newImg


def resize_image(img, outputShape, fillValue=0.):
    '''
    resize every frame of a 3-d matrix to defined output shape
    if the original image is too big it will be truncated
    if the original image is too small, value defined as fillValue will filled in. default: 0
    '''

    width = outputShape[1]
    height = outputShape[0]

    if width < 1:
        raise ValueError('width should be bigger than 0!!')

    if height < 1:
        raise ValueError('height should be bigger than 0!!')

    if len(img.shape) !=2 and len(img.shape) !=3 :
        raise ValueError('input image should be a 2-d or 3-d array!!')

    if len(img.shape) == 2: # 2-d image
        startWidth = img.shape[-1]
        startHeight = img.shape[-2]
        newImg = np.array(img)
        if startWidth > width:
            newImg = newImg[:,0:width]
        elif startWidth < width:
            attachRight = np.zeros((startHeight,width-startWidth))
            attachRight[:] = fillValue
            attachRight.astype(img.dtype)
            newImg = np.hstack((newImg,attachRight))

        if startHeight > height:
            newImg = newImg[0:height,:]
        elif startHeight < height:
            attachBottom = np.zeros((height - startHeight,width))
            attachBottom[:] = fillValue
            attachBottom.astype(img.dtype)
            newImg = np.vstack((newImg,attachBottom))

    if len(img.shape) == 3: # 3-d matrix
        startDepth = img.shape[0]
        startWidth = img.shape[-1]
        startHeight = img.shape[-2]
        newImg = np.array(img)
        if startWidth > width:
            newImg = newImg[:,:,0:width]
        elif startWidth < width:
            attachRight = np.zeros((startDepth,startHeight,width-startWidth))
            attachRight[:] = fillValue
            attachRight.astype(img.dtype)
            newImg = np.concatenate((img,attachRight),axis=2)

        if startHeight > height:
            newImg = newImg[:,0:height,:]
        elif startHeight < height:
            attachBottom = np.zeros((startDepth,height-startHeight,width))
            attachBottom[:] = fillValue
            attachBottom.astype(img.dtype)
            newImg = np.concatenate((newImg,attachBottom),axis=1)

    return newImg


def expand_image_cv2(img):

    if len(img.shape) != 2:
        raise ValueError('Input image should be 2d!')

    dtype = img.dtype
    img = img.astype(np.float32)
    rows,cols = img.shape
    diagonal = int(np.sqrt(rows**2+cols**2))
    M = np.float32([[1,0,(diagonal-cols)/2],[0,1,(diagonal-rows)/2]])
    newImg = cv2.warpAffine(img,M,(diagonal,diagonal))
    return newImg.astype(dtype)


def expand_image(img):

    if len(img.shape) == 2:
        rows,cols = img.shape
        diagonal = int(np.sqrt(rows**2+cols**2))
        top = np.zeros(((diagonal-rows)/2,cols),dtype=img.dtype)
        down = np.zeros((diagonal-img.shape[0]-top.shape[0],cols),dtype=img.dtype)
        tall = np.vstack((top,img,down))
        left = np.zeros((tall.shape[0],(diagonal-cols)/2),dtype=img.dtype)
        right = np.zeros((tall.shape[0],diagonal-img.shape[1]-left.shape[1]),dtype=img.dtype)
        newImg = np.hstack((left,tall,right))
        return newImg
    elif len(img.shape) == 3:
        frames,rows,cols = img.shape
        diagonal = int(np.sqrt(rows**2+cols**2))
        top = np.zeros((frames,(diagonal-rows)/2,cols),dtype=img.dtype)
        down = np.zeros((frames,diagonal-img.shape[1]-top.shape[1],cols),dtype=img.dtype)
        tall = np.concatenate((top,img,down),axis=1)
        left = np.zeros((frames,tall.shape[1],(diagonal-cols)/2),dtype=img.dtype)
        right = np.zeros((frames,tall.shape[1],diagonal-img.shape[2]-left.shape[2]),dtype=img.dtype)
        newImg = np.concatenate((left,tall,right),axis=2)
        return newImg
    else:
        raise ValueError('Input image should be 2d or 3d!')


def zoom_image(img, zoom, interpolation ='cubic'): #'cubic','linear','area','nearest','lanczos4'
    '''
    zoom a 2d image. if zoom is a single value, it will apply to both axes, if zoom has two values it will be applied to
    height and width respectively
    zoom[0]: height
    zoom[1]: width
    '''
    if len(img.shape) != 2:
        raise ValueError('Input image should be 2d!')

    try: zoomH = float(zoom[0]); zoomW = float(zoom[1])
    except TypeError: zoomH = float(zoom); zoomW = float(zoom)

    if interpolation == 'cubic': interpo = cv2.INTER_CUBIC
    if interpolation == 'linear': interpo = cv2.INTER_LINEAR
    if interpolation == 'area': interpo = cv2.INTER_AREA
    if interpolation == 'nearest': interpo = cv2.INTER_NEAREST
    if interpolation == 'lanczos4': interpo = cv2.INTER_LANCZOS4

    newImg= cv2.resize(img.astype(np.float),dsize=(int(img.shape[1]*zoomW),int(img.shape[0]*zoomH)),interpolation=interpo)
    return newImg


def moveImage(img, Xoffset, Yoffset, width, height, fill_value=0.0):
    '''
    move image defined by Xoffset and Yoffset

    new canvas size is defined by width and height

    empty pixels will be filled with zeros
    '''
    if len(img.shape) != 2:
        raise ValueError('Input image should be 2d!')

    M = np.float32([[1,0,Xoffset],[0,1,Yoffset]])

    newImg = cv2.warpAffine(img, M, (width,height), borderValue=fill_value)

    return newImg


def rotate_image(img, angle, borderValue=0.0):
    '''
    rotate an image conterclock wise by an angle defined by 'angle' in degree

    pixels go out side will be tropped
    pixels with no value will be filled as zeros
    '''

    if len(img.shape) != 2:
        raise ValueError('Input image should be 2d!')

    rows,cols = img.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    # M = cv2.getRotationMatrix2D((0,0),angle,1)
    newImg = cv2.warpAffine(img,M,(cols,rows),borderValue=borderValue)

    return newImg


def rigid_transform(img, zoom=None, rotation=None, offset=None, outputShape=None, mode='constant', fill_value=0.0):

    '''
    rigid transformation of a 2d-image or 3d-matrix by using scipy
    :param img: input image/matrix
    :param zoom:
    :param rotation: in degree, counterclock wise
    :param offset: tuple (xoffset, yoffset) pixel value of starting point of output image
    :param outputShape: the shape of output image, (height, width)
    :return: new image or matrix after transformation
    '''

    if len(img.shape) != 2 and len(img.shape) != 3:
        raise LookupError('Input image is not a 2d or 3d array!')

    newImg = img.astype(np.float32)

    if zoom is not None:
        if len(img.shape) == 2:
            newZoom = (zoom,zoom)
        elif len(img.shape) == 3:
            newZoom = (1,zoom,zoom)
        newImg = ni.zoom(newImg, zoom=newZoom, mode=mode, cval=fill_value)

    if rotation is not None:
        newImg = expand_image(newImg)
        if len(img.shape) == 2:
            newImg = ni.rotate(newImg, angle=rotation, reshape=False, mode=mode, cval=fill_value)
        elif len(img.shape) == 3:
            newImg = ni.rotate(newImg, angle=rotation, axes=(1,2), reshape=False, mode=mode, cval=fill_value)

    if offset is not None:
        if len(img.shape) == 2:
            newImg = ni.shift(newImg, (offset[1],offset[0]), mode=mode, cval=fill_value)
        if len(img.shape) == 3:
            newImg = ni.shift(newImg, (0,offset[1],offset[0]), mode=mode, cval=fill_value)

    if outputShape:
        newImg = resize_image(newImg, outputShape)

    return newImg.astype(img.dtype)


def rigid_transform_cv2_2d(img, zoom=None, rotation=None, offset=None, outputShape=None, fill_value=0.0):

    '''
    rigid transformation of a 2d-image by using opencv
    :param img: input image/matrix
    :param zoom:
    :param rotation: in degree, counterclock wise
    :param offset: tuple (xoffset, yoffset) pixel value of starting point of output image
    :param outputShape: the shape of output image, (height, width)
    :return: new image or matrix after transformation
    '''

    if len(img.shape) != 2:
        raise LookupError('Input image is not a 2d or 3d array!')

    newImg = np.array(img).astype(np.float)
    # minValue = np.amin(newImg)

    if zoom is not None:
        newImg = zoom_image(img, zoom=zoom)

    if rotation:
        newImg = expand_image_cv2(newImg)
        newImg = rotate_image(newImg, rotation, borderValue=fill_value)

    if (outputShape is None) and (offset is None):
        return newImg
    else:
        if outputShape is None:
            outputShape = newImg.shape
        if offset is None:
            offset = (0,0)
        newImg = moveImage(newImg, offset[0], offset[1], outputShape[1], outputShape[0], fill_value=fill_value)

        return newImg.astype(img.dtype)


def rigid_transform_cv2_3d(img, zoom=None, rotation=None, offset=None, outputShape=None, fill_value=0.0):

    if len(img.shape) != 3:
        raise LookupError('Input image is not a 3d array!')

    # if outputShape is None:
    #
    #     if zoom is not None:
    #         try:
    #             newHeight = int(img.shape[1] * zoom[0])
    #             newWidth = int(img.shape[2] * zoom[1])
    #         except TypeError:
    #             newHeight = int(img.shape[1] * zoom)
    #             newWidth = int(img.shape[2] * zoom)
    #     else:
    #         newHeight = img.shape[1]
    #         newWidth = img.shape[2]
    # else:
    #     newHeight = outputShape[0]
    #     newWidth = outputShape[1]

    frame_1 = rigid_transform_cv2_2d(img[0, :, :], zoom=zoom, rotation=rotation, offset=offset,
                                     outputShape=outputShape, fill_value=fill_value)
    newHeight = frame_1.shape[0]
    newWidth = frame_1.shape[1]

    newImg = np.empty((img.shape[0],newHeight,newWidth),dtype=img.dtype)

    for i in range(img.shape[0]):
        newImg[i,:,:] = rigid_transform_cv2_2d(img[i, :, :], zoom=zoom, rotation=rotation, offset=offset,
                                               outputShape=outputShape, fill_value=fill_value)

    return newImg


def rigid_transform_cv2(img, zoom=None, rotation=None, offset=None, outputShape=None, fill_value=0.0):

    '''
    rigid transformation of a 2d-image or 3d-matrix by using opencv
    :param img: input image/matrix
    :param zoom:
    :param rotation: in degree, counterclock wise
    :param offset: tuple (xoffset, yoffset) pixel value of starting point of output image
    :param outputShape: the shape of output image, (height, width)
    :return: new image or matrix after transformation
    '''

    if len(img.shape) == 2:
        return rigid_transform_cv2_2d(img, zoom=zoom, rotation=rotation, offset=offset, outputShape=outputShape,
                                      fill_value=fill_value)
    elif len(img.shape) == 3:
        return rigid_transform_cv2_3d(img, zoom=zoom, rotation=rotation, offset=offset, outputShape=outputShape,
                                      fill_value=fill_value)
    else:
        raise ValueError('Input image is not a 2d or 3d array!')


def boxcartime_dff(data,
                   window,# boxcar size in seconds
                   fs # sample rate in ms
                   ):
    """
    Created on Mon Nov 24 14:37:02 2014

    [dff] = boxcartime_dff(data[t,y,x], rollingwindow[in s], samplerate[in ms])
    boxcar average uses scipy.signal package, imported locally

    @author: mattv
    """
    import scipy.signal as sig

    if data.ndim != 3:
        raise LookupError('input images must be a 3-dim array format [t,y,x]')

    exposure = np.float(fs/1000) #convert exposure from ms to s
    win = np.float(window)
    win = win/exposure
    win = np.ceil(win)

    # rolling average
    kernal = np.ones(win, dtype=('float'))
    padsize = data.shape[0] + win*2
    mov_ave = np.zeros([padsize+win-1], dtype=('float'))
    mov_pad = np.zeros([padsize], dtype=('float'))
    mov_dff = np.zeros([data.shape[0]-win, data.shape[1], data.shape[2]])
    for y in range(data.shape[1]):
        for x in range(data.shape[2]):
            # put data within padded array
            mov_pad[win:(padsize-win)] = data[:,y,x]
            # moving average by convolution
            mov_ave = sig.fftconvolve(mov_pad, kernal)/win
            # cut off pad
            mov_ave = mov_ave[win*2:1+mov_ave.shape[0]-win*2]
            # use moving average as f0 for df/f
            mov_dff[:,y,x] = (data[(win/2):data.shape[0]-(win/2),y,x] - mov_ave)/mov_ave

    return mov_dff


def normalize_movie(movie,
                    baselinePic = None,  # picture for baseline
                    baselineType = 'mean'  # 'mean' or 'median'
                    ):
    '''
    return average image, movie minus avearage, and dF over F for each pixel
    '''
    movie = np.array(movie, dtype = np.float32)

    if baselinePic is not None:

      if movie.shape[1:] != baselinePic.shape:
          raise LookupError('The shape of "baselinePic" should match the shape of the frame shape of "movie"!')

      averageImage = baselinePic

    elif baselineType == 'mean':
        averageImage = np.mean(movie, axis = 0)

    elif baselineType == 'median':
        averageImage = np.median(movie, axis = 0)

    else:
        raise LookupError('The "baselineType" should be "mean" or "median"!!')

    normalizedMovie = np.subtract(movie,averageImage)
    dFoverFMovie = np.divide(normalizedMovie,averageImage)

    return averageImage, normalizedMovie, dFoverFMovie


def temporal_filter_movie(mov,  # array of movie
                        Fs,  # sampling rate
                        Flow,  # low cutoff frequency
                        Fhigh,  # high cutoff frequency
                        mode = 'box'): # filter mode, '1/f' or 'box'):

    if len(mov.shape) != 3:
        raise LookupError('The "mov" array should have 3 dimensions!')

    frameNum = mov.shape[0]
    freqs = np.fft.fftfreq(frameNum, d = (1./float(Fs)))

    filterArray = np.ones(frameNum)

    for i in range(frameNum):
        if ((freqs[i] > 0) and (freqs[i] < Flow) or (freqs[i] > Fhigh)) or \
           ((freqs[i] < 0) and (freqs[i] > -Flow) or (freqs[i] < -Fhigh)):
            filterArray[i] = 0

    if mode == '1/f':
        filterArray[1:] = filterArray[1:] / abs(freqs[1:])
        filterArray[0] = 0
        filterArray = (filterArray - np.amin(filterArray)) / (np.amax(filterArray) - np.amin(filterArray))
    elif mode == 'box':
        filterArray[0] = 0
    else: raise NameError('Variable "mode" should be either "1/f" or "box"!')

    if Flow == 0:
        filterArray[0] = 1

    movFFT = np.fft.fft(mov, axis = 0)

    for i in range(mov.shape[1]):
        for j in range(mov.shape[2]):
            movFFT[:,i,j] = movFFT[:,i,j] * filterArray

    movF = np.real(np.fft.ifft(movFFT, axis = 0))

    return movF


def generate_rectangle_mask(shape, center, width, height, isplot = False):

    if len(shape) !=2: raise LookupError('Shape should be two dimensional.')

    mask = np.zeros(shape); mask[:] = np.nan
    mask[int(round(center[0]-height/2)):int(round(center[0]+height/2)),int(round(center[1]-width/2)):int(round(center[1]+width/2))] = 1

    if np.isnan(np.nansum(mask[:])):
        raise ArithmeticError('No element in mask!')

    if isplot == True:
        f = plt.figure(); ax = f.add_subplot(111)
        pt.plot_mask_borders(mask, plotAxis=ax)

    return mask


def generate_oval_mask(shape, center, width, height, isplot = False):

    if len(shape) !=2: raise LookupError('Shape should be two dimensional.')

    mask = np.zeros(shape); mask[:] = np.nan

    width = float(width); height = float(height)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if ((i-center[0])/(height/2))**2 + ((j-center[1])/(width/2))**2 <= 1:
                mask[i,j]=1

    if np.isnan(np.nansum(mask[:])):
        raise ArithmeticError('No element in mask!')

    if isplot == True:
        f = plt.figure(); ax = f.add_subplot(111)
        pt.plot_mask_borders(mask, plotAxis=ax)

    return mask


def get_trace(movie, mask, maskMode ='binary'):
    '''
    get a trace across a movie with averaged value in a mask

    maskMode: 'binary': ones in roi, zeros outside
              'binaryNan': ones in roi, nans outside
              'weighted': weighted values in roi, zeros outside (note: all pixels equal to zero will be considered outside roi
              'weightedNan': weighted values in roi, nans outside
    '''

    if maskMode == 'binary':
        if np.where(mask==0)[0].size + np.where(mask==1)[0].size < mask.size:
            raise ValueError('Binary mask should only contain zeros and ones!!')
        else:
            finalMask = np.array(mask.astype(np.float))
            pixelNum = np.sum(finalMask.flatten())
    elif maskMode == 'binaryNan':
        if np.sum(np.isnan(mask).flatten())+np.where(mask==1)[0].size < mask.size:
            raise ValueError('BinaryNan mask should only contain nans and ones!!')
        else:
            finalMask = np.ones(mask.shape,dtype=np.float)
            finalMask[np.isnan(mask)] = 0
            pixelNum = mask.size - np.sum(np.isnan(mask).flatten())
    elif maskMode == 'weighted':
        if np.isnan(mask).any(): raise ValueError('Weighted mask should not contain nan(s)!!')
        else:
            finalMask = np.array(mask.astype(np.float))
            pixelNum = mask.size - np.where(mask==0)[0].size
    elif maskMode == 'weightedNan':
        finalMask = np.array(mask.astype(np.float))
        finalMask[np.isnan(mask)] = 0
        pixelNum = mask.size - np.where(finalMask==0)[0].size
    else:
        raise LookupError('maskMode not understood. Should be one of "binary", "binaryNan", "weighted", "weightedNan".')

    trace = np.sum(np.multiply(movie,finalMask),(1,2))/pixelNum

    return trace


def get_trace_binaryslicer(bl_obj, mask, mask_mode = 'binary'):
    '''

    :param bl_obj: the binary slicer object of a large matrix
    :param mask: the mask
    :param mask_mode: same as 'mask_mode' in function get_trace

    maskMode: 'binary': ones in roi, zeros outside
              'binaryNan': ones in roi, nans outside
              'weighted': weighted values in roi, zeros outside (note: all pixels equal to zero will be considered outside roi
              'weightedNan': weighted values in roi, nans outside

    :return: extracted trace
    '''

    if len(bl_obj.shape) != 3: raise ValueError('BinarySlicer object should be 3d!')
    if len(mask.shape) != 2: raise ValueError('Mask should be 2d!')
    if bl_obj.shape[1] != mask.shape[0] or bl_obj.shape[2] != mask.shape[1]:
        raise ValueError('the size of each frame of the BinarySlicer object should be the same as the size of mask')

    if mask_mode == 'binary':
        if np.where(mask==0)[0].size + np.where(mask==1)[0].size < mask.size:
            raise ValueError('Binary mask should only contain zeros and ones!!')
        else:
            mask_ind = np.where(mask != 0)
            # print(mask_ind)
            min_row = min(mask_ind[0]); max_row = max(mask_ind[0]) + 1
            min_col = min(mask_ind[1]); max_col = max(mask_ind[1]) + 1
            finalMask = np.array(mask.astype(np.float))[min_row:max_row, min_col:max_col]
    elif mask_mode == 'binaryNan':
        if np.sum(np.isnan(mask).flatten())+np.where(mask==1)[0].size < mask.size:
            raise ValueError('BinaryNan mask should only contain nans and ones!!')
        else:
            mask_ind = np.where(mask != np.nan)
            min_row = min(mask_ind[0]); max_row = max(mask_ind[0]) + 1
            min_col = min(mask_ind[1]); max_col = max(mask_ind[1]) + 1
            finalMask = np.ones(mask.shape,dtype=np.float)
            finalMask[np.isnan(mask)] = 0
            finalMask = finalMask[min_row:max_row, min_col:max_col]
    elif mask_mode == 'weighted':
        if np.isnan(mask).any(): raise ValueError('Weighted mask should not contain nan(s)!!')
        else:
            mask_ind = np.where(mask != 0)
            min_row = min(mask_ind[0]); max_row = max(mask_ind[0]) + 1
            min_col = min(mask_ind[1]); max_col = max(mask_ind[1]) + 1
            finalMask = np.array(mask.astype(np.float))[min_row:max_row, min_col:max_col]
    elif mask_mode == 'weightedNan':
        finalMask = np.array(mask.astype(np.float))
        finalMask[np.isnan(mask)] = 0
        mask_ind = np.where(finalMask != 0)
        min_row = min(mask_ind[0]); max_row = max(mask_ind[0]) + 1
        min_col = min(mask_ind[1]); max_col = max(mask_ind[1]) + 1
        finalMask = finalMask[min_row:max_row, min_col:max_col]

    mov = bl_obj[:,min_row:max_row, min_col:max_col]
    # print(mov)
    return get_trace(mov, finalMask, maskMode='weighted')


def get_trace_binaryslicer2(bl_obj, mask, mask_mode = 'binary', loading_frame_num = 1000):
    '''

    get trace for a given mask from a BinarySlicer object, by loading chunk each time

    :param bl_obj: the binary slicer object of a large matrix
    :param mask: the mask
    :param mask_mode: same as 'mask_mode' in function get_trace
    :param loading_frame_num: frame number of each chunk

    maskMode: 'binary': ones in roi, zeros outside
              'binaryNan': ones in roi, nans outside
              'weighted': weighted values in roi, zeros outside (note: all pixels equal to zero will be considered outside roi
              'weightedNan': weighted values in roi, nans outside

    :return: extracted trace
    '''

    if loading_frame_num <= 1: raise ValueError('loading_frame_num should be a integer larger than 1!')
    if len(bl_obj.shape) != 3: raise ValueError('BinarySlicer object should be 3d!')
    if len(mask.shape) != 2: raise ValueError('Mask should be 2d!')
    if bl_obj.shape[1] != mask.shape[0] or bl_obj.shape[2] != mask.shape[1]:
        raise ValueError('the size of each frame of the BinarySlicer object should be the same as the size of mask')

    frameNum = bl_obj.shape[0]

    print('\nInput movie shape:', bl_obj.shape)

    chunkNum = frameNum // loading_frame_num
    if frameNum % loading_frame_num == 0:
        print('Translating in chunks: '+ str(chunkNum)+' x '+str(loading_frame_num)+' frame(s)')
    else:
        chunkNum += 1
        print('Translating in chunks: '+str(chunkNum-1)+' x '+str(loading_frame_num)+' frame(s)'+' + '+str(frameNum % loading_frame_num)+' frame(s)')

    traces = []
    for i in range(chunkNum):
        indStart = i*loading_frame_num
        indEnd = (i+1)*loading_frame_num
        if indEnd > frameNum: indEnd = frameNum
        print('Extracting signal from frame '+str(indStart)+' to frame '+str(indEnd)+'.\t'+str(i*100./chunkNum)+'%')
        currMov = bl_obj[indStart:indEnd,:,:]
        traces.append(get_trace(currMov, mask, maskMode=mask_mode))

    return np.concatenate(traces)


def get_trace_binaryslicer3(bl_obj, masks, mask_mode = 'binary', loading_frame_num = 1000):
    '''

    get trace for a given mask from a BinarySlicer object, by loading chunk each time

    :param bl_obj: the binary slicer object of a large matrix
    :param masks: a dictionary of masks
    :param mask_mode: same as 'mask_mode' in function get_trace
    :param loading_frame_num: frame number of each chunk

    maskMode: 'binary': ones in roi, zeros outside
              'binaryNan': ones in roi, nans outside
              'weighted': weighted values in roi, zeros outside (note: all pixels equal to zero will be considered outside roi
              'weightedNan': weighted values in roi, nans outside

    :return: extracted trace
    '''

    if loading_frame_num <= 1: raise ValueError('loading_frame_num should be a integer larger than 1!')
    if len(bl_obj.shape) != 3: raise ValueError('BinarySlicer object should be 3d!')


    frameNum = bl_obj.shape[0]

    print('\nInput movie shape:', bl_obj.shape)

    chunkNum = frameNum // loading_frame_num
    if frameNum % loading_frame_num == 0:
        print('Translating in chunks: '+ str(chunkNum)+' x '+str(loading_frame_num)+' frame(s)')
    else:
        chunkNum += 1
        print('Translating in chunks: '+str(chunkNum-1)+' x '+str(loading_frame_num)+' frame(s)'+' + '+str(frameNum % loading_frame_num)+' frame(s)')

    traces = {}
    for key in masks.keys(): traces.update({'trace_'+key:[]})

    for i in range(chunkNum):
        indStart = i*loading_frame_num
        indEnd = (i+1)*loading_frame_num
        if indEnd > frameNum: indEnd = frameNum
        print('Extracting signal from frame '+str(indStart)+' to frame '+str(indEnd)+'.\t'+str(i*100./chunkNum)+'%')
        currMov = bl_obj[indStart:indEnd,:,:]
        for key, mask in masks.items():
            if len(mask.shape) != 2: raise ValueError('Mask "' + key + '" should be 2d!')
            if bl_obj.shape[1] != mask.shape[0] or bl_obj.shape[2] != mask.shape[1]:
                raise ValueError('the size of each frame of the BinarySlicer object should be the same as the size of mask "' + key + '"!')
            traces['trace_'+key].append(get_trace(currMov, mask, maskMode=mask_mode))

    for key in traces.keys():
        traces[key] = np.concatenate(traces[key])

    return traces


def hit_or_miss(coor, mask):
    '''
    check if a cooridnate (coor) is in a mask, input mask can be int or float, nan and zero will considered as outside, any
    non-nan, non-zero pixel will be considered as inside. Mask does not need to be continuous.
    '''
    mask[np.isnan(mask)] = 0; mask[mask>0] = 1
    mask = mask.astype(np.int8)
    corMask = np.zeros(mask.shape, dtype = np.int8)
    corMask[np.round(coor[0]),np.round(coor[1])] = 1
    if np.sum(np.multiply(corMask, mask)) > 0:
        return True
    if np.sum(np.multiply(corMask, mask)) == 0:
        return False


def harmonic_amplitude(f,  # function value
           period,  # how many fundamental harmonic periods inside the function
           n): # return the n-th harmonic
    '''
    calculate the amplitude and phase of the n-th harmonic components of a
    function. the input function should have whole number times of period of
    the fundamental harmonic.
    '''

    if (type(period) != int) | (period <= 0):
        raise ArithmeticError('"period" should be a positive integer!')

    if (type(n) != int) | (n < 0):
        raise ArithmeticError('"n" should be a non-negative positive integer!')

    L = len(f)
    x = np.arange(L)

    if n == 0:
        har = np.sum(np.multiply(f, np.exp(-1j*2*np.pi*period*n*x/L)))/L
        har_meg = np.abs(har)
        har_phase = np.nan

    else:
        har = 2 * np.sum(np.multiply(f, np.exp(-1j*2*np.pi*period*n*x/L)))/L
        har_meg = np.abs(har)
        har_phase = np.angle(har)

    return har_meg, har_phase


def discretize(array, binSize):
    '''
    discretize the array by binSize
    '''

    bins = np.arange(np.floor(np.nanmin(array)) - (0.5 * binSize),
                     np.ceil(np.nanmax(array)) + (1.5 * binSize),
                     binSize)

    flatArray = np.ndarray.flatten(array)

    indArray = np.digitize(flatArray, bins)

    newArray = np.zeros(flatArray.shape)
    newArray[:] = np.nan

    for i in range(len(indArray)):
        if np.isnan(flatArray[i]) == False:
            newArray[i] = bins[indArray[i]]

    newArray = np.array(newArray).reshape(array.shape)

    return newArray


def seed_pixel(markers):
    '''
    marker centroid of every marked local minimum
    '''


    newMarkers = np.zeros(markers.shape).astype(np.int32)
    intMarkers = markers.astype(np.int32)

    for i in range(1,np.amax(intMarkers)+1):
        aa = np.zeros(markers.shape).astype(np.int32)
        aa[intMarkers == i] = 1
        aapixels = np.argwhere(aa)
        center = np.mean(aapixels.astype(np.float32), axis = 0)

        newMarkers[int(np.round(center[0])),int(np.round(center[1]))] = i

    return newMarkers


def is_adjacent(array1, array2, borderWidth = 2):
    '''
    decide if two patches are adjacent within border width
    '''

    p1d = ni.binary_dilation(array1, iterations = borderWidth-1).astype(np.int8)
    p2d = ni.binary_dilation(array2, iterations = borderWidth-1).astype(np.int8)

    if np.amax(p1d + p2d) > 1:
        return True
    else:
        return False


def remove_small_patches(mask, areaThr=100, structure=([1, 1, 1], [1, 1, 1], [1, 1, 1])):
    '''
    remove small isolated patches
    '''

    if mask.dtype == np.bool:pass
    elif issubclass(mask.dtype.type, np.integer):
        if np.amin(mask)<0 or np.amax(mask)>1:raise ValueError('Values of input image should be either 0 or 1.')
    else: raise TypeError('Data type of input image should be either np.bool or integer.')

    patches, n = ni.label(mask,structure)
    newMask = np.zeros(mask.shape,dtype=np.uint8)

    if n==0: return newMask
    else:
        for i in range(1,n+1):
            currPatch = np.zeros(mask.shape,dtype=np.uint8)
            currPatch[patches==i]=1
            if np.sum(currPatch.flatten())>=areaThr:newMask += currPatch

    return newMask.astype(np.bool)


def get_area_edges(img,
                   firstGaussianSigma=50.,
                   medianFilterWidth=100.,
                   areaThr=(0.1,0.9),
                   edgeThrRange=(5,16),
                   secondGaussianSigma=10.,
                   thr=0.2,
                   borderWidth=2,
                   lengthThr=20,
                   isPlot=True):
    '''
    get binary edge of areas
    '''

    img=img.astype(np.float)
    imgFlat = img - ni.filters.gaussian_filter(img,firstGaussianSigma)
    imgMedianFiltered = array_nor(ni.filters.median_filter(imgFlat, medianFilterWidth))
    imgPatch=np.array(imgMedianFiltered)
    imgPatch[imgMedianFiltered<areaThr[0]]=areaThr[0];imgPatch[imgMedianFiltered>areaThr[1]]=areaThr[1]
    imgPatch=(array_nor(imgPatch) * 255).astype(np.uint8)


    # plt.imshow(imgPatch,vmin=0,vmax=255,cmap='gray')
    # plt.show()

#    import tifffile as tf
#    tf.imsave('Rorb_example_vasMap_filtered.tif',array_nor(imgPatch.astype(np.float32)))


    cuttingStep = np.arange(edgeThrRange[0],edgeThrRange[1])
    cuttingStep = np.array([cuttingStep[0:-1],cuttingStep[1:]]).transpose()
    edge_cv2 = np.zeros(img.shape).astype(np.uint8)
    for i in range(cuttingStep.shape[0]):
        currEdge = cv2.Canny(imgPatch,cuttingStep[i,0],cuttingStep[i,1])/255
        edge_cv2 += currEdge
        if isPlot:
            if i==1: firstEdgeSet=currEdge
            if i==cuttingStep.shape[0]-1: lastEdgeSet=currEdge

    edgesF=ni.filters.gaussian_filter(edge_cv2.astype(np.float),secondGaussianSigma)

    edgesThr = np.zeros(edgesF.shape).astype(np.uint8)
    edgesThr[edgesF<thr]=0;edgesThr[edgesF>=thr]=1

    edgesThin = sm.skeletonize(edgesThr)

    edgesThin = remove_small_patches(edgesThin, lengthThr)
    if borderWidth>1: edgesThick=ni.binary_dilation(edgesThin,iterations=borderWidth-1)
    else: edgesThick=edgesThin

    if isPlot:
        displayEdges = np.zeros((edgesThick.shape[0],edgesThick.shape[1],4)).astype(np.uint8)
        displayEdges[edgesThick==1]=np.array([255,0,0,255]).astype(np.uint8)
        displayEdges[edgesThick==0]=np.array([0,0,0,0]).astype(np.uint8)

        f,ax=plt.subplots(2,5,figsize=(15,5))
        ax[0,0].imshow(img,cmap='gray');ax[0,0].set_title('original image');ax[0,0].axis('off')
        ax[0,1].imshow(imgFlat,cmap='gray');ax[0,1].set_title('flattened image');ax[0,1].axis('off')
        ax[0,2].imshow(imgPatch,cmap='gray');ax[0,2].set_title('image for edge detection');ax[0,2].axis('off')
        ax[0,3].imshow(firstEdgeSet,cmap='gray');ax[0,3].set_title('first edge set');ax[0,3].axis('off')
        ax[0,4].imshow(lastEdgeSet,cmap='gray');ax[0,4].set_title('last edge set');ax[0,4].axis('off')
        ax[1,0].imshow(edgesF,cmap='hot');ax[1,0].set_title('filtered edges sum');ax[1,0].axis('off')
        ax[1,1].imshow(edgesThr,cmap='gray');ax[1,1].set_title('binary thresholded edges');ax[1,1].axis('off')
        ax[1,2].imshow(edgesThick,cmap='gray');ax[1,2].set_title('binary edges');ax[1,2].axis('off')
        ax[1,3].imshow(img,cmap='gray');ax[1,3].imshow(displayEdges);ax[1,3].set_title('original image with edges');ax[1,3].axis('off')
        ax[1,4].imshow(imgPatch,cmap='gray');ax[1,4].imshow(displayEdges);ax[1,4].set_title('blurred image with edges');ax[1,4].axis('off')
        plt.tight_layout()
        return edgesThick.astype(np.bool), f

    else: return edgesThick.astype(np.bool)


def z_downsample(img, downSampleRate, is_verbose=True):
    '''
    downsample input image in z direction
    '''

    if len(img.shape) != 3:
        raise ValueError('Input array shoud be 3D!')


    newFrameNum = img.shape[0] // downSampleRate
    newImg = np.empty((newFrameNum,img.shape[1],img.shape[2]),dtype=img.dtype)

    if is_verbose:
        print('Start downsampling...')
    for i in range(newFrameNum):
#            print (float(i)*100/newFrameNum),'%'
        currChunk = img[i*downSampleRate:(i+1)*downSampleRate,:,:].astype(np.float)
        currFrame = np.mean(currChunk,axis=0)
        newImg[i,:,:]=currFrame.astype(img.dtype)
    if is_verbose:
        print('End of downsampling.')
    return newImg


def get_masks(labeled, minArea=None, maxArea=None, isSort=True, keyPrefix = None, labelLength=None):
    '''
    get mask dictionary from labeled map (labeled by scipy.ndimage.label function), masks with area smaller than
    minArea and maxArea will be discarded.

    :param labeled: 2d array with non-negative int, labelled map (ideally the output of scipy.ndimage.label function)
    :param minArea: positive int, minimum area criterion of retained masks
    :param maxArea: positive int, maximum area criterion of retained masks
    :param isSort: bool, sort the masks by area or not
    :param keyPrefix: str, the key prefix for returned dictionary
    :param labelLength: positive int, the length of key index

    :return masks: dictionary of 2d binary masks
    '''

    maskNum = np.max(labeled.flatten())
    masks = {}
    for i in range(1, maskNum + 1):
        currMask = np.zeros(labeled.shape, dtype=np.uint8)
        currMask[labeled == i] = 1

        if minArea is not None and np.sum(currMask.flatten()) < minArea:
            continue
        elif maxArea is not None and np.sum(currMask.flatten()) > maxArea:
            continue
        else:
            if labelLength is not None:
                mask_index = ft.int2str(i, labelLength)
            else:
                mask_index = str(i)

            if keyPrefix is not None:
                currKey = keyPrefix + '_' + mask_index
            else:
                currKey = mask_index
            masks.update({currKey: currMask})

    if isSort:
        masks = sort_masks(masks, keyPrefix=keyPrefix, labelLength=labelLength)

    return masks


def get_marked_masks(labeled, markCoor):
    '''
    return one binary masks which contain the marked coordinate
    labeled (maps labeled by scipy.ndimage.label function)
    '''

    masks = get_masks(labeled)
    for key, value in masks.items():
        if hit_or_miss(markCoor, value): return value
    return None


def sort_masks(masks, keyPrefix=None, labelLength=3):
    '''
    sort a dictionary of binary masks, big to small
    '''

    maskNum = len(masks.keys())
    order = []
    for key, mask in masks.items():
        order.append([key,np.sum(mask.flatten())])

    order = sorted(order, key=lambda a:a[1], reverse=True)

    newMasks = {}
    for i in range(len(order)):
        if keyPrefix is not None: currKey = keyPrefix+'_'+ft.int2str(i,labelLength)
        else: currKey = ft.int2str(i,labelLength)
        newMasks.update({currKey:masks[order[i][0]]})
    return newMasks


# def temp_downsample(A, rate, verbose=False):
#     '''
#     down sample a 3-d array in 0 direction
#     '''
#
#     if len(A.shape) != 3: raise ValueError('input array should be 3-d.')
#     rate = int(rate)
#     dataType = A.dtype
#     newZDepth = (A.shape[0] - (A.shape[0]%rate))/rate
#     newA = np.empty((newZDepth,A.shape[1],A.shape[2]),dtype=dataType)
#
#     for i in range(newZDepth):
#         if verbose:
#             print (float(i)*100/newZDepth),'%'
#             currChunk = A[i*rate:(i+1)*rate,:,:].astype(np.float)
#             currFrame = np.mean(currChunk,axis=0)
#             newA[i,:,:]=currFrame.astype(dataType)
#     return newA


def get_average_movie(mov, frameTS, onsetTimes, chunkDur, isReturnN=False):
    '''
    :param mov: image movie
    :param frameTS: the timestamps for each frame of the raw movie
    :param onsetTimes: time stamps of onset of each trigger
    :param chunkDur: duration of each chunk
    :return: averageed movie of all chunks
    '''

    meanFrameDur = np.mean(np.diff(frameTS))

    chunkFrameDur = int(np.ceil(chunkDur / meanFrameDur))

    # print('chunkDur:', chunkDur)
    # print('meanFrameDur:', meanFrameDur)
    # print('chunkFrameDur:', chunkFrameDur)

    sumMov = None
    n = 0.

    curr_onset = -1
    onset_num = len(onsetTimes)
    t0 = time.time()

    for i, onset in enumerate(onsetTimes):
        if onset >= frameTS[0] and onset + chunkDur <= frameTS[-1]:

            if i // (onset_num // 10) > curr_onset:
                # print(t0 - time.time(), ' second :', (i // (onset_num // 10)) * 10, '%')
                print('{:09.2f} second: {:2d} %'.format(time.time() - t0, (i // (onset_num // 10)) * 10))
                curr_onset = i // (onset_num // 10)

            onsetFrameInd = np.argmin(np.abs(frameTS-onset))
            # print('Chunk:',int(n),'; Starting frame index:',onsetFrameInd,'; Ending frame index', onsetFrameInd+chunkFrameDur)

            if onsetFrameInd+chunkFrameDur <= mov.shape[0]:
                if sumMov is None: sumMov = np.zeros((chunkFrameDur,mov.shape[1],mov.shape[2]), dtype=np.float64)
                sumMov += mov[onsetFrameInd:onsetFrameInd+chunkFrameDur,:,:].astype(np.float64)
                n += 1.
            else:
                print('Ending frame index ('+str(int(onsetFrameInd+chunkFrameDur))+') is larger than frames in movie (' +
                      str(int(mov.shape[0]))+'.\nExclude this trigger.')
                continue

    if isReturnN:
        return sumMov.astype(np.float32) / n, int(n)
    else:
        return sumMov.astype(np.float32) / n


def get_average_movie2(mov, frameTS, onsetTimes, chunkDur, verbose=True):
    '''
    :param mov: image movie
    :param frameTS: the timestamps for each frame of the raw movie
    :param onsetTimes: time stamps of onset of each trigger
    :param chunkDur: duration of each chunk
    :return: averageed movie of all chunks, number of chunks that were averaged
    '''

    meanFrameDur = np.mean(np.diff(frameTS))

    chunkFrameDur = int(np.ceil(chunkDur / meanFrameDur))

    # print('chunkDur:', chunkDur)
    # print('meanFrameDur:', meanFrameDur)
    # print('chunkFrameDur:', chunkFrameDur)

    sumMov = None
    real_count = 0

    curr_onset = -1
    onset_num = len(onsetTimes)
    t0 = time.time()

    for i, onset in enumerate(onsetTimes):

        if verbose and (i // (onset_num // 10) > curr_onset):
            print('{:09.2f} second: {:2d} %'.format(time.time() - t0, (i // (onset_num // 10)) * 10))
            curr_onset = i // (onset_num // 10)

        if onset < frameTS[0]:  # the onset is before imaging start time. Exclude this onset.
            continue
            # print('onset number:', count, 'is before imaging start time. Exclude this onset.')

        else:
            onsetFrameInd = np.argmin(np.abs(frameTS-onset))
            # if verbose:
            #     print('Chunk:',int(count),'; Starting frame index:',onsetFrameInd,'; Ending frame index', onsetFrameInd+chunkFrameDur)

            if onsetFrameInd+chunkFrameDur <= mov.shape[0]:
                if sumMov is None: sumMov = np.zeros((chunkFrameDur,mov.shape[1],mov.shape[2]))
                sumMov += mov[onsetFrameInd:onsetFrameInd+chunkFrameDur,:,:].astype(np.float32)
                real_count += 1.
            else:  # the chunk exceeds the end of imaging.
                continue
                # print('the chunk of onset number', count, 'exceeds the end of imaging. Exclude this onset.')


    if sumMov is None:
        print('\nNo valid chunk found!')
        return np.zeros((chunkFrameDur,mov.shape[1],mov.shape[2]), dtype=np.float32), 0
    else:
        if verbose:
            print('\n' + str(int(real_count)) + ' valid chunks found.')
        return sumMov.astype(np.float32) / real_count, int(real_count)


def regression_detrend_1d(sig, trend):
    """
    detrend a signal trace by subtracting global trend (surround neural pil). It uses linear regress to determine the
    contribution of the global trend to the signal.

    ref:
    1. J Neurosci. 2016 Jan 27;36(4):1261-72. doi: 10.1523/JNEUROSCI.2744-15.2016. Resolution of High-Frequency
    Mesoscale Intracortical Maps Using the Genetically Encoded Glutamate Sensor iGluSnFR. Xie Y, Chan AW, McGirr A,
    Xue S, Xiao D, Zeng H, Murphy TH.
    2. Neuroimage. 1998 Oct;8(3):302-6. The inferential impact of global signal covariates in functional neuroimaging
    analyses. Aguirre GK1, Zarahn E, D'Esposito M.

    :param sig: input signal, 1-d array
    :param trend: binary, global trend, same shape as sig
    :return: detrended signal, contribution, rvalue
    """

    if len(sig.shape) != 1:
        raise ValueError('Input signal should be 1-dimensional!')

    if len(trend.shape) != 1:
        raise ValueError('Input trend should be 1-dimensional!')

    if sig.shape != trend.shape:
        raise ValueError('Input trend should have same shape as input signal')

    sig = sig.astype(np.float)
    trend = trend.astype(np.float)
    trend = trend - np.mean(trend)

    slope, intercept, r_value, p_value, stderr = stats.linregress(trend, sig)

    sig_detrend = (sig - trend * slope).astype(sig.dtype)

    return sig_detrend, slope, r_value


def merge_weighted_rois(roi1, roi2):
    """
    merge two WeightedROI objects, most useful for merge ON and OFF subfields
    """
    if (roi1.pixelSizeX != roi2.pixelSizeX) or (roi1.pixelSizeY != roi2.pixelSizeY):
        raise ValueError('The pixel sizes of the two WeightedROI objects should match!')

    if roi1.pixelSizeUnit != roi2.pixelSizeUnit:
        raise ValueError('The pixel size units of the two WeightedROI objects should match!')

    mask1 = roi1.get_weighted_mask()
    mask2 = roi2.get_weighted_mask()

    return WeightedROI(mask1 + mask2, pixelSize=[roi1.pixelSizeY, roi1.pixelSizeX], pixelSizeUnit=roi1.pixelSizeUnit)


def merge_binary_rois(roi1, roi2):
    """
    merge two ROI objects, most useful for merge ON and OFF subfields
    """
    if (roi1.pixelSizeX != roi2.pixelSizeX) or (roi1.pixelSizeY != roi2.pixelSizeY):
        raise ValueError('The pixel sizes of the two WeightedROI objects should match!')

    if roi1.pixelSizeUnit != roi2.pixelSizeUnit:
        raise ValueError('The pixel size units of the two WeightedROI objects should match!')

    mask1 = roi1.get_binary_mask()
    mask2 = roi2.get_binary_mask()
    mask3 = np.logical_or(mask1, mask2).astype(np.int8)

    return ROI(mask3, pixelSize=[roi1.pixelSizeY, roi1.pixelSizeX], pixelSizeUnit=roi1.pixelSizeUnit)


def get_peak_weighted_roi(arr, thr):
    """
    return: a WeightROI object representing the mask which contains the peak of arr and cut by the lev_thr (lev_thr)
    """
    nanLabel = np.isnan(arr)
    arr2 = arr.copy()
    arr2[nanLabel] = np.nanmin(arr)
    labeled, _ = ni.label(arr2 >= thr)
    peakCoor = np.array(np.where(arr2 == np.amax(arr2))).transpose()[0]
    peakMask = get_marked_masks(labeled, peakCoor)
    if peakMask is None:
        'Threshold too high! No ROI found. Returning None'; return None
    else:
        return WeightedROI(arr2 * peakMask)


def pairwise_distance(coords):
    """
    giving coordinates of a set of points, return the pairwise distances of all pairs
    :param coords: 2d array, shape is (n, 2). first column: x coordinates, second column: y coordinates.
    :return:
    """

    if len(coords.shape) != 2:
        raise ValueError("input coordinates should be 2d array.")

    if coords.shape[1] != 2:
        raise ValueError("input coordinates should have 2 columns.")

    if coords.shape[0] < 2:
        return np.array([])
    else:
        point_num = coords.shape[0]
        pairs = np.zeros((point_num * (point_num - 1), 4))

        pair_ind = 0
        for i in range(0, point_num - 1):
            for j in range(i + 1, point_num):
                pairs[pair_ind, :] = [coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]]
                pair_ind = pair_ind + 1

        dis = np.sqrt(np.square(pairs[:, 0] - pairs[:, 2]) + np.square(pairs[:, 1], pairs[:, 3]))
        return dis


def pairwise_magnification(coords1, coords2):
    """
    giving two sets of coordinates of a set of points, say receptive field center location and cortical
    location of a set of rois in field of view

    return the pairwise magnification distance of coords1 over distance of coords2 of all pairs

    :param coords1: 2d array, shape is (n, 2). first column: x coordinates, second column: y coordinates.
    :param coords2: 2d array, shape is (n, 2). first column: x coordinates, second column: y coordinates.
                    coords1 one and coords2 should have same shape.
    :return:
    """

    if len(coords1.shape) != 2:
        raise ValueError("input coordinates should be 2d array.")

    if coords1.shape[1] != 2:
        raise ValueError("input coordinates should have 2 columns.")

    if coords1.shape != coords2.shape:
        raise ValueError("two input coordinates should have same shape.")

    if coords1.shape[0] < 2:
        return np.array([])
    else:
        point_num = coords1.shape[0]
        pairs = np.zeros((point_num * (point_num - 1) / 2, 8))

        pair_ind = 0
        for i in range(0, point_num - 1):
            for j in range(i + 1, point_num):

                pairs[pair_ind, :] = [coords1[i, 0], coords1[i, 1], coords1[j, 0], coords1[j, 1],
                                      coords2[i, 0], coords2[i, 1], coords2[j, 0], coords2[j, 1]]
                pair_ind = pair_ind + 1

        mag = np.sqrt(np.square(pairs[:, 0] - pairs[:, 2]) + np.square(pairs[:, 1], pairs[:, 3])) / \
              np.sqrt(np.square(pairs[:, 4] - pairs[:, 6]) + np.square(pairs[:, 5], pairs[:, 7]))

        return mag


def get_circularity(mask, is_skimage=True):
    """
    return circularity of the shape marked by the input mask. If the mask label more than one
    continuous regions, only analyze the first one retuned by scipy.ndimage.label.

    This does not consider holes.

    there are two ways to estimate perimeter:
    1. is_skimage=True uses the shape through the center of border pixels. if the labeled region is
       large and wide in all orientation, the measurement is very precise. But if the labeled region
       is small or narrow, it underestmates the perimeter thus overestimates the circularity. Sometimes
       it can be larger than 1.

    2. is_skimage=False uses the outer boundary line of border pixels. This will treat all shapes as
       rectangle, thus systematically underestimates the circularity. upper bound will be the
       circularity of square: 0.7853981633974483.

    :param mask: 2d binary array, if not binary, all pixel <= zero will be considered as 0.
                 all pixels > 0 will be considered as 1.
    :param is_skimage: bool. if Ture, use skimage.measure.perimeter to estimate perimeter
    :return: circularity, defined by 4 * pi * area / (perimeter) ^ 2
    """

    if len(mask.shape) != 2:
        raise ValueError('input mask should be a 2d array.')

    msk = np.zeros(mask.shape, dtype=np.uint8)
    msk[mask>0] = 1

    labeled, roi_num = ni.label(msk)

    if roi_num > 1: # found more than one labeled regions
        # raise(ValueError('input mask should have only one continuous region labeled. {} found.'.format(roi_num)))
        print('input mask has {} (> 1) continuous regions labeled. only analyze the first one'.format(roi_num))

        msk[:] = 0
        msk[labeled == 1] = 1
    elif roi_num == 0: # found no labeled region
        print('Did not find labeled region. Returning None')
        return None
    else: # found one labeled region
        pass

    if is_skimage:
        perimeter = measure.perimeter(msk)
    else:
        rows, cols = np.where(msk == 1)
        perimeter = 2. * (max(rows) - min(rows) + 1.) + 2. * (max(cols) - min(cols) + 1.)
    area = np.sum(msk.flat)

    return 4 * np.pi * area / (perimeter ** 2)


def fit_ellipse(mask):
    """
    using opencv to fit a mask into ellipse

    :param mask: 2d array, dtype np.uint8
    :return:
    """

    mask2 = np.array(mask)

    if len(mask.shape) != 2:
        raise ValueError('input "mask" should be a 2d array.')

    if not mask.dtype == np.uint8:
        raise ValueError("input mask should have dtype as np.uint8")

    fitted = cv2.findContours(image=mask2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    if len(fitted) == 3:
        cons = fitted[1]
    elif len(fitted) == 2:
        cons = fitted[0]
    else:
        raise ValueError

    if len(cons) == 0:
        print('ImageAnalysis.fit_ellipse: No contour detected. Returning None.')
        return None
    elif cons[0].shape[0] < 5:
        print('ImageAnalysis.fit_ellipse: No contour detected. Returning None.')
        return None
    else:
        if len(cons) > 1:
            print('ImageAnalysis.fit_ellipse: More than one contours detected. Taking the first one.')
        con = cons[0]

    box = cv2.fitEllipse(con)
    ell = Ellipse.from_cv2_box(box)
    return ell


class ROI(object):
    '''
    class of binary ROI
    '''

    def __init__(self, mask, pixelSize = None, pixelSizeUnit = None):
        '''
        :param mask: 2-d array, if not binary, non-zero and non-nan pixel will be included in mask,
                     zero-pixel will be considered as background
        :param pixelSize: float, can be None, one value (square pixel) or (width, height) for non-square pixel
        :param pixelSizeUnit: str, the unit of pixel size
        '''

        if len(mask.shape)!=2: raise ValueError('Input mask should be 2d.')

        self.dimension = mask.shape
        self.pixels = np.where(np.logical_and(mask!=0, ~np.isnan(mask)))

        if pixelSize is None:
            self.pixelSizeX = self.pixelSizeY = pixelSize
            self.pixelSizeUnit = None
        elif (not hasattr(pixelSize, '__len__')):
            self.pixelSizeX = self.pixelSizeY = pixelSize
        elif len(pixelSize)==2:
            self.pixelSizeY = pixelSize[0]; self.pixelSizeX = pixelSize[1]
        else:
            raise LookupError('pixel size should be either None or scalar or list(array) of two sclars!!')

        if isinstance(pixelSizeUnit, str) :
            self.pixelSizeUnit = pixelSizeUnit
        elif pixelSizeUnit is None:
            self.pixelSizeUnit = None
        else:
            try:
                self.pixelSizeUnit = pixelSizeUnit.decode('utf-8')
            except Exception as e:
                print('Do not understand input "pixelSizeUnit": {}. Setting it to None'.format(pixelSizeUnit))
                print(e)
                self.pixelSizeUnit = None

    def __str__(self):
        return 'NeuroAnalysisTools.core.ImageAnalysis.ROI object'

    def set_pixel_size(self, pixelSize):
        if pixelSize is None: self.pixelSizeX = self.pixelSizeY = pixelSize
        elif (not hasattr(pixelSize, '__len__')): self.pixelSizeX = self.pixelSizeY = pixelSize
        elif len(pixelSize)==2: self.pixelSizeY = pixelSize[0]; self.pixelSizeX = pixelSize[1]
        else: raise LookupError('pixel size should be either None or scalar or list(array) of two sclars!!')

    def set_pixel_size_unit(self, pixelSizeUnit):
        if self.pixelSizeY is None or self.pixelSizeX is None:
            print('ROI pixel size is None. Setting pixel size unit to None')
            self.pixelSizeUnit=None
        else: self.pixelSizeUnit = pixelSizeUnit

    def get_indices(self):
        '''
        :return: index list of the pixels in the ROI
        '''
        return self.pixels

    def get_pixel_array(self):
        """
        :return: ndarray version of list of pixels, each row is a pixel, each column is dimension [row, col]
        """
        return np.array(self.pixels).transpose()

    def get_pixel_list(self):
        """
        :return: list version of pixels, each item of the list is a pixel, [row, col]
        """
        pixel_array = self.get_pixel_array()
        return [list(p) for p in pixel_array]

    def get_pixel_tuple(self):
        """
        :return: tuple version of pixels, each item of the list is a pixel, [row, col]
        """
        pixel_array = self.get_pixel_array()
        return tuple([tuple(p) for p in pixel_array])

    def get_binary_mask(self):
        '''
        generate binary mask of the ROI, return 2d array, with 0s and 1s, dtype np.uint8
        '''
        mask = np.zeros(self.dimension,dtype=np.uint8)
        mask[self.pixels] = 1
        return mask

    def get_nan_mask(self):
        '''
        generate float mask of the ROI, return 2d array, with nans and 1s, dtype np.float32
        '''
        mask = np.zeros(self.dimension,dtype=np.float32)
        mask[:] = np.nan
        mask[self.pixels] = 1
        return mask

    def get_pixel_area(self, verbose=False):
        '''
        return the area coverage of the ROI
        '''

        if (self.pixelSizeX is not None) and (self.pixelSizeX is not None):
            if verbose:
                print('returning area with unit:' + self.pixelSizeUnit + '^2')
            return float(len(self.pixels[0]))*self.pixelSizeX*self.pixelSizeY
        else:
            print('Did not find information about pixel size. '
                  'Returning area as pixel counts without unit.')
            return len(self.pixels[0])

    def get_binary_area(self):
        '''
        :return: number of pixels in the roi
        '''
        return len(self.pixels[0])

    def get_center(self, is_real_size=False):
        '''
        return the center coordinates [Y, X] of the centroid of the mask

        :param is_real_size: bool, if True and there is information about pixel size,
                             the values will reflect pixel size; else return values at
                             index scale.
        '''
        center = np.mean(np.array(self.pixels,dtype=np.float).transpose(),axis=0)

        if is_real_size and self.pixelSizeX is not None and self.pixelSizeY is not None:
            return center * np.array([self.pixelSizeY, self.pixelSizeX])
        else:
            return center

    def get_binary_trace(self, mov):
        '''
        return trace of this ROI (binary format, 0s and 1s) in a given movie
        '''
        binaryMask = self.get_binary_mask().astype(np.float32)
        trace = np.multiply(mov, np.array([binaryMask])).sum(axis=1).sum(axis=1)
        # print(trace)
        return trace / self.get_binary_area()

    def get_binary_trace_pixelwise(self, mov):
        '''
        return trace of this ROI (binary format, 0s and 1s) in a given movie,
        calculation is done in pixelwise fashion
        '''
        pixels = self.get_pixel_array()
        trace = np.zeros(mov.shape[0], dtype=np.float32)
        for pixel in pixels:
            # trace += mov[:, pixel[0], pixel[1]]  # somehow this is less precise !! do not use
            trace = trace + mov[:, int(pixel[0]), int(pixel[1])].flatten().astype(np.float32)
        # print(trace)
        return trace / self.get_binary_area()

    def plot_binary_mask(self, plotAxis=None, color='#ff0000', alpha=1):
        '''
        return display image (RGBA uint8 format) which can be plotted by plt.imshow, alpha: transparency 0-1
        '''
        mask = self.get_binary_mask()
        displayImg = pt.binary_2_rgba(mask, foregroundColor=color, backgroundColor='#000000', foregroundAlpha=int(alpha * 255), backgroundAlpha=0)
        if plotAxis is None: f=plt.figure();plotAxis=f.add_subplot(111);plotAxis.imshow(displayImg,interpolation='nearest')
        return displayImg

    def plot_binary_mask_border(self, **kwargs):
        pt.plot_mask_borders(self.get_nan_mask(), **kwargs)

    def to_h5_group(self, h5Group):
        '''
        add attributes and dataset to a h5 data group
        '''
        h5Group.attrs['dimension'] = self.dimension
        h5Group.attrs['description'] = str(self)
        if self.pixelSizeX is None: h5Group.attrs['pixelSize'] = 'None'
        else: h5Group.attrs['pixelSize'] = [self.pixelSizeY, self.pixelSizeX]
        if self.pixelSizeUnit is None: h5Group.attrs['pixelSizeUnit'] = 'None'
        else: h5Group.attrs['pixelSizeUnit'] = self.pixelSizeUnit

        dataDict = dict(self.__dict__)
        _ = dataDict.pop('dimension');_ = dataDict.pop('pixelSizeX');_ = dataDict.pop('pixelSizeY');_ = dataDict.pop('pixelSizeUnit')
        for key, value in dataDict.items():
            if value is None: h5Group.create_dataset(key,data='None')
            else: h5Group.create_dataset(key,data=value)

    def binary_overlap(self, roi):
        """
        :param roi: another ROI object, should have same dimension as self
        :return: the number of overlapping pixels between self and the input roi
        """
        if roi.dimension != self.dimension:
            raise ValueError('the dimensions of input roi are different from self dimensions!')

        pixel_list = list(self.get_pixel_list()) + list(roi.get_pixel_list())
        pixel_tuple = tuple([tuple(p) for p in pixel_list])
        pixel_set = set(pixel_tuple)
        return len(pixel_tuple) - len(pixel_set)

    @staticmethod
    def from_h5_group(h5Group):
        '''
        load ROI (either ROI or WeightedROI) object from a hdf5 data group
        '''

        dimension = h5Group.attrs['dimension']
        pixelSize = h5Group.attrs['pixelSize']
        if pixelSize == 'None': pixelSize = None
        pixelSizeUnit = h5Group.attrs['pixelSizeUnit']
        if pixelSizeUnit == 'None': pixelSizeUnit = None
        pixels = h5Group['pixels'][()]

        if 'weights' in h5Group.keys():
            weights = h5Group['weights'][()]
            mask = np.zeros(dimension, dtype=np.float32)
            mask[tuple(pixels)] = weights
            roi = WeightedROI(mask, pixelSize=pixelSize, pixelSizeUnit=pixelSizeUnit)
        else:
            mask = np.zeros(dimension, dtype=np.uint8)
            mask[tuple(pixels)] = 1
            roi = ROI(mask, pixelSize=pixelSize, pixelSizeUnit=pixelSizeUnit)

        for key in h5Group.keys():
            if key not in ['pixels', 'weights']:
                if h5Group[key].value == 'None':
                    setattr(roi, key, None)
                else:
                    setattr(roi, key, h5Group[key].value)

        return roi


class WeightedROI(ROI):

    def __init__(self, mask, pixelSize = None, pixelSizeUnit = None):
        super(WeightedROI,self).__init__(mask, pixelSize = pixelSize, pixelSizeUnit = pixelSizeUnit)
        self.weights = mask[self.pixels]

    def __str__(self):
        return 'NeuroAnalysisTools.core.ImageAnalysis.WeightedROI object'

    def get_peak(self):
        return np.max(self.weights)

    def get_weight_sum(self):
        return sum(self.weights)

    def get_mean_weight(self):
        return np.mean(self.weights)

    def get_weighted_mask(self):
        mask = np.zeros(self.dimension,dtype=np.float32)
        mask[self.pixels] = self.weights
        return mask

    def get_weighted_nan_mask(self):
        mask = np.zeros(self.dimension,dtype=np.float32)
        mask[:]=np.nan
        mask[self.pixels] = self.weights
        return mask

    def get_weighted_center(self):
        pixelCor = np.array(self.pixels,dtype=np.float)
        center = np.sum(np.multiply(pixelCor,np.array(self.weights)),axis=1)/np.sum(self.weights)
        return center

    def get_weighted_center_in_coordinate(self, yCor, xCor):
        '''
        return weighted center of the ROI in the coordinate system defined by np.meshgrid(xCor, yCor)
        '''
        weightMask = self.get_weighted_mask()

        if np.sum(weightMask.flatten()) == 0:
            return [np.nan, np.nan]
        else:
            xMap, yMap = np.meshgrid(xCor, yCor)
            xCenter = np.sum((xMap*weightMask).flatten())/np.sum(weightMask.flatten())
            yCenter = np.sum((yMap*weightMask).flatten())/np.sum(weightMask.flatten())
            return [yCenter, xCenter]

    def plot_weighted_mask(self, plotAxis=None, is_colorbar=False, cmap='Reds', interpolation='nearest', **kwargs):
        '''
        return display image (RGBA uint8 format) which can be plotted by plt.imshow
        '''
        mask = self.get_weighted_mask()

        if plotAxis is None:
            f=plt.figure(); plotAxis=f.add_subplot(111)

        curr_plot = plotAxis.imshow(mask, cmap=cmap, interpolation=interpolation, **kwargs)

        if is_colorbar:
            plotAxis.get_figure().colorbar(curr_plot)

        return plotAxis.get_figure()

    def get_weighted_trace(self, mov, is_area_weighted=False):
        '''
        return trace of this ROI in a given movie, the contribution of each pixel in the roi was defined by its weight
        :param is_area_weighted: bool, if False, total area of the mask is calculated in a binary fashion
                                       if True, total area of mask is calculated in a weighted fashion
        '''
        weightedMask = self.get_weighted_mask()
        trace = np.multiply(mov, np.array([weightedMask])).sum(axis=-1).sum(axis=-1)
        # print(trace)
        if is_area_weighted:
            return trace / self.get_binary_area()
        elif not is_area_weighted:
            return trace / self.get_weight_sum()
        else:
            raise ValueError('is_area_weighted should be a boolean variable.')

    def get_weighted_trace_pixelwise(self, mov, is_area_weighted=False):
        '''
        return trace of this ROI in a given movie, the contribution of each pixel in the roi was defined by its weight
        :param is_area_weighted: bool, if False, total area of the mask is calculated in a binary fashion
                                       if True, total area of mask is calculated in a weighted fashion
        calculation is done in pixelwise fashion
        '''
        pixels = self.get_pixel_array()
        trace = np.zeros(mov.shape[0], dtype=np.float32)
        for i, pixel in enumerate(pixels):
            # trace += mov[:, pixel[0], pixel[1]]  # somehow this is less precise !! do not use
            trace = trace + self.weights[i] * (mov[:, pixel[0], pixel[1]]).astype(np.float32)
        # print(trace)
        if not is_area_weighted:
            return trace / self.get_binary_area()
        elif is_area_weighted:
            return trace / self.get_weight_sum()
        else:
            raise ValueError('is_area_weighted should be a boolean variable.')

    def ellipse_fitting(self, thr=None, is_plot=False):
        """
        using opencv to fit a ellipse

        :param thr: float, threshold to threshold the mask
        :param is_plot: bool
        :return ell: NeuroAnalysisTools.ImageAnalysis.Ellipse object
        """

        if thr is None:
            mask_thr = self.get_binary_mask() * 255
        else:
            mask = self.get_weighted_mask()
            mask_thr = np.zeros(mask.shape, dtype=np.uint8)
            mask_thr[mask >= thr] = 255

        ell = fit_ellipse(mask_thr)

        if is_plot:
            f = plt.figure()
            ax = f.add_subplot(111)
            img = np.array([mask_thr, mask_thr, mask_thr]).transpose((1, 2, 0)).copy()
            if ell is not None:
                img = ell.draw(img=img, thickness=2)
                ax.set_title('angle={} deg'.format(ell.angle))
            img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
            ax.imshow(img, interpolation='nearest')
            plt.show()

        return ell

    @staticmethod
    def from_h5_group(h5Group):
        '''
        load WeightedROI (either ROI or WeightedROI) object from a hdf5 data group
        '''

        dimension = h5Group.attrs['dimension']
        pixelSize = h5Group.attrs['pixelSize']
        if pixelSize == 'None': pixelSize = None
        pixelSizeUnit = h5Group.attrs['pixelSizeUnit']
        if pixelSizeUnit == 'None': pixelSizeUnit = None
        pixels = h5Group['pixels'][()]
        weights = h5Group['weights'][()]
        mask = np.zeros(dimension, dtype=np.float32)
        mask[tuple(pixels)] = weights

        roi = WeightedROI(mask, pixelSize=pixelSize, pixelSizeUnit=pixelSizeUnit)

        for key in h5Group.keys():
            if key not in ['pixels', 'weights']:
                if h5Group[key].value == 'None':
                    setattr(roi, key, None)
                else:
                    setattr(roi, key, h5Group[key].value)

        return roi


class Ellipse(object):
    """
    ellipse object

    :attribute center: tuple of two floats, (center height, center width)
    :attribute axes: tuple of two positive floats, (radius of the long axis, radius of short axis)
    :attribute angle: float, degree, counterclockwise rotation of long axis, from right direction
    """

    def __init__(self, center, axes, angle):
        """
        ellipse object

        :param center: tuple of two floats, (center row, center col)
        :param axes: tuple of two positive floats, (radius of the long axis, radius of short axis)
        :param angle: float, degree, counterclockwise rotation of long axis, from right direction
        """
        self.center = center

        if axes[0] <= 0. or axes[1] <= 0.:
            raise ValueError('length of axes should be larger than 0.')

        if axes[0] >= axes[1]:
            self.axes = axes
            self.angle = angle % 180.
        else:
            self.axes = (axes[1], axes[0])
            self.angle = (angle + 90.) % 180.

    def get_aspect_ratio(self):
        return(self.axes[0] / self.axes[1])

    def get_cv2_ellips(self):
        """
        :return: the ellipse in opencv3 format for drawing
        """
        # return ((int(round(self.center[1])), int(round(self.center[0]))),
        #         (int(round(self.axes[0])), int(round(self.axes[1]))),
        #         -self.angle, 0, 360)
        return ((int(self.center[1]), int(self.center[0])),
                (int(self.axes[0]), int(self.axes[1])),
                -self.angle, 0, 360)

    def get_area(self):
        return np.pi * self.axes[0] * self.axes[1]

    def get_binary_mask(self, shape):
        """
        :param shape: tuple of 2 positive integers (height, width)
        :return: binary mask of the ellipse with given shape
        """
        mask = np.zeros(shape=shape, dtype=np.uint8)
        ell_cv2 = self.get_cv2_ellips()
        mask = cv2.ellipse(mask, center=ell_cv2[0], axes=ell_cv2[1], angle=ell_cv2[2], startAngle=0, endAngle=360,
                           color=1, thickness=-1)
        return mask.astype(np.uint8)

    def get_intensity(self, img):
        """
        :param img: 2d gray scale image
        :return: mean intensity of ellipse
        """

        if len(img.shape) != 2:
            raise ValueError('input image should be 2d array.')

        mask = self.get_binary_mask(img.shape)
        return np.mean(img[mask])

    def draw(self, img, color=(0, 255, 0), thickness=3):
        """
        :param img: 3d array, (height x width x channel), opencv frame
        :param color: list or tuple of 3 integers (unsigned 8-bit). RGB color
        :param thickness:
        :return:
        """

        ell_cv2 = self.get_cv2_ellips()

        # change color from RGB to BGR for opencv
        new_color = (color[2], color[1], color[0])
        img_marked = cv2.ellipse(img=img, center=ell_cv2[0], axes=ell_cv2[1], angle=ell_cv2[2], startAngle=ell_cv2[3],
                                 endAngle=ell_cv2[4], color=new_color, thickness=thickness)

        # img_marked = cv2.ellipse(img, box=self.to_cv2_box(), color=color, thickness=thickness)

        return img_marked

    def copy(self):
        return Ellipse(center=self.center,
                       axes=self.axes,
                       angle=self.angle)

    def info(self):
        s = 'center: ({:6.2f}, {:6.2f})\n'.format(self.center[0], self.center[1])
        s += 'axes:  ({:6.2f}, {:6.2f})\n'.format(self.axes[0], self.axes[1])
        s += 'angle: {:8.2f} deg\n'.format(self.angle)
        s += 'area: {:9.2f}\n'.format(self.get_area())
        return s

    @staticmethod
    def from_cv2_box(box):
        """
        get Ellipse object from cv2 rotated rectangle object (from cv2.fitEllipse() function)
        """
        center = (box[0][1], box[0][0])
        axes = (box[1][0] / 2., box[1][1] / 2.)
        angle = -box[2]
        return Ellipse(center=center, axes=axes, angle=angle)

    def to_cv2_box(self):
        return ((self.center[1], self.center[0]), (self.axes[0] * 2., self.axes[1] * 2), -self.angle)


if __name__ == '__main__':

    #============================================================
    # a = np.random.rand(100,100)
    # mask = generate_oval_mask(a,[45,58],20,30,isplot=True)
    # plt.show()

    #============================================================
    # a = np.arange(400).reshape((20,20))
    # b = rigid_transform(a,2,30,(1,5),(30,25))
    # f,ax=plt.subplots(1,2)
    # ax[0].imshow(a,interpolation='nearest')
    # ax[1].imshow(b,interpolation='nearest')
    # plt.show()
    #============================================================

    #============================================================
    # import tifffile as tf
    # imgPath = r"E:\data2\2015-05-28-Average-Ai93-Rorb-Scnn1a-map\AverageVasMap_Ai93.tif"
    # img = tf.imread(imgPath)
    # edges = get_area_edges(img)
    # plt.show()
    #============================================================

    #============================================================
    # aa=np.zeros((15,15),dtype=np.uint8)
    # aa[4,5]=1
    # aa[5,6]=1
    # aa[12:15,8:13]=1
    # bb=remove_small_patches(aa,5)
    # f,ax=plt.subplots(1,2)
    # ax[0].imshow(aa,interpolation='nearest');ax[1].imshow(bb,interpolation='nearest')
    # plt.show()
    #============================================================

    #============================================================
    # a=5; b=7
    # print distance(a,b)
    #
    # c=[5,6]; d=[8,2]
    # print distance(c,d)
    #
    # e=np.random.rand(5,6); f=np.random.rand(5,6)
    # print distance(e,f)
    #============================================================

    #============================================================
    # a=np.array(range(15)+range(10)[::-1]).reshape((5,5))
    # print(a)
    # labeled,_ = ni.label(a>7)
    # peakCoor = np.array(np.where(a==np.amax(a))).transpose()[0]
    # print(peakCoor)
    # peakMask = get_marked_masks(labeled,peakCoor)
    # plt.imshow(peakMask,interpolation='nearest')
    # plt.show()
    #============================================================

    #============================================================
    # mov = np.arange(64).reshape((4,4,4))
    # print(mov)
    #
    # mask1 = np.zeros((4,4)); mask1[2,2]=1; mask1[1,1]=1
    # trace1 = get_trace(mov,mask1,maskMode='binary')
    # assert(trace1[2] == 39.5)
    #
    # mask2 = np.zeros((4,4),dtype=np.float); mask2[:]=np.nan; mask2[2,2]=1; mask2[1,1]=1
    # trace2 = get_trace(mov,mask2,maskMode='binaryNan')
    # assert(trace2[2] == 39.5)
    #
    # mask3 = np.zeros((4,4),dtype=np.float); mask3[2,2]=1; mask3[1,1]=2
    # trace3 = get_trace(mov,mask3,maskMode='weighted')
    # assert(trace3[2] == 58)
    #
    # mask4 = np.zeros((4,4),dtype=np.float); mask4[:]=np.nan; mask4[2,2]=1; mask4[1,1]=2
    # trace4 = get_trace(mov,mask4,maskMode='weightedNan')
    # assert(trace4[2] == 58)
    #============================================================

    #============================================================
    # mov = np.arange(64).reshape((4,4,4))
    # np.save(r'E:\data\python_temp_folder\test_array.npy',mov)
    # bl_obj = BinarySlicer(r'E:\data\python_temp_folder\test_array.npy')
    #
    # mask1 = np.zeros((4,4)); mask1[2,2]=1; mask1[1,1]=1
    # mask2 = np.zeros((4,4)); mask2[3,0]=1; mask2[3,1]=1
    #
    # masks = {'mask1':mask1, 'mask2':mask2}
    # traces = get_trace_binaryslicer3(bl_obj,masks,mask_mode='binary',loading_frame_num=2)
    # print(traces)
    # assert(traces['trace_mask1'][2] == 39.5)
    # assert(traces['trace_mask2'][3] == 60.5)
    #============================================================


    # ============================================================
    # roi1 = np.zeros((10, 10))
    # roi1[4:8, 3:7] = 1
    # roi1 = ROI(roi1)
    # print(roi1.get_pixel_array())
    # print(roi1.get_pixel_list())
    # print(roi1.get_pixel_tuple())
    # roi2 = np.zeros((10, 10))
    # roi2[5:9, 5:8] = 1
    # roi2 = ROI(roi2)
    # print(roi1.binary_overlap(roi2))
    # ============================================================

    # ============================================================
    # sig = np.arange(100) * 0.1
    # trend = np.zeros((100,))
    # trend[5] = 1
    # detrended, slope, r = regression_detrend_1d(sig, trend)
    # print(detrended)
    # print(slope)
    # print(r)
    # ============================================================

    # ============================================================
    mov = np.arange(27).reshape((3,3,3)).astype(np.float)
    print(mov)
    mask1 = np.array([[0.,0.,0.],[0.,3.,0.],[0.,0.,0.]])
    roi1 = WeightedROI(mask1)
    mask2 = np.array([[1,1,1],[1,0,1],[1,1,0]])
    roi2 = ROI(mask2)
    trace1 = roi1.get_weighted_trace(mov)
    trace2 = roi2.get_binary_trace(mov)



    print(trace1)
    print(trace2)


    print('for debug')
