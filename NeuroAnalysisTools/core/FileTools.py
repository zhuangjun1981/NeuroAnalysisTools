__author__ = 'junz'

import numpy as np
import pickle
import os
import shutil
import struct
import h5py
import warnings
import numbers
import collections
from . import ImageAnalysis as ia

try:
    import tifffile as tf
except ImportError:
    import skimage.external.tifffile as tf

try:
    import cv2
except ImportError as e:
    print('cannot import OpenCV. {}'.format(e))


def unpack_uint32(uint32_array, endian='L'):
    """
    Unpacks an array of 32-bit unsigned integers into bits.

    Default is least significant bit first.

    *Not currently used by sync dataset because get_bit is better and does
        basically the same thing.  I'm just leaving it in because it could
        potentially account for endianness and possibly have other uses in
        the future.

    """
    if not uint32_array.dtype == np.uint32:
        raise TypeError("Must be uint32 ndarray.")
    buff = np.getbuffer(uint32_array)
    uint8_array = np.frombuffer(buff, dtype=np.uint8)
    uint8_array = np.fliplr(uint8_array.reshape(-1, 4))
    bits = np.unpackbits(uint8_array).reshape(-1, 32)
    if endian.upper() == 'B':
        bits = np.fliplr(bits)
    return bits


def get_bit(uint_array, bit):
    """
    Returns a bool array for a specific bit in a uint ndarray.

    Parameters
    ----------
    uint_array : (numpy.ndarray)
        The array to extract bits from.
    bit : (int)
        The bit to extract.

    """
    return np.bitwise_and(uint_array, 2 ** bit).astype(bool).astype(np.uint8)


def is_integer(var):
    return isinstance(var, numbers.Integral)


def saveFile(path,data):
    f = open(path,'wb')
    pickle.dump(data, f, protocol=2)
    f.close()


def loadFile(path, encoding='bytes'):
    f = open(path,'rb')
    data = pickle.load(f, encoding=encoding)
    f.close()
    return data


def copy(src, dest):
    '''
    copy everything from one path to another path. Work for both direcory and file.
    if src is a file, it will be copied into dest
    if src is a directory, the dest will have the same content as src
    '''

    if os.path.isfile(src):
        print('Source is a file. Starting copy...')
        try: shutil.copy(src,dest); print('End of copy.')
        except Exception as error: print(error)

    elif os.path.isdir(src):
        print('Source is a directory. Starting copy...')
        try: shutil.copytree(src, dest); print('End of copy.')
        except Exception as error: print(error)

    else: raise IOError('Source is neither a file or a directory. Can not be copied!')


def list_all_files(folder):
    '''
    get a list of full path of all files in a folder (including subfolder)
    '''
    files = []
    for folder_path, subfolder_paths, file_names in os.walk(folder):
        for file_name in file_names:
            files.append(os.path.join(folder_path,file_name))
    return files


def batchCopy(pathList, destinationFolder, isDelete=False):
    '''
    copy everything in the pathList into destinationFolder
    return a list of paths which can not be copied.
    '''

    if not os.path.isdir(destinationFolder): os.mkdir(destinationFolder)

    unCopied=[]

    for path in pathList:
        print('\nStart copying '+path+' ...')
        if os.path.isfile(path):
            print('This path is a file. Keep copying ...')
            try:
                shutil.copy(path,destinationFolder)
                print('End of copying.')
                if isDelete:
                    print('Deleting this file ...')
                    try: os.remove(path); print('End of deleting.\n')
                    except Exception as error: print('Can not delete this file.\nError message:\n'+str(error)+'\n')
                else: print('')
            except Exception as error: unCopied.append(path);print('Can not copy this file.\nError message:\n'+str(error)+'\n')

        elif os.path.isdir(path):
            print('This path is a directory. Keep copying ...')
            try:
                _, folderName = os.path.split(path)
                shutil.copytree(path,os.path.join(destinationFolder,folderName))
                print('End of copying.')
                if isDelete:
                    print('Deleting this directory ...')
                    try: shutil.rmtree(path); print('End of deleting.\n')
                    except Exception as error: print('Can not delete this directory.\nError message:\n'+str(error)+'\n')
                else: print('')
            except Exception as error: unCopied.append(path);print('Can not copy this directory.\nError message:\n'+str(error)+'\n')
        else:
            unCopied.append(path)
            print('This path is neither a file or a directory. Skip!\n')

    return unCopied


def importRawJCam(path,
                  dtype = np.dtype('>f'),
                  headerLength = 96, # length of the header, measured as the data type defined above
                  columnNumIndex = 14, # index of number of rows in header
                  rowNumIndex = 15, # index of number of columns in header
                  frameNumIndex = 16, # index of number of frames in header
                  decimation = None, #decimation number
                  exposureTimeIndex = 17): # index of exposure time in header, exposure time is measured in ms
    '''
    import raw JCam files into np.array


        raw file format:
        data type: 32 bit sigle precision floating point number
        data format: big-endian single-precision float, high-byte-first motorola
        header length: 96 floating point number
        column number index: 14
        row number index: 15
        frame number index: 16
        exposure time index: 17
    '''
    imageFile = np.fromfile(path,dtype=dtype,count=-1)

    columnNum = np.int(imageFile[columnNumIndex])
    rowNum = np.int(imageFile[rowNumIndex])

    if decimation is not None:
        columnNum /= decimation
        rowNum /= decimation

    frameNum = np.int(imageFile[frameNumIndex])

    if frameNum == 0: # if it is a single frame image
        frameNum += 1


    exposureTime = np.float(imageFile[exposureTimeIndex])

    imageFile = imageFile[headerLength:]

    print('width =', str(columnNum), 'pixels')
    print('height =', str(rowNum), 'pixels')
    print('length =', str(frameNum), 'frame(s)')
    print('exposure time =', str(exposureTime), 'ms')

    imageFile = imageFile.reshape((frameNum,rowNum,columnNum))

    return imageFile, exposureTime


def readBinaryFile(path,
                   position,
                   count = 1,
                   dtype = np.dtype('>f'),
                   whence = os.SEEK_SET):
    '''
    read arbitary part of a binary file,
    data type defined by dtype,
    start position defined by position (counts accordinating to dtype)
    length defined by count.
    '''

    f = open(path, 'rb')
    f.seek(position * dtype.alignment, whence)
    data = np.fromfile(f, dtype = dtype, count = count)
    f.close()
    return data


def readBinaryFile2(f,
                    position,
                    count = 1,
                    dtype = np.dtype('>f'),
                    whence = os.SEEK_SET):
    '''
    similar as readBinaryFile but without opening and closing file object
    '''
    f.seek((position * dtype.alignment), whence)
    data = np.fromfile(f, dtype = dtype, count = count)
    return data


def readBinaryFile3(f,
                    position,
                    count = 1,
                    dtype = np.dtype('>f'),
                    whence = os.SEEK_SET):
    '''
    reading arbituray byte from file without opening and closing file object
    not using np.fromfile
    '''
    f.seek(position * dtype.alignment, whence)
    data = struct.unpack('>f', f.read(count * dtype.alignment))
    return data


def importRawJPhys(path,
                   dtype = np.dtype('>f'),
                   headerLength = 96, # length of the header for each channel
                   channels = ('photodiode2','read','trigger','photodiode'),# name of all channels
                   sf = 10000): # sampling rate, Hz
    '''
    import raw JPhys files into np.array
    one dictionary contains header for each channel
    the other contains values for each for each channel
    '''

    JPhysFile = np.fromfile(path,dtype=dtype,count=-1)
    channelNum = len(channels)

    channelLength = len(JPhysFile) / channelNum

    if len(JPhysFile) % channelNum != 0:
        raise ArithmeticError('Length of the file should be divisible by channel number!')

    header = {}
    body = {}

    for index, channelname in enumerate(channels):
        channelStart = index * channelLength
        channelEnd = channelStart + channelLength

        header.update({channels[index]: JPhysFile[channelStart:channelStart+headerLength]})
        body.update({channels[index]: JPhysFile[channelStart+headerLength:channelEnd]})

    body.update({'samplingRate':sf})

    return header, body


def importRawNewJPhys(path,
                      dtype = np.dtype('>f'),
                      headerLength = 96, # length of the header for each channel
                      channels = ('photodiode2',
                                  'read',
                                  'trigger',
                                  'photodiode',
                                  'sweep',
                                  'visualFrame',
                                  'runningRef',
                                  'runningSig',
                                  'reward',
                                  'licking'),# name of all channels
                      sf = 10000): # sampling rate, Hz
    '''
    import new style raw JPhys files into np.array
    one dictionary contains header for each channel
    the other contains values for each for each channel
    '''

    JPhysFile = np.fromfile(path,dtype=dtype,count=-1)
    channelNum = len(channels)

    channelLength = len(JPhysFile) / channelNum
#    print('length of JPhys:', len(JPhysFile))
#    print('length of JPhys channel number:', channelNum)

    if len(JPhysFile) % channelNum != 0:
        raise ArithmeticError('Length of the file should be divisible by channel number!')

    JPhysFile = JPhysFile.reshape([int(channelLength), int(channelNum)])

    headerMatrix = JPhysFile[0:headerLength,:]
    bodyMatrix = JPhysFile[headerLength:,:]

    header = {}
    body = {}

    for index, channelname in enumerate(channels):

        header.update({channels[index]: headerMatrix[:,index]})
        body.update({channels[index]: bodyMatrix[:,index]})

    body.update({'samplingRate':sf})

    return header, body


def importRawJPhys2(path,
                    imageFrameNum,
                    photodiodeThr = .95, #threshold of photo diode signal,
                    dtype = np.dtype('>f'),
                    headerLength = 96, # length of the header for each channel
                    channels = ('photodiode2','read','trigger','photodiode'),# name of all channels
                    sf = 10000.): # sampling rate, Hz
    '''
    extract important information from JPhys file
    '''


    JPhysFile = np.fromfile(path,dtype=dtype,count=-1)
    channelNum = len(channels)

    channelLength = len(JPhysFile) / channelNum

    if channelLength % 1 != 0:
        raise ArithmeticError('Bytes in each channel should be integer !')

    channelLength = int(channelLength)

    # get trace for each channel
    for index, channelname in enumerate(channels):
        channelStart = index * channelLength
        channelEnd = channelStart + channelLength
#        if channelname == 'expose':
#            expose = JPhysFile[channelStart+headerLength:channelEnd]

        if channelname == 'read':
            read = JPhysFile[channelStart+headerLength:channelEnd]

        if channelname == 'photodiode':
            photodiode = JPhysFile[channelStart+headerLength:channelEnd]

#        if channelname == 'trigger':
#            trigger = JPhysFile[channelStart+headerLength:channelEnd]

    # generate time stamp for each image frame
    imageFrameTS = []
    for i in range(1,len(read)):
        if read[i-1] < 3.0 and read[i] >= 3.0:
            imageFrameTS.append(i*(1./sf))

    if len(imageFrameTS) < imageFrameNum:
        raise LookupError("Expose period number is smaller than image frame number!")
    imageFrameTS = imageFrameTS[0:imageFrameNum]

    # first time of visual stimulation
    visualStart = None

    for i in range(80,len(photodiode)):
        if ((photodiode[i] - photodiodeThr) * (photodiode[i-1] - photodiodeThr)) < 0 and \
           ((photodiode[i] - photodiodeThr) * (photodiode[i-75] - photodiodeThr)) < 0: #first frame of big change
                visualStart = i*(1./sf)
                break

    return np.array(imageFrameTS), visualStart


def importRawNewJPhys2(path,
                       imageFrameNum,
                       photodiodeThr = .95, #threshold of photo diode signal,
                       dtype = np.dtype('>f'),
                       headerLength = 96, # length of the header for each channel
                       channels = ('photodiode2',
                                   'read',
                                   'trigger',
                                   'photodiode',
                                   'sweep',
                                   'visualFrame',
                                   'runningRef',
                                   'runningSig',
                                   'reward',
                                   'licking'),# name of all channels
                       sf = 10000.): # sampling rate, Hz
    '''
    extract important information from new style JPhys file
    '''


    JPhysFile = np.fromfile(path,dtype=dtype,count=-1)
    channelNum = len(channels)

    channelLength = len(JPhysFile) / channelNum

    if len(JPhysFile) % channelNum != 0:
        raise ArithmeticError('Length of the file should be divisible by channel number!')

    JPhysFile = JPhysFile.reshape([channelLength, channelNum])

    bodyMatrix = JPhysFile[headerLength:,:]

    # get trace for each channel
    for index, channelname in enumerate(channels):

        if channelname == 'read':
            read = bodyMatrix[:,index]

        if channelname == 'photodiode':
            photodiode = bodyMatrix[:,index]

#        if channelname == 'trigger':
#            trigger = JPhysFile[channelStart+headerLength:channelEnd]

    # generate time stamp for each image frame
    imageFrameTS = []
    for i in range(1,len(read)):
        if (read[i-1] < 3.0) and (read[i] >= 3.0):
            imageFrameTS.append(i*(1./sf))

    if len(imageFrameTS) < imageFrameNum:
        raise LookupError("Expose period number is smaller than image frame number!")
    imageFrameTS = imageFrameTS[0:imageFrameNum]

    # first time of visual stimulation
    visualStart = None

    for i in range(80,len(photodiode)):
        if ((photodiode[i] - photodiodeThr) * (photodiode[i-1] - photodiodeThr)) < 0 and \
           ((photodiode[i] - photodiodeThr) * (photodiode[i-75] - photodiodeThr)) < 0: #first frame of big change
                visualStart = i*(1./sf)
                break

    return np.array(imageFrameTS), visualStart


def getLog(logPath):
    '''
    get log dictionary from a specific path (including file names)
    '''

    f = open(logPath,'r')
    displayLog = pickle.load(f)
    f.close()
    return displayLog


def generateAVI(saveFolder,
                fileName,
                matrix,
                frameRate=25.,
                encoder='XVID',
                zoom=1,
                isDisplay=True
                ):

    '''
    :param saveFolder:
    :param fileName: can be with '.avi' or without '.avi'
    :param matrix: can be 3 dimensional (gray value) or 4 dimensional
                   if the length of the 4th dimension equals 3, it will be considered as rgb
                   if the length of the 4th dimension equals 4, it will be considered as rgba
    :param frameRate:
    :param encoder:
    :param zoom:
    :return: generate the .avi movie file
    '''

    if len(matrix.shape) == 4:
        if matrix.shape[3] == 3:
            r, g, b = np.rollaxis(matrix, axis = -1)
        elif matrix.shape[3] == 4:
            r, g, b, a = np.rollaxis(matrix, axis = -1)
        else: raise IndexError('The depth of matrix is not 3 or 4. Can not get RGB color!')
        r = r.reshape(r.shape[0],r.shape[1],r.shape[2],1)
        g = g.reshape(g.shape[0],g.shape[1],g.shape[2],1)
        b = b.reshape(b.shape[0],b.shape[1],b.shape[2],1)
        newMatix = np.concatenate((r,g,b),axis=3)
        newMatrix = (ia.array_nor(newMatix) * 255).astype(np.uint8)
    elif len(matrix.shape) == 3:
        s = (ia.array_nor(matrix) * 255).astype(np.uint8)
        s = s.reshape(s.shape[0],s.shape[1],s.shape[2],1)
        newMatrix = np.concatenate((s,s,s),axis=3)
    else: raise IndexError('The matrix dimension is neither 3 or 4. Can not get RGB color!')


    fourcc = cv2.cv.CV_FOURCC(*encoder)

    if fileName[-4:] != '.avi':
        fileName += '.avi'

    size = (int(newMatrix.shape[1]*zoom),int(newMatrix.shape[2]*zoom))

    filePath = os.path.join(saveFolder,fileName+'.avi')
    out = cv2.VideoWriter(filePath,fourcc, frameRate, size)

    for i in range(newMatrix.shape[0]):
        out.write(newMatrix[i,:,:,:])
        if isDisplay:
            cv2.imshow('movie',newMatrix[i,:,:,:])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()


def importRawJCamF(path,
                   saveFolder = None,
                   dtype = np.dtype('<u2'),
                   headerLength = 116,
                   tailerLength = 218,
                   column = 2048,
                   row = 2048,
                   frame = None, #how many frame to read
                   crop = None):

    if frame is not None:
        data = np.fromfile(path,dtype=dtype,count=frame*column*row+headerLength)
        header = data[0:headerLength]
        tailer = []
        mov = data[headerLength:].reshape((frame,column,row))
    else:
        data = np.fromfile(path,dtype=dtype)
        header = data[0:headerLength]
        tailer = data[len(data)-tailerLength:len(data)]
        frame = int((len(data)-headerLength-tailerLength)/(column*row))
        # print(len(data[headerLength:len(data)-tailerLength]))
        mov = data[headerLength:len(data)-tailerLength].reshape((frame,column,row))

    if saveFolder:
        if crop:
            try:
                mov = mov[:,crop[0]:crop[1],crop[2]:crop[3]]
            except Exception as e:
                print('importRawJCamF: Can not understant the paramenter "crop":'+str(crop)+'\ncorp should be: [rowStart,rowEnd,colStart,colEnd]')
                print('\nTrace back: \n' + e)

        fileName = os.path.splitext(os.path.split(path)[-1])[0] + '.tif'
        tf.imsave(os.path.join(saveFolder,fileName),mov)

    return mov, header, tailer


def int2str(num,length=None):
    '''
    generate a string representation for a integer with a given length
    :param num: non-negative int, input number
    :param length: positive int, length of the string
    :return: string represetation of the integer
    '''

    rawstr = str(int(num))
    if length is None or length == len(rawstr):return rawstr
    elif length < len(rawstr): raise ValueError('Length of the number is longer then defined display length!')
    elif length > len(rawstr): return '0'*(length-len(rawstr)) + rawstr


def imageToHdf5(array_like, save_path, hdf5_path, spatial_zoom=None, chunk_size=1000, compression=None):
    """
    save a array_like object (hdf5 dataset, BinarySlicer object, np.array, etc) into a hdf5 file

    :param array_like: 3-d, array_like object (hdf5 dataset, BinarySlicer object, np.array, etc), dimension (zyx)
    :param save_path: str, the path of the hdf5 file to be saved
    :param hdf5_path: str, the path of the dataset within the hdf5 file
    :param spatial_zoom: tuple of 2 floats or one float, spatial zoom of y and x
    :param chunk_size: int, the number of frames of each chunk of processing
    :param compression: str, "gzip", "lzf", "szip"
    :return:
    """

    original_shape = array_like.shape
    original_dtype = array_like.dtype

    print('\ntransforming image array to hdf5 format. \noriginal shape: ' + str(original_shape))

    if len(original_shape) != 3:
        raise ValueError('the array_like should be 3-d!')

    if spatial_zoom is not None:
        try:
            zoom = np.array([spatial_zoom[0], spatial_zoom[1]])
        except TypeError:
            zoom = np.array([spatial_zoom, spatial_zoom])

        new_shape = (np.array(original_shape)[1:3] * zoom).astype(np.int)
        new_shape = (original_shape[0], new_shape[0], new_shape[1])
    else:
        new_shape = original_shape

    print('shape after transformation: ' + str(new_shape))

    save_file = h5py.File(save_path)
    if compression is not None:
        dset = save_file.create_dataset(hdf5_path, new_shape, dtype=original_dtype, compression=compression)
    else:
        dset = save_file.create_dataset(hdf5_path, new_shape, dtype=original_dtype)

    chunk_num = original_shape[0] // chunk_size

    for i in range(chunk_num):
        chunk_start = i * chunk_size
        chunk_end = (i + 1) * chunk_size
        print('transforming chunk: [' + str(chunk_start) + ':' + str(chunk_end) + '] ...')
        curr_chunk =  array_like[chunk_start : chunk_end, :, :]
        if spatial_zoom is not None:
            curr_chunk = ia.rigid_transform_cv2(curr_chunk, zoom=zoom).astype(original_dtype)
        dset[i * chunk_size : (i + 1) * chunk_size, :, :] = curr_chunk


    if chunk_num % chunk_size != 0:
        print('transforming chunk: [' + str(chunk_size * chunk_num) + ':' + str(original_shape[0]) + '] ...')
        last_chunk = array_like[chunk_size * chunk_num:, :, :]
        if spatial_zoom is not None:
            last_chunk = ia.rigid_transform_cv2(last_chunk, zoom=zoom).astype(original_dtype)
        dset[chunk_size * chunk_num:, :, :] = last_chunk

    save_file.close()


def update_key(group, dataset_name, dataset_data, is_overwrite=True):
    '''
    check if a dataset exists in a h5file group. if not create this dataset in group with dataset_name and dataset_data,
    if yes: do nothing or overwrite

    :param group: h5file group instance
    :param dataset_name: str
    :param dataset_data: data type can be used as h5file dataset
    :param is_overwrite: bool, if True, automatically overwrite
                               if False, ask for manual confirmation for overwriting.
    '''
    if dataset_name not in list(group.keys()):
        group.create_dataset(dataset_name, data=dataset_data)
    else:
        if is_overwrite:
            print('overwriting dataset "' + dataset_name + '" in group "' + str(group) + '".')
            del group[dataset_name]
            group.create_dataset(dataset_name, data=dataset_data)
        else:
            check = ''
            while check != 'y' and check != 'n':
                check = input(dataset_name + ' already exists in group ' + str(group) + '. Overwrite? (y/n)\n')
                if check == 'y':
                    del group[dataset_name]
                    group.create_dataset(dataset_name, data=dataset_data)
                elif check == 'n':
                    pass


def write_dictionary_to_h5group_recursively(target, source, is_overwrite=True):
    """
    add a dictionary to an h5py object as target. if target is h5py.Dataset instance, this dataset will be
    deleted and a group with same name will be created. if tarte is h5py.Group instance, the (key: value)
    pairs in the source will be added into this group recursively
    :param target: h5py.Dataset or h5py.Group object as target to save data
    :param source: dictionary, all the values should be instances that can be saved into hdf5 file
    :param is_overwrite: bool, if True, automatically overwrite
                               if False, ask for manual confirmation for overwriting.
    :return: nothing
    """

    if not isinstance(source, dict):
        raise TypeError('source should be a dictionary.')

    if isinstance(target, h5py.Dataset):
        parent = target.parent
        name = target.name

        if is_overwrite:
            del parent[name]
            curr_group = parent.create_group(name)
            write_dictionary_to_h5group_recursively(curr_group, source, is_overwrite=True)
        else:
            check = ''
            while check != 'y' and check != 'n':
                check = input(name + ' already exists in group ' + str(parent) + '. Overwrite? (y/n)\n')
                if check == 'y':
                    del parent[name]
                    curr_group = parent.create_group(name)
                    write_dictionary_to_h5group_recursively(curr_group, source, is_overwrite=True)
                elif check == 'n':
                    pass

    elif isinstance(target, h5py.Group):
        for key, value in list(source.items()):
            if key not in list(target.keys()):
                if isinstance(value, dict):
                    curr_group = target.create_group(key)
                    write_dictionary_to_h5group_recursively(curr_group, value, is_overwrite=is_overwrite)
                else:
                    target.create_dataset(key, data=value)
            else:
                if isinstance(value, dict):
                    write_dictionary_to_h5group_recursively(target[key], value, is_overwrite=is_overwrite)
                else:
                    update_key(target, key, value, is_overwrite=is_overwrite)
    else:
        raise TypeError('target: "' + target.name + '" should be either h5py.Dataset or h5py.Group classes.')


def read_sync(f_path, analog_downsample_rate=None, by_label=True, digital_labels=None,
              analog_labels=None):
    """
    convert sync output to a dictionary

    :param f_path: path to the sync output .h5 file
    :param analog_downsample_rate: int, temporal downsample factor for analog channel
    :param by_label: bool, if True: only extract channels with string labels
                           if False: extract all saved channels by indices
    :param digital_labels: list of strings,
                           selected labels for extracting digital channels. Overwrites
                           'by_label' for digital channel. Use this only if you know what
                           you are doing.
    :param analog_labels: list of strings,
                          selected labels for extracting analog channels. Overwrites
                          'by_label' for analog channel. Use this only if you know what
                          you are doing.
    :return: sync_dict: {'digital_channels': {'rise': rise_ts (in seconds),
                                              'fall': fall_ts (in seconds)},
                         'analog_channels': analog_traces,
                         'analog_sample_rate': analog_fs (float)}
    """

    ds = SyncDataset(path=f_path)

    # print(ds.meta_data)
    # print(ds.analog_meta_data)

    # read digital channels
    digital_channels = {}

    if digital_labels is not None:
        digital_cns = digital_labels
    elif by_label:
        digital_cns = ds.meta_data['line_labels']
    else:
        digital_cns = [dl for dl in ds.meta_data['line_labels'] if dl]
        if len(digital_cns) > 0:
            warnings.warn('You choose to extract digital channels by index. But there are '
                          'digital channels with string labels: {}. All the string labels '
                          'will be lost.'.format(str(digital_cns)))
        digital_cns = range(ds.meta_data['ni_daq']['event_bits'])
        digital_cns = [str(cn) for cn in digital_cns]

    # print(digital_cns)

    for digital_i, digital_cn in enumerate(digital_cns):
        if digital_cn:
            digital_channels[digital_cn] = {'rise': ds.get_rising_edges(line=digital_i, units='seconds'),
                                            'fall': ds.get_falling_edges(line=digital_i, units='seconds')}

    # read analog channels
    data_f = h5py.File(f_path, 'r')
    if 'analog_meta' not in data_f.keys():
        data_f.close()
        print ('no analog data found in file: {}.'.format(f_path))
        return {'digital_channels': digital_channels}
    else:

        analog_channels = {}

        if analog_downsample_rate is None:
            analog_downsample_rate = 1
        analog_fs = ds.analog_meta_data['analog_sample_rate'] / analog_downsample_rate
        if analog_labels is not None:
            analog_cns = analog_labels
            for analog_cn in analog_cns:
                analog_channels[str(analog_cn)] = ds.get_analog_channel(channel=analog_cn,
                                                                        downsample=analog_downsample_rate)
        elif by_label:
            analog_cns = [al for al in ds.analog_meta_data['analog_labels'] if al]

            for analog_cn in analog_cns:
                analog_channels[str(analog_cn)] = ds.get_analog_channel(channel=analog_cn,
                                                                        downsample=analog_downsample_rate)
        else:
            analog_cns = [al for al in ds.analog_meta_data['analog_labels'] if al]
            if len(analog_cns) > 0:
                warnings.warn('You choose to extract analog channels by index. But there are '
                              'analog channels with string labels: {}. All the string labels '
                              'will be lost.'.format(str(digital_cns)))
            analog_cns = ds.analog_meta_data['analog_channels']

            for analog_ind, analog_cn in enumerate(analog_cns):
                analog_channels[str(analog_cn)] = ds.get_analog_channel(channel=analog_ind,
                                                                        downsample=analog_downsample_rate)

        return {'digital_channels': digital_channels,
                'analog_channels': analog_channels,
                'analog_sample_rate': analog_fs}


def look_for_unique_file(source, identifiers, file_type=None, print_prefix='', is_full_path=False,
                         is_verbose=True):

    fns = look_for_file_list(source=source,
                             identifiers=identifiers,
                             file_type=file_type,
                             is_full_path=is_full_path)

    if len(fns) == 0:
        if is_verbose:
            print('{}Did not find file. Returning None.'.format(print_prefix))
        return
    elif len(fns) > 1:
        if is_verbose:
            print('{}Found more than one files. Returning None.'.format(print_prefix))
        return
    else:
        return fns[0]


def look_for_file_list(source, identifiers, file_type=None, is_full_path=False):

    if file_type is not None:
        ft_len = len(file_type)
        fns = [fn for fn in os.listdir(source) if len(fn) >= ft_len and
               fn[-ft_len:] == file_type]
    else:
        fns = [fn for fn in os.listdir(source)]

    for identifier in identifiers:
        fns = [fn for fn in fns if identifier in fn]

    fns.sort()

    if is_full_path:
        fns = [os.path.abspath(os.path.join(source, f)) for f in fns]

    return fns


class SyncDataset(object):
    """
    A sync dataset.  Contains methods for loading
        and parsing the binary data.

    Parameters
    ----------
    path : str
        Path to HDF5 file.

    Examples
    --------
    >>> dset = Dataset('my_h5_file.h5')
    >>> print(dset.meta_data)
    >>> dset.stats()
    >>> dset.close()

    >>> with Dataset('my_h5_file.h5') as dset:
    ...     print(dset.meta_data)
    ...     dset.stats()

    """

    def __init__(self, path):
        self.dfile = self.load(path)

    def _process_times(self):
        """
        Preprocesses the time array to account for rollovers.
            This is only relevant for event-based sampling.

        """
        times = self.get_all_events()[:, 0:1].astype(np.int64)

        intervals = np.ediff1d(times, to_begin=0)
        rollovers = np.where(intervals < 0)[0]

        for i in rollovers:
            times[i:] += 4294967296

        return times

    def load(self, path):
        """
        Loads an hdf5 sync dataset.

        Parameters
        ----------
        path : str
            Path to hdf5 file.

        """
        self.dfile = h5py.File(path, 'r')
        self.meta_data = eval(self.dfile['meta'][()])
        self.line_labels = self.meta_data['line_labels']
        self.times = self._process_times()
        return self.dfile

    @property
    def sample_freq(self):
        try:
            return float(self.meta_data['ni_daq']['sample_freq'])
        except KeyError:
            return float(self.meta_data['ni_daq']['counter_output_freq'])

    def get_bit(self, bit):
        """
        Returns the values for a specific bit.

        Parameters
        ----------
        bit : int
            Bit to return.
        """
        return get_bit(self.get_all_bits(), bit)

    def get_line(self, line):
        """
        Returns the values for a specific line.

        Parameters
        ----------
        line : str
            Line to return.

        """
        bit = self._line_to_bit(line)
        return self.get_bit(bit)

    def get_bit_changes(self, bit):
        """
        Returns the first derivative of a specific bit.
            Data points are 1 on rising edges and 255 on falling edges.

        Parameters
        ----------
        bit : int
            Bit for which to return changes.

        """
        bit_array = self.get_bit(bit)
        return np.ediff1d(bit_array, to_begin=0)

    def get_line_changes(self, line):
        """
        Returns the first derivative of a specific line.
            Data points are 1 on rising edges and 255 on falling edges.

        Parameters
        ----------
        line : (str)
            Line name for which to return changes.

        """
        bit = self._line_to_bit(line)
        return self.get_bit_changes(bit)

    def get_all_bits(self):
        """
        Returns the data for all bits.

        """
        return self.dfile['data'][:, -1]

    def get_all_times(self, units='samples'):
        """
        Returns all counter values.

        Parameters
        ----------
        units : str
            Return times in 'samples' or 'seconds'

        """
        if self.meta_data['ni_daq']['counter_bits'] == 32:
            times = self.get_all_events()[:, 0]
        else:
            times = self.times
        units = units.lower()
        if units == 'samples':
            return times
        elif units in ['seconds', 'sec', 'secs']:
            freq = self.sample_freq
            return times / freq
        else:
            raise ValueError("Only 'samples' or 'seconds' are valid units.")

    def get_all_events(self):
        """
        Returns all counter values and their cooresponding IO state.
        """
        return self.dfile['data'][()]

    def get_events_by_bit(self, bit, units='samples'):
        """
        Returns all counter values for transitions (both rising and falling)
            for a specific bit.

        Parameters
        ----------
        bit : int
            Bit for which to return events.

        """
        changes = self.get_bit_changes(bit)
        return self.get_all_times(units)[np.where(changes != 0)]

    def get_events_by_line(self, line, units='samples'):
        """
        Returns all counter values for transitions (both rising and falling)
            for a specific line.

        Parameters
        ----------
        line : str
            Line for which to return events.

        """
        line = self._line_to_bit(line)
        return self.get_events_by_bit(line, units)

    def _line_to_bit(self, line):
        """
        Returns the bit for a specified line.  Either line name and number is
            accepted.

        Parameters
        ----------
        line : str
            Line name for which to return corresponding bit.

        """
        if type(line) is int:
            return line
        elif type(line) is str:
            return self.line_labels.index(line)
        else:
            raise TypeError("Incorrect line type.  Try a str or int.")

    def _bit_to_line(self, bit):
        """
        Returns the line name for a specified bit.

        Parameters
        ----------
        bit : int
            Bit for which to return the corresponding line name.
        """
        return self.line_labels[bit]

    def get_rising_edges(self, line, units='samples'):
        """
        Returns the counter values for the rizing edges for a specific bit or
            line.

        Parameters
        ----------
        line : str
            Line for which to return edges.

        """
        bit = self._line_to_bit(line)
        changes = self.get_bit_changes(bit)
        return self.get_all_times(units)[np.where(changes == 1)]

    def get_falling_edges(self, line, units='samples'):
        """
        Returns the counter values for the falling edges for a specific bit
            or line.

        Parameters
        ----------
        line : str
            Line for which to return edges.

        """
        bit = self._line_to_bit(line)
        changes = self.get_bit_changes(bit)
        return self.get_all_times(units)[np.where(changes == 255)]

    def get_nearest(self,
                    source,
                    target,
                    source_edge="rising",
                    target_edge="rising",
                    direction="previous",
                    units='indices',
                    ):
        """
        For all values of the source line, finds the nearest edge from the
            target line.

        By default, returns the indices of the target edges.

        Args:
            source (str, int): desired source line
            target (str, int): desired target line
            source_edge [Optional(str)]: "rising" or "falling" source edges
            target_edge [Optional(str): "rising" or "falling" target edges
            direction (str): "previous" or "next". Whether to prefer the
                previous edge or the following edge.
            units (str): "indices"

        """
        source_edges = getattr(self,
                               "get_{}_edges".format(source_edge.lower()))(source.lower(), units="samples")
        target_edges = getattr(self,
                               "get_{}_edges".format(target_edge.lower()))(target.lower(), units="samples")
        indices = np.searchsorted(target_edges, source_edges, side="right")
        if direction.lower() == "previous":
            indices[np.where(indices != 0)] -= 1
        elif direction.lower() == "next":
            indices[np.where(indices == len(target_edges))] = -1
        if units in ["indices", 'index']:
            return indices
        elif units == "samples":
            return target_edges[indices]
        elif units in ['sec', 'seconds', 'second']:
            return target_edges[indices] / self.sample_freq
        else:
            raise KeyError("Invalid units.  Try 'seconds', 'samples' or 'indices'")

    def get_analog_channel(self,
                           channel,
                           start_time=0.0,
                           stop_time=None,
                           downsample=1):
        """
        Returns the data from the specified analog channel between the
            timepoints.

        Args:
            channel (int, str): desired channel index or label
            start_time (Optional[float]): start time in seconds
            stop_time (Optional[float]): stop time in seconds
            downsample (Optional[int]): downsample factor

        Returns:
            ndarray: slice of data for specified channel

        Raises:
            KeyError: no analog data present

        """
        if isinstance(channel, str):
            channel_index = self.analog_meta_data['analog_labels'].index(channel)
            channel = self.analog_meta_data['analog_channels'].index(channel_index)

        if "analog_data" in self.dfile.keys():
            dset = self.dfile['analog_data']
            analog_meta = self.get_analog_meta()
            sample_rate = analog_meta['analog_sample_rate']
            start = int(start_time * sample_rate)
            if stop_time:
                stop = int(stop_time * sample_rate)
                return dset[start:stop:downsample, channel]
            else:
                return dset[start::downsample, channel]
        else:
            raise KeyError("No analog data was saved.")

    def get_analog_meta(self):
        """
        Returns the metadata for the analog data.
        """
        if "analog_meta" in self.dfile.keys():
            return eval(self.dfile['analog_meta'][()])
        else:
            raise KeyError("No analog data was saved.")

    @property
    def analog_meta_data(self):
        return self.get_analog_meta()

    def line_stats(self, line, print_results=True):
        """
        Quick-and-dirty analysis of a bit.

        """
        # convert to bit
        bit = self._line_to_bit(line)

        # get the bit's data
        bit_data = self.get_bit(bit)
        total_data_points = len(bit_data)

        # get the events
        events = self.get_events_by_bit(bit)
        total_events = len(events)

        # get the rising edges
        rising = self.get_rising_edges(bit)
        total_rising = len(rising)

        # get falling edges
        falling = self.get_falling_edges(bit)
        total_falling = len(falling)

        if total_events <= 0:
            if print_results:
                print("*" * 70)
                print("No events on line: %s" % line)
                print("*" * 70)
            return None
        elif total_events <= 10:
            if print_results:
                print("*" * 70)
                print("Sparse events on line: %s" % line)
                print("Rising: %s" % total_rising)
                print("Falling: %s" % total_falling)
                print("*" * 70)
            return {
                'line': line,
                'bit': bit,
                'total_rising': total_rising,
                'total_falling': total_falling,
                'avg_freq': None,
                'duty_cycle': None,
            }
        else:

            # period
            period = self.period(line)

            avg_period = period['avg']
            max_period = period['max']
            min_period = period['min']
            period_sd = period['sd']

            # freq
            avg_freq = self.frequency(line)

            # duty cycle
            duty_cycle = self.duty_cycle(line)

            if print_results:
                print("*" * 70)

                print("Quick stats for line: %s" % line)
                print("Bit: %i" % bit)
                print("Data points: %i" % total_data_points)
                print("Total transitions: %i" % total_events)
                print("Rising edges: %i" % total_rising)
                print("Falling edges: %i" % total_falling)
                print("Average period: %s" % avg_period)
                print("Minimum period: %s" % min_period)
                print("Max period: %s" % max_period)
                print("Period SD: %s" % period_sd)
                print("Average freq: %s" % avg_freq)
                print("Duty cycle: %s" % duty_cycle)

                print("*" * 70)

            return {
                'line': line,
                'bit': bit,
                'total_data_points': total_data_points,
                'total_events': total_events,
                'total_rising': total_rising,
                'total_falling': total_falling,
                'avg_period': avg_period,
                'min_period': min_period,
                'max_period': max_period,
                'period_sd': period_sd,
                'avg_freq': avg_freq,
                'duty_cycle': duty_cycle,
            }

    def period(self, line, edge="rising"):
        """
        Returns a dictionary with avg, min, max, and st of period for a line.
        """
        bit = self._line_to_bit(line)

        if edge.lower() == "rising":
            edges = self.get_rising_edges(bit)
        elif edge.lower() == "falling":
            edges = self.get_falling_edges(bit)

        if len(edges) > 2:

            timebase_freq = self.meta_data['ni_daq']['counter_output_freq']
            avg_period = np.mean(np.ediff1d(edges[1:])) / timebase_freq
            max_period = np.max(np.ediff1d(edges[1:])) / timebase_freq
            min_period = np.min(np.ediff1d(edges[1:])) / timebase_freq
            period_sd = np.std(avg_period)

        else:
            raise IndexError("Not enough edges for period: %i" % len(edges))

        return {
            'avg': avg_period,
            'max': max_period,
            'min': min_period,
            'sd': period_sd,
        }

    def frequency(self, line, edge="rising"):
        """
        Returns the average frequency of a line.
        """

        period = self.period(line, edge)
        return 1.0 / period['avg']

    def duty_cycle(self, line):
        """
        Doesn't work right now.  Freezes python for some reason.

        Returns the duty cycle of a line.

        """
        return "fix me"
        bit = self._line_to_bit(line)

        rising = self.get_rising_edges(bit)
        falling = self.get_falling_edges(bit)

        total_rising = len(rising)
        total_falling = len(falling)

        if total_rising > total_falling:
            rising = rising[:total_falling]
        elif total_rising < total_falling:
            falling = falling[:total_rising]
        else:
            pass

        if rising[0] < falling[0]:
            # line starts low
            high = falling - rising
        else:
            # line starts high
            high = np.concatenate(falling, self.get_all_events()[-1, 0]) - \
                   np.concatenate(0, rising)

        total_high_time = np.sum(high)
        all_events = self.get_events_by_bit(bit)
        total_time = all_events[-1] - all_events[0]
        return 1.0 * total_high_time / total_time

    def stats(self):
        """
        Quick-and-dirty analysis of all bits.  Prints a few things about each
            bit where events are found.
        """
        bits = []
        for i in range(32):
            bits.append(self.line_stats(i, print_results=False))
        active_bits = [x for x in bits if x is not None]
        print("Active bits: ", len(active_bits))
        for bit in active_bits:
            print("*" * 70)
            print("Bit: %i" % bit['bit'])
            print("Label: %s" % self.line_labels[bit['bit']])
            print("Rising edges: %i" % bit['total_rising'])
            print("Falling edges: %i" % bit["total_falling"])
            print("Average freq: %s" % bit['avg_freq'])
            print("Duty cycle: %s" % bit['duty_cycle'])
        print("*" * 70)
        return active_bits

    def plot_all(self,
                 start_time,
                 stop_time,
                 auto_show=True,
                 ):
        """
        Plot all active bits.

        Yikes.  Come up with a better way to show this.

        """
        import matplotlib.pyplot as plt
        for bit in range(32):
            if len(self.get_events_by_bit(bit)) > 0:
                self.plot_bit(bit,
                              start_time,
                              stop_time,
                              auto_show=False, )
        if auto_show:
            plt.show()

    def plot_bits(self,
                  bits,
                  start_time=0.0,
                  end_time=None,
                  auto_show=True,
                  ):
        """
        Plots a list of bits.
        """
        import matplotlib.pyplot as plt

        subplots = len(bits)
        f, axes = plt.subplots(subplots, sharex=True, sharey=True)
        if not isinstance(axes, collections.Iterable):
            axes = [axes]

        for bit, ax in zip(bits, axes):
            self.plot_bit(bit,
                          start_time,
                          end_time,
                          auto_show=False,
                          axes=ax)
        # f.set_size_inches(18, 10, forward=True)
        f.subplots_adjust(hspace=0)

        if auto_show:
            plt.show()

    def plot_bit(self,
                 bit,
                 start_time=0.0,
                 end_time=None,
                 auto_show=True,
                 axes=None,
                 name="",
                 ):
        """
        Plots a specific bit at a specific time period.
        """
        import matplotlib.pyplot as plt

        times = self.get_all_times(units='sec')
        if not end_time:
            end_time = 2 ** 32

        window = (times < end_time) & (times > start_time)

        if axes:
            ax = axes
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if not name:
            name = self._bit_to_line(bit)
        if not name:
            name = str(bit)

        bit = self.get_bit(bit)
        ax.step(times[window], bit[window], where='post')
        ax.set_ylim(-0.1, 1.1)
        # ax.set_ylabel('Logic State')
        ax.yaxis.set_ticks_position('none')
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xlabel('time (seconds)')
        ax.legend([name])

        if auto_show:
            plt.show()

        return plt.gcf()

    def plot_line(self,
                  line,
                  start_time=0.0,
                  end_time=None,
                  auto_show=True,
                  ):
        """
        Plots a specific line at a specific time period.
        """
        import matplotlib.pyplot as plt
        bit = self._line_to_bit(line)
        self.plot_bit(bit, start_time, end_time, auto_show=False)

        # plt.legend([line])
        if auto_show:
            plt.show()

        return plt.gcf()

    def plot_lines(self,
                   lines,
                   start_time=0.0,
                   end_time=None,
                   auto_show=True,
                   ):
        """
        Plots specific lines at a specific time period.
        """
        import matplotlib.pyplot as plt
        bits = []
        for line in lines:
            bits.append(self._line_to_bit(line))
        self.plot_bits(bits,
                       start_time,
                       end_time,
                       auto_show=False, )

        plt.subplots_adjust(left=0.025, right=0.975, bottom=0.05, top=0.95)
        if auto_show:
            plt.show()

        return plt.gcf()

    def close(self):
        """
        Closes the dataset.
        """
        self.dfile.close()

    def __enter__(self):
        """
        So we can use context manager (with...as) like any other open file.

        Examples
        --------
        >>> with Dataset('my_data.h5') as d:
        ...     d.stats()

        """
        return self

    def __exit__(self, type, value, traceback):
        """
        Exit statement for context manager.
        """
        self.close()


if __name__=='__main__':

    # ----------------------------------------------------------------------------
    sync_path = r"D:\data2\rabies_tracing_project\method_development" \
                r"\2017-10-05-read-sync\171003_M345521_FlashingCircle_106_171003165755.h5"
    # sync_path = r"\\allen\programs\braintv\workgroups\nc-ophys\ImageData\Soumya\trees\m255" \
    #             r"\log_m255\sync_pkl\m255_presynaptic_pop_vol1_stimDG_bessel170918013215.h5"
    sync_dict = read_sync(f_path=sync_path, by_label=False, digital_labels=['vsync_2p'],
                          analog_labels=['photodiode'], analog_downsample_rate=None)
    # sync_dict = read_sync(f_path=sync_path, by_label=False, digital_labels=None,
    #                       analog_labels=None, analog_downsample_rate=None)
    # print(sync_dict)
    # ----------------------------------------------------------------------------

    #----------------------------------------------------------------------------
    # mov = np.random.rand(250,512,512,4)
    # generateAVI(r'C:\JunZhuang\labwork\data\python_temp_folder','tempMov',mov)
    #----------------------------------------------------------------------------
    # print(int2str(5))
    # print(int2str(5,2))
    # print(int2str(155,6))
    #----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # imageToHdf5(np.random.rand(10, 100, 100), save_path=r'E:\data\python_temp_folder\test_file.hdf5',
    #             hdf5_path='/data', spatial_zoom=(0.5, 0.2))
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # ff = h5py.File(r"E:\data\python_temp_folder\test4.hdf5")
    # test_dict = {'a':1, 'b':2, 'c': {'A': 4, 'B': 5}}
    # write_dictionary_to_h5group_recursively(target=ff, source=test_dict, is_overwrite=True)
    # ff.close()
    #
    # ff = h5py.File(r"E:\data\python_temp_folder\test4.hdf5")
    # test_dict2 = {'a': {'C': 6, 'D': 7}, 'c': {'A': 4, 'B': 6}, 'd':10, 'e':{'E':11, 'F':'xx'}}
    # write_dictionary_to_h5group_recursively(target=ff, source=test_dict2, is_overwrite=False)
    # ff.close()
    # ----------------------------------------------------------------------------


    print('well done!')