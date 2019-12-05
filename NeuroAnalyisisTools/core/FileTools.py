__author__ = 'junz'

import numpy as np
import pickle
import os
import shutil
import struct
import h5py
import warnings
import numbers

try:
    import ImageAnalysis as ia
except (AttributeError, ImportError):
    from . import ImageAnalysis as ia

try:
    import tifffile as tf
except ImportError:
    import skimage.external.tifffile as tf

try: import cv2
except ImportError as e: print('cannot import OpenCV. {}'.format(e))

try: import sync.dataset as sync_dset
except ImportError as e: print('cannot import sync.dataset. {}'.format(e))


def is_integer(var):
    return isinstance(var, numbers.Integral)


def saveFile(path,data):
    f = open(path,'wb')
    pickle.dump(data, f, protocol=2)
    f.close()


def loadFile(path):
    f = open(path,'rb')
    data = pickle.load(f, encoding='bytes')
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

    if frame:
        data = np.fromfile(path,dtype=dtype,count=frame*column*row+headerLength)
        header = data[0:headerLength]
        tailer = []
        mov = data[headerLength:].reshape((frame,column,row))
    else:
        data = np.fromfile(path,dtype=dtype)
        header = data[0:headerLength]
        tailer = data[len(data)-tailerLength:len(data)]
        frame = (len(data)-headerLength-tailerLength)/(column*row)
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

    ds = sync_dset.Dataset(path=f_path)

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