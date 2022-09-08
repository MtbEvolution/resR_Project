import PIL as pl
import numpy as np
import nd2reader as nd2
import tifffile as tf
import os,glob,pickle as pk

__all__ = ['sort2folder',
           'softread_nd2file',
           'softread_tiffile',
           'softread_file',
           'get_slice_by_index',
           'get_slice_by_index',
           'load_softread']

def sort2folder(src_folder):
    files = sorted(glob.glob(src_folder + '*.nd2'))
    for f in files:
        header = f.split('/')[-1]
        well = header.split('_')[1][4:]
        subfolder = '{}{}/'.format(src_folder, well)
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)
        os.rename(f, '{}{}'.format(subfolder, header))
        

def ImageJinfo2dict(ImageJinfo):
    """
    Extract metadata from imageJ processsed .tif files
    :param ImageJinfo: imagej converted metadata
    :return: dictionary of metadata
    """
    if "\n" in ImageJinfo:
        ImageJinfo = ImageJinfo.split("\n")
    ImageJdict = {}
    for info in ImageJinfo:
        info = info.split(" = ")
        if len(info) == 2:
            key = info[0]
            val = info[1]
            if key.startswith(" "):
                key = key[1:]
            if val.startswith(" "):
                val = val[1:]
            ImageJdict[key] = val
    return ImageJdict


def softread_nd2file(nd2file, header=None):
    import nd2reader as nd2
    # convert .nd2 file to standard OMEGA input
    # img order: v->t->c
    if not isinstance(nd2file, str):
        raise ValueError('Illegal input!')
    if not nd2file.endswith('.nd2'):
        raise ValueError('Input {} is not a .nd2 file.'.format(nd2file.split('/')[-1]))
    if header is None:
        header = nd2file.split('/')[-1].split('.')[0]
    sorted_data = {}
    img = nd2.ND2Reader(nd2file)
    sizes = img.sizes
    metadata = img.metadata
    shape = (sizes['x'], sizes['y'])
    channels = metadata['channels']
    n_channels = len(channels)
    n_fields = sizes['v'] if 'v' in sizes else 1
    n_time = sizes['t'] if 't' in sizes else 1
    n_zcoords = sizes['z'] if 'z' in sizes else 1
    sorted_data['format'] = 'nd2'
    sorted_data['header'] = header
    sorted_data['metadata'] = metadata
    sorted_data['pixel_microns'] = metadata['pixel_microns']
    sorted_data['ND2_object'] = img
    sorted_data['channels'] = channels
    sorted_data['n_fields'] = n_fields
    sorted_data['n_channels'] = n_channels
    sorted_data['n_timepoints'] = n_time
    sorted_data['n_zpositions'] = n_zcoords
    sorted_data['shape'] = shape
    return sorted_data


def softread_tiffile(tif, header=None, channels=None, pixel_microns=0.065):
    import tifffile as tf
    # convert .tif file to standard OMEGA input
    # only supports single view, single/multi- channel tif data
    if not isinstance(tif, str):
        raise ValueError('Illegal input!')
    if not tif.endswith('.tif'):
        raise ValueError('Input {} is not a .tif file.'.format(tif.split('/')[-1]))
    if header is None:
        header = tif.split('/')[-1].split('.')[0]
    sorted_data = {}
    img = tf.TiffFile(tif)
    series = img.series[0]
    if len(series.shape) == 2:
        x, y = series.shape
        n_channels = 1
    elif len(series.shape) == 3:
        n_channels, x, y = series.shape
    else:
        raise ValueError('Hyperstack not supported!')
    if 'Info' in img.imagej_metadata:
        metadata = ImageJinfo2dict(img.imagej_metadata['Info'])
    else:
        metadata = {}
    shape = (x, y)
    if channels is not None:
        if len(channels) != n_channels:
            raise ValueError('{} channels found, {} channel names specified!'.format(n_channels,
                                                                                     len(channels)))
    else:
        channels = ['C{}'.format(i + 1) for i in range(n_channels)]

    sorted_data['format'] = 'tif'
    sorted_data['header'] = header
    sorted_data['metadata'] = metadata
    sorted_data['pixel_microns'] = float(metadata['dCalibration']) if 'dCalibration' in metadata else float(
        pixel_microns)
    sorted_data['TiffFile_object'] = img
    sorted_data['channels'] = channels
    sorted_data['n_fields'] = 1
    sorted_data['n_channels'] = 1
    sorted_data['n_timepoints'] = 1
    sorted_data['n_zpositions'] = 1
    sorted_data['shape'] = shape
    return sorted_data


def softread_file(img_file):
    if img_file.endswith('.nd2'):
        data_dict = softread_nd2file(img_file)
    elif img_file.endswith('.tif'):
        data_dict = softread_tiffile(img_file)
    else:
        raise ValueError('Data format .{} not supported!'.format(img_file.split('.')[-1]))
    return data_dict


def get_slice_by_index(data_dict, channel=0, position=0, time=0, zposition=0):
    sliced_img = np.array([])
    if data_dict['format'] == 'tif':
        sliced_img = data_dict['TiffFile_object'].pages[channel].asarray()
    elif data_dict['format'] == 'nd2':
        sliced_img = np.array(data_dict['ND2_object'].get_frame_2D(c=channel, v=position, t=time, z=zposition))
    return sliced_img


def load_softread(data_dict):
    header = data_dict['header']
    if data_dict['format'] == 'tif':
        reader_obj = data_dict.pop('TiffFile_object', None)
        sorted_data = data_dict.copy()
        img_stack_id = '{}_Time{}_Position{}_Z{}'.format(header, 0, 0, 0)
        sorted_data['volumes'] = {}
        sorted_data['volumes'][img_stack_id] = {c: reader_obj.pages[i].asarray() \
                                                for i, c in enumerate(sorted_data['channels'])}

    elif data_dict['format'] == 'nd2':
        reader_obj = data_dict.pop('ND2_object', None)
        sorted_data = data_dict.copy()
        sorted_data['volumes'] = {}
        for t in range(sorted_data['n_timepoints']):
            for v in range(sorted_data['n_fields']):
                for z in range(sorted_data['n_zpositions']):
                    img_stack_id = '{}_Time{}_Position{}_Z{}'.format(header, t, v, z)
                    sorted_data['volumes'][img_stack_id] = {c: np.array(reader_obj.get_frame_2D(t=t, v=v, c=i, z=z))\
                                                            for i, c in enumerate(sorted_data['channels'])}
    return sorted_data


"""
def folder2image_nd2(src_folder,
                     config,
                     ref_channel=-1):
    image_list = []
    files = sorted(glob.glob(src_folder+'*.nd2'))
    for file in files:
        img_dict = load_softread(softread_file(file))
        for key in img_dict['volumes'].keys():
            img_obj = Image()
            img_obj.load_data(image_id=key,
                              image_dict=img_dict['volumes'][key],
                              config = config,
                              ref_channel=ref_channel)
            image_list.append(img_obj)
    return image_list


def unit_func(img_obj, dst_folder, classifier):
    img_obj._autoRun(classifier)
    img_obj.plotSegmentation(dst_folder)
    img_obj.dump(dst_folder)

def process_nd2list(img_objects, dst_folder, classifier, n_cores=8):
    import multiprocessing as mp
    # setup output folder and locate image data

    if not os.path.isdir(dst_folder):
        os.mkdir(dst_folder)

    # multiprocessing
    counter = 0
    for i in range(0, len(img_objects), n_cores):
        processes = []
        for j in range(0, min(n_cores, len(img_objects) - i)):
            processes.append(mp.Process(target=unit_func,
                                        args=(img_objects[counter], dst_folder, classifier)))
            counter += 1
        for p in processes:
            p.start()
        for p in processes:
            p.join()
            
            
"""