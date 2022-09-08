from skimage import segmentation, feature, filters, morphology
import numpy as np
from .helper_image import *

def patch_segmentation_basic(img,
                             method='isodata',
                             **kwargs):
    """
    primary image segmentation that splits disconnected pixel patches into Clusters

    :param img: input image, usually phase contrast data
    :param shape_indexing_sigma:
    :param min_particle_size:
    :return:
    """
    binary = make_mask(img,method, **kwargs)
    labels = measure.label(binary, connectivity=1)
    regionprop_table = get_regionprop(labels, intensity_image=img)
    return binary, labels, regionprop_table


def patch_segmentation_shapeindex(image,
                                  mask,
                                  shapeindex,
                                  threshold=(0,100),
                                  disk_radius=2,
                                  min_seed_size=10,
                                  watershed_line=True,
                                  ridge=None):
    v1,v2 = threshold
    init_seeds = (shapeindex < v2) & \
                 (shapeindex > v1)
    init_seeds = morphology.opening(init_seeds, morphology.disk(disk_radius)).astype(bool)
    init_seeds = morphology.remove_small_objects(init_seeds, min_size=min_seed_size)
    labeled_seeds = measure.label(init_seeds, connectivity=2)
    if ridge is not None:
        labels = segmentation.watershed(ridge, labeled_seeds,
                                        mask=mask,
                                        connectivity=2,
                                        compactness=100,
                                        watershed_line=watershed_line)
    else:
        labels = segmentation.watershed(image, labeled_seeds,
                                        mask=mask,
                                        connectivity=2,
                                        compactness=100,
                                        watershed_line=watershed_line)
    regionprop_table = get_regionprop(labels, intensity_image=image)
    return labels, regionprop_table


def make_mask(img,
              method='legacy',
              **kwargs):
    if method == 'isodata':
        binary = generate_binary_mask_isodata(img)
    elif method == 'sauvola':
        binary = generate_binary_mask_sauvola(img,
                                              window_size=kwargs['window_size'],
                                              k=kwargs['k'])
    elif method == 'combined':
        binary = generate_binary_mask_combined(img,
                                               window_size=kwargs['window_size'],
                                               k=kwargs['k'],
                                               min_size=kwargs['min_size'])
    elif method == 'legacy':
        binary = generate_binary_mask_legacy(img, block_sizes=kwargs['block_sizes'])
    else:
        raise ValueError('Method {} does not exist!'.format(method))
    binary = morphology.remove_small_objects(binary, min_size=kwargs['min_size']).astype(bool)
    binary = (filters.median(binary, morphology.disk(2)) > 0).astype(int)
    if kwargs['binary_opening']:
        binary = morphology.opening(binary, morphology.disk(1)) * 1
    return binary


def generate_binary_mask_sauvola(img, window_size=55, k=0.2,**kwargs):
    """
    2021.04.02
    generate a binarized reference mask using the Sauvola local thresholding method.
    :param img: input reference image (phase-contrast), dual-bandpass filtered
    :param window_size: local window size
    :param k: parameter k
    :return: binarized mask
    """
    return (img <= filters.threshold_sauvola(img, window_size=window_size, k=k,**kwargs))*1


def generate_binary_mask_isodata(img):
    """
    basic isodata filter
    :param img: input reference image (phase-contrast)
    :return:
    """
    return (img<filters.threshold_isodata(img))*1


def generate_binary_mask_combined(img, window_size=55,k=0.05,
                                  min_size=50,**kwargs):
    """

    :param img:
    :param window_size:
    :param k:
    :param min_size:
    :param kwargs:
    :return:
    """
    mask1 = generate_binary_mask_sauvola(img, window_size=window_size,
                                         k=k,**kwargs)
    mask2 = generate_binary_mask_isodata(img)
    mask = morphology.dilation(morphology.remove_small_objects((mask1 + mask2)>0,
                                                               min_size=min_size))
    return mask


def generate_binary_mask_legacy(img, block_sizes=(15, 81), **kwargs):
    local_mask1 = img < filters.threshold_local(img, block_size=block_sizes[0], method='gaussian')
    local_mask2 = img < filters.threshold_local(img, block_size=block_sizes[1], method='gaussian')

    glob_mask = img < filters.threshold_isodata(img, nbins=256)
    glob_mask = morphology.binary_opening(glob_mask, morphology.disk(1))
    glob_mask = morphology.remove_small_objects(glob_mask, min_size=25)
    glob_mask = morphology.remove_small_holes(glob_mask, area_threshold=25)
    #glob_mask = morphology.binary_dilation(glob_mask, morphology.disk(2))
    binary = ((local_mask1 + local_mask2) * glob_mask) > 0
    binary = morphology.binary_opening(binary, morphology.disk(3))
    return binary


def shape_index_conversion(img, shapeindex_sigma=1):
    """
    Jan J Koenderink and Andrea J van Doorn proposed a simple solution for numeric representation of local
    curvature, here we use an 8-bit converted shape index measure as the basis to infer different parts of a cell body
    :param img: input image
    :param shape_indexing_sigma: sigma for computing Hessian eigenvalues
    :return: shape index converted image (8-bit)
    """
    shapeindex = feature.shape_index(img, sigma=shapeindex_sigma)
    shapeindex = np.nan_to_num(shapeindex, copy=True)
    shapeindex = (exposure.equalize_hist(shapeindex) * 255).astype(np.uint8)
    return shapeindex


def config2selem(config_str):
    selem_list = config_str.split(',')
    if len(selem_list)!=3:
        raise ValueError('Structure element is not recognized!')
    use_selem = bool(int(selem_list[0]))
    if use_selem and selem_list[1] == 'disk':
        return morphology.disk(int(selem_list[2]))
    elif use_selem and selem_list[1] == 'square':
        return morphology.square(int(selem_list[2]))
    elif not use_selem:
        return None