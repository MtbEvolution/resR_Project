import numpy as np
import pandas as pd
from skimage import filters, measure
from skimage.morphology import disk, remove_small_holes, remove_small_objects, binary_opening, binary_dilation


def dimension_scaler(image, axis=(0,3)):
    """
    image used for segmentation is reformated as a four-dimensional tensor with each dimension representing:
    #0 - number of classes
    #1,2 - 2D image
    #3 - number of channels (e.g. this should be 3 for RGB images)
    
    if class labels are provided for a single channel image, the axis paramter should be (3).
    if a multi-channel image is provided with no class labesl, the axis paramter should be (0).
    """
    
    if len(image.shape)==4:
        return image
    elif len(image.shape)==3 and len(axis)==1:
        return np.expand_dims(image,axis=axis)
    elif len(image.shape)==2 and len(axis)==2:
        return np.expand_dims(image,axis=axis)
    else:
        raise ValueError("Image shape and provided axis information don't match")

        
def mask_custom(img,th1=0,th2='auto',
                return_threshold=False,config=None,):
    if th2 == 'auto':
        th2 = filters.threshold_isodata(img)
    binary = ((img<th2)&(img>th1))*1
    if not return_threshold:
        return binary.astype(int)
    else:
        return binary.astype(int), (th1,th2)


def mask_sauvola(img,
                 dark_foreground = True, 
                 return_threshold = False, 
                 config=None,
                 **kwargs):
    th = filters.threshold_sauvola(img,**kwargs)
    if dark_foreground:
        binary = (img<th)*1
    else:
        binary = (img>th)*1
    if return_threshold:
        return binary.astype(int), th
    else:
        return binary.astype(int)
    
    
def mask_isodata(img,
                 dark_foreground = True, 
                 return_threshold = False, 
                 config=None,
                 **kwargs):
    th = filters.threshold_isodata(img,**kwargs)
    if dark_foreground:
        binary = (img<th)*1
    else:
        binary = (img>th)*1
    if return_threshold:
        return binary.astype(int), th
    else:
        return binary.astype(int)
    
    
def mask_local(img,
               dark_foreground = True, 
               return_threshold = False,
               block_size=15,
               config=None,
               **kwargs):
    th = filters.threshold_local(img,block_size=block_size,**kwargs)
    if dark_foreground:
        binary = (img<th)*1
    else:
        binary = (img>th)*1
    if return_threshold:
        return binary.astype(int), th
    else:
        return binary.astype(int)
    
    
def mask_legacy(img,
                block_sizes=(15,81),
                method='gaussian',
                dilation_selem=disk(1),
                opening_selem=disk(1),
                config=None):
    """
    binarization method used in OMEGA (MOMIA1), deprecated. 
    """
    local_mask1 = mask_local(img,block_sizes[0],method=method)
    local_mask2 = mask_local(img,block_sizes[1],method=method)
    glob_mask = mask_isodata(img)
    glob_mask = binary_dilation(glob_mask, dilation_selem)
    binary = ((local_mask1 + local_mask2) * glob_mask) > 0
    binary = binary_opening(binary, opening_selem)
    return binary.astype(int)


def mask_composite(img, config=None):
    
    # reset configurations when provided
    if config is not None:
        window_size = int(config['segmentation']['sauvola_window_size'])
        k = float(config['segmentation']['sauvola_k'])
        min_particle_size = int(config['segmentation']['min_size'])
        opening = bool(int(config['segmentation']['binary_opening']))
        overlap_threshold = float(config['segmentation']['overlap_threshold'])
        dilation = bool(int(config['segmentation']['binary_dilation']))
    else:
        window_size = 9 # sauvola local threshold window size
        k = 0.05 # sauvola local threshold regulation
        min_particle_size = 10 # minimal particle size included
        opening = False # opening to delink neighboring particles
        overlap_threshold = 0.5 # threshold of overlap between masks rendered by local and global thresholding
        dilation = True
        
    # local threshold
    mask1 = mask_sauvola(img,window_size=window_size,k=k)
    # isodata threshold
    mask2 = mask_isodata(img)
    # composite mask 
    init_mask = remove_small_objects((mask1+mask2)>0,min_size=min_particle_size)
    if opening:
        init_mask = binary_opening(init_mask)
    
    # label composite mask
    labels = measure.label(init_mask*1)
    filtered_labels = labels.copy()
    
    # measure particles, define foreground/background
    region_props = pd.DataFrame(measure.regionprops_table(labels,intensity_image=mask2,properties=['coords','mean_intensity']))
    foreground = (region_props['mean_intensity']>=overlap_threshold)*1
    region_props['foreground']=foreground
    
    # remove background particles
    x,y = np.vstack(region_props[foreground==0]['coords'].values).T
    filtered_labels[x,y]=0
    binary = (filtered_labels>0)*1
    if dilation:
        binary = binary_dilation(binary)
    return binary.astype(int)

def _make_mask(img, method = 3, **kwargs):
    if method == 0:
        return mask_custom(img,**kwargs)
    if method == 1:
        return mask_isodata(img,**kwargs)
    if method == 2:
        return mask_sauvola(img,**kwargs)
    if method == 3:
        return mask_local(img, **kwargs)
    if method == 4:
        return mask_legacy(img, **kwargs)
    if method == 5:
        return mask_composite(img, **kwargs)