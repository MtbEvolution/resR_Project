import numpy as np
from skimage import morphology, filters, measure


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


def identical_shapes(image_list):
    x, y = 0,0
    shape_matched = True
    for i, img in enumerate(image_list):
        if i == 0:
            x, y = img.shape
        else:
            if x != img.shape[0] or y != img.shape[1]:
                shape_matched=False
    return shape_matched


def value_constrain(value, min_value, max_value):
    return max(min_value, min(value,max_value))


def touching_edge(img_shape,optimized_bbox):
    (rows, columns) = img_shape
    (x1, y1, x2, y2) = optimized_bbox
    if min(x1, y1, rows-x2-1,columns-y2-1) <= 0:
        return True
    else:
        return False
    
def norm_ratio(measure_of_two):
    sorted_v = np.sort(measure_of_two)
    return sorted_v[0]/sorted_v[1]


def set_default_by_kwarg(key,kwarg,default):
    """
    :param key: kwarg key
    :param kwarg: kwargs
    :param default: default value
    """
    if key in kwarg:
        return kwarg[key]
    else:
        return default


def in_range(val,min_val,max_val):
    if (val < min_val) or (val > max_val):
        return(False)
    else:
        return(True)
    
"""
===================================. outlier detection .================================
"""

def outlier_zscore(data,threshold=2):
    z = (data-data.mean())/data.std()
    return (np.abs(z)>threshold)*1

def rolling_average1D(data, window_size=5):
    kernel = np.ones(window_size)/window_size
    return np.convolve(data,kernel,mode='same')

def min_max_norm(data):
    return (data-data.min())/(data.max()-data.min())


    

"""
===================================. CV helper func .================================
"""

def invert_normalize(data, max_percentile=99):
    """
    invert then adjust the image drange to 0-1
    :param data: image data, usually masked
    :param max_percentile: percentile maximum for normalization
    :return: inverted, normalized data
    """
    max_val = np.percentile(data, max_percentile)
    inverted = max_val-data
    inverted[inverted <= 0] = 0
    normalized = inverted/inverted.max()
    return normalized

def percentile_normalize(data,perc1=5,perc2=95,fix_lower_bound=500,fix_higher_bound=None):
    q1,q2 = np.percentile(data,perc1),np.percentile(data,perc2)
    if fix_lower_bound is not None:
        q1 = min(q1,fix_lower_bound)
    if fix_higher_bound is not None:
        q2 = max(q2,fix_higher_bound)
    float_d = data.copy().astype(float)
    transformed = (float_d-q1)/(q2-q1)
    transformed[transformed>1]=1
    transformed[transformed<0]=0
    return transformed


def adjust_image(img, dtype=16, adjust_gamma=True, gamma=1):
    from skimage import exposure
    """
    adjust image data depth and gamma value
    :param img: input image
    :param dtype: bit depth, 8, 12 or 16
    :param adjust_gamma: whether or not correct gamma
    :param gamma: gamma value
    :return: adjusted image
    """
    if isinstance(dtype, int) & (dtype > 2):
        n_range = (0, 2 ** dtype - 1)
    else:
        print("Illegal input found where an integer no less than 2 was expected.")
    outimg = exposure.rescale_intensity(img, out_range=n_range)
    if adjust_gamma:
        outimg = exposure.adjust_gamma(outimg, gamma=gamma)
    return outimg


"""
===================================. miscellaneous .================================
"""

def reorient_data(data, skew='left'):
    skewed_direction = 'right'
    interpolated = np.interp(np.linspace(0,1,100),np.linspace(0,1,len(data)),data)
    if interpolated[:50].mean()>interpolated.mean():
        skewed_direction='left'
    if skewed_direction == skew:
        return False
    else:
        return True
    
    
def current_datetime():
    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string


def rank2group(n=1006,n_group=50):
    ext = int(0.5*(n%n_group))
    step = int(n/n_group)
    indices = []
    for i in range(n_group):
        group = np.arange(i*step,i*step+step)+ext
        group = group[group<n-1]
        indices.append(group)
    return indices