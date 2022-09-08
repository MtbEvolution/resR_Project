import numpy as np
import scipy.ndimage as ndi
from numba import jit
from .generic import *
from skimage import filters,transform


"""
=================================== Section 1. xy drift correction ===================================

"""

def shift_image(img, shift):
    """
    correct xy drift between phase contrast image and fluorescent image(s)
    :param img: input image
    :param shift: subpixel xy drift
    :return: drift corrected image
    """
    offset_image = ndi.fourier_shift(np.fft.fftn(img), shift)
    offset_image = np.fft.ifftn(offset_image)
    offset_image = np.round(offset_image.real)
    offset_image[offset_image <= 0] = 10
    # rescaled to avoid int16 overflow
    offset_image[offset_image>=65530] = 65530
    return offset_image.astype(np.uint16)

def drift_detection(img, reference, upsample_factor=10):
    """
    wrapped around phase_cross_correlation in skimage's registration module
    """
    from skimage.registration import phase_cross_correlation
    shift, error, _diff = registration.phase_cross_correlation(reference,
                                                               img,
                                                               upsample_factor=upsample_factor)
    return shift

def drift_correction(img, reference, max_offset, upsample_factor=10):
    """
    correct xy drift by phase correlation in Fourier space. 
    """
    shift = drift_detection(img, reference, upsample_factor)
    if max(np.abs(shift)) <= max_offset:
        return shift_image(img, shift)
    else:
        return img
    
    

"""
=================================== Section 2. background correction ===================================

"""    
    
    
    
class rolling_ball:
    def __init__(self, radius=50):
        self.width = 0
        if radius <= 10:
            self.shrink_factor = 1
            self.arc_trim_per = 24
        elif radius <= 20:
            self.shrink_factor = 2
            self.arc_trim_per = 24
        elif radius <= 100:
            self.shrink_factor = 4
            self.arc_trim_per = 32
        else:
            self.shrink_factor = 8
            self.arc_trim_per = 40
        self.radius = radius / self.shrink_factor
        self.build()

    def build(self):
        x_trim = int(self.arc_trim_per * self.radius / 100)
        half_width = int(self.radius - x_trim)
        self.width = int(2 * half_width + 1)
        r_squre = np.ones((self.width, self.width)) * (self.radius ** 2)
        squared = np.square(np.linspace(0, self.width - 1, self.width) - half_width)
        x_val = np.tile(squared, (self.width, 1))
        y_val = x_val.T
        self.ball = np.sqrt(r_squre - x_val - y_val)


def rolling_ball_bg_subtraction(data, **kwargs):
    rolling_ball_radius = set_default_by_kwarg('rolling_ball_radius',kwargs,40)
    output_data = data.copy()
    smoothed = filters.gaussian(data, sigma=1) * 65535
    ball = rolling_ball(radius=rolling_ball_radius)
    shrunk_img = shrink_img_local_min(smoothed, shrink_factor=ball.shrink_factor)
    bg = rolling_ball_bg(shrunk_img, ball.ball)
    bg_rescaled = transform.resize(bg, data.shape, anti_aliasing=True)
    output_data = output_data - bg_rescaled
    output_data[output_data <= 0] = 0
    return output_data.astype(np.uint16)


@jit(nopython=True, cache=True)
def shrink_img_local_min(image, shrink_factor=4):
    s = shrink_factor
    r, c = image.shape[0], image.shape[1]
    r_s, c_s = int(r / s), int(c / s)
    shrunk_img = np.ones((r_s, c_s))
    for x in range(r_s):
        for y in range(c_s):
            shrunk_img[x, y] = image[x * s:x * s + s, y * s:y * s + s].min()
    return shrunk_img


@jit(nopython=True, cache=True)
def rolling_ball_bg(image, ball):
    width = ball.shape[0]
    radius = int(width / 2)
    r, c = image.shape[0], image.shape[1]
    bg = np.ones((r, c)).astype(np.float32)
    # ignore edges to begin with
    for x in range(r):
        for y in range(c):
            x1, x2, y1, y2 = max(x - radius, 0), min(x + radius + 1, r), max(y - radius, 0), min(y + radius + 1, c)
            cube = image[x1:x2, y1:y2]
            cropped_ball = ball[radius - (x - x1):radius + (x2 - x), radius - (y - y1):radius + (y2 - y)]
            bg_cropped = bg[x1:x2, y1:y2]
            bg_mask = ((cube - cropped_ball).min()) + cropped_ball
            bg[x1:x2, y1:y2] = bg_cropped * (bg_cropped >= bg_mask) + bg_mask * (bg_cropped < bg_mask)
    return (bg)


def subtract_background(data, method='rolling_ball', **kwargs):
    if method == 'rolling_ball':
        return rolling_ball_bg_subtraction(data, **kwargs)
    else:
        raise ModuleNotFoundError('Method {} not found!'.format(method))

        
        
"""
=================================== Section 3. Bbox functions ===================================
"""
        

def optimize_bbox(img_shape,
                  bbox,
                  edge_width=8):
    """
    function to optimize bounding box
    :param img_shape: shape of the image to be mapped onto
    :param bbox: inital bbox
    :param edge_width: max edge width
    :return:
    """
    (rows,columns) = img_shape
    (x1,y1,x2,y2) = bbox

    return max(0,x1-edge_width),max(0,y1-edge_width),min(rows-1,x2+edge_width),min(columns-1,y2+edge_width)


def optimize_bbox_batch(img_shape,
                        region_prop_table,
                        edge_width=8):
    """
    function to optimize bounding box
    :param img_shape: shape of the image to be mapped onto
    :param bbox: inital bbox
    :param edge_width: max edge width
    :return:
    """

    (rows, columns) = img_shape
    if 'bbox-0' in region_prop_table.columns:
        (x1, y1, x2, y2) = region_prop_table[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].values.T
    elif '$bbox-0' in region_prop_table.columns:
        (x1, y1, x2, y2) = region_prop_table[['$bbox-0', '$bbox-1', '$bbox-2', '$bbox-3']].values.T
    else:
        raise ValueError('Bbox columns should be "bbox-n" or "$bbox-n" where n is 0, 1, 2, or 3.')
    new_x1, new_y1, new_x2, new_y2 = x1 - edge_width, y1 - edge_width, x2 + edge_width, y2 + edge_width
    new_x1[new_x1 <= 0] = 0
    new_y1[new_y1 <= 0] = 0
    new_x2[new_x2 >= rows - 1] = rows - 1
    new_y2[new_y2 >= columns - 1] = columns - 1

    touching_edge = (((x1 <= 0.5 * edge_width) + (y1 <= 0.5 * edge_width) + \
                    (x2 >= rows - 1 + 0.5 * edge_width) + (y2 >= columns - 1 + 0.5 * edge_width))>0).astype(np.int)
    return np.array([new_x1, new_y1, new_x2, new_y2, touching_edge]).T


"""
=================================== Section 4. Future functions ===================================
"""

