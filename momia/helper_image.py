import numpy as np
import warnings
from scipy import fftpack, stats
from skimage import filters, feature, exposure, measure, morphology, transform, draw, registration
from scipy.interpolate import splprep, splev, RectBivariateSpline
import scipy.ndimage as ndi
from numba import jit
from .helper_generic import *
from .optimize import *
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations_with_replacement
from matplotlib import pyplot as plt


def fft(img, subtract_mean=True):
    """
    fast Fourier transform module
    :param img: input image
    :param subtract_mean:
    :return: FFT transformed image
    """
    warnings.filterwarnings("ignore")
    if subtract_mean:
        img = img - np.mean(img)
    return fftpack.fftshift(fftpack.fft2(img))


def fft_reconstruction(fft_img, filters):
    """
    reconstruct image after FFT band pass filtering.
    :param fft_img: input FFT transformed image
    :param filters: low/high frequency bandpass filters
    :return: bandpass filtered, restored phase contrast image
    """

    warnings.filterwarnings("ignore")
    if len(filters) > 0:
        for filter in filters:
            try:
                fft_img *= filter
            except:
                raise ValueError("Illegal input filter found, shape doesn't match?")
    return fftpack.ifft2(fftpack.ifftshift(fft_img)).real


def bandpass_filter(pixel_microns, img_width=2048, img_height=2048, high_pass_width=0.2, low_pass_width=20):

    """

    :param pixel_microns: pixel unit length
    :param img_width: width of image by pixel
    :param img_height: height of image by pixel
    :param high_pass_width: 1/f where f is the lower bound of high frequency signal
    :param low_pass_width: 1/f where f is the upper bound of low frequency signal
    :return: high/low bandpass filters

    """
    u_max = round(1 / pixel_microns, 3) / 2
    v_max = round(1 / pixel_microns, 3) / 2
    u_axis_vec = np.linspace(-u_max / 2, u_max / 2, img_width)
    v_axis_vec = np.linspace(-v_max / 2, v_max / 2, img_height)
    u_mat, v_mat = np.meshgrid(u_axis_vec, v_axis_vec)
    centered_mesh = np.sqrt(u_mat ** 2 + v_mat ** 2)
    if high_pass_width == 0:
        high_pass_filter = np.ones((img_width, img_height)).astype(np.int)
    else:
        high_pass_filter = np.e ** (-(centered_mesh * high_pass_width) ** 2)
    if low_pass_width == 0:
        low_pass_filter = np.ones((2048, 2048)).astype(np.int)
    else:
        low_pass_filter = 1 - np.e ** (-(centered_mesh * low_pass_width) ** 2)
    return high_pass_filter, low_pass_filter


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


def invert_normalization(data, max_percentile=99):
    """
    invert then normalize
    :param data: image data, usually masked
    :param max_percentile: percentile maximum for normalization
    :return: inverted, normalized data
    """
    max_val = np.percentile(data, max_percentile)
    inverted = max_val-data
    inverted[inverted <= 0] = 0
    normalized = inverted/inverted.max()
    return normalized


def adjust_image(img, dtype=16, adjust_gamma=True, gamma=1):
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
    return outimg.astype(np.uint16)


def find_contour_marching_squares(image,
                                  binary_mask,
                                  level=0.1,
                                  dilation=True):
    """
    2021.04.05
    expand or shrink the contours to maximize average Sobel intensity
    :param image: reference image (phase contrast)
    :param binary_mask: binary mask for
    :param sobel: Sobel edge
    :return: initial contour
    """
    if not identical_shapes([image,binary_mask]):
        raise ValueError('Input images are of different sizes!')
    if dilation:
        binary_mask = morphology.binary_dilation(binary_mask).copy()
    inverted_mask = np.zeros(image.shape)
    inverted_mask[binary_mask==1] = invert_normalization(image[binary_mask==1])
    inverted_mask = filters.gaussian(inverted_mask, sigma=0.5)
    contours = measure.find_contours(inverted_mask, level=level)
    return contours


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
    return max(min_value,min(value,max_value))


def spline_approximation(init_contour, n=200, smooth_factor=1, closed=True):
    """
    spline func.
    :param init_contour:
    :param n:
    :param smooth_factor:
    :param closed:
    :return:
    """
    if closed:
        tck, u = splprep(init_contour.T, u=None, s=smooth_factor, per=1)
    else:
        tck, u = splprep(init_contour.T, u=None, s=smooth_factor)
    u_new = np.linspace(u.min(), u.max(), n)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.array([x_new, y_new]).T


def contour_moving_window_smoothing(contour,
                                    window_size='auto',
                                    window_fraction=0.015,
                                    min_window_size=5,
                                    max_window_size=11):
    """
    moving average smoothing of a contour line
    :param contour: input contour line
    :param window_size: window size, automatically determined by window fraction if not specified
    :param window_fraction: approximate fraction of the contour length for estimation of window size
    :param min_window_size: minimum window size
    :param max_window_size: maximum window size
    :return: moving average smoothed contour line
    """
    if window_size == 'auto':
        w = min(max(int(window_fraction*len(contour))*2+1,min_window_size),max_window_size)
    elif isinstance(window_size,int) and int(window_size)%2==1:
        w = min(max(window_size,min_window_size),max_window_size)
    else:
        raise ValueError('Illegal window size!')
    half_w = int(w/2)
    roll_steps = np.arange(-half_w,half_w+1)
    smoothed = np.sum([np.roll(contour,x,axis=0) for x in roll_steps],axis=0)/w
    smoothed = np.vstack([smoothed,smoothed[0].reshape(-1,2)])
    return smoothed


def bilinear_interpolate_numpy(im, x, y):
    """
    bilinear interpolation 2D
    :param im: target image
    :param x: x coordinates
    :param y: y coordinates
    :return: interpolated data
    """
    h,l = im.shape
    padded = np.zeros((h+1,l+1))
    padded[:h,:l] += im
    im = padded
    x0 = x.astype(int)
    x1 = x0 + 1
    y0 = y.astype(int)
    y1 = y0 + 1
    Ia = im[x0,y0]
    Ib = im[x0,y1]
    Ic = im[x1,y0]
    Id = im[x1,y1]
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    return np.round((Ia*wa) + (Ib*wb) + (Ic*wc) + (Id*wd), 4)


def unit_perpendicular_vector(data, closed=True):
    """
    :param data:
    :param closed:
    :return:
    """
    p1 = data[1:]
    p2 = data[:-1]
    dxy = p1 - p2
    ang = np.arctan2(dxy.T[1], dxy.T[0]) + 0.5 * np.pi
    dx, dy = np.cos(ang), np.sin(ang)
    unit_dxy = np.array([dx, dy]).T
    if not closed:
        unit_dxy = np.concatenate([[unit_dxy[0]], unit_dxy])
    else:
        unit_dxy = np.concatenate([unit_dxy, [unit_dxy[0]]])
    return unit_dxy


def find_poles(smoothed_skeleton,
               smoothed_contour,
               find_pole1=True,
               find_pole2=True):
    # find endpoints and their nearest neighbors on a midline
    length = len(smoothed_skeleton)
    extended_pole1 = [smoothed_skeleton[0]]
    extended_pole2 = [smoothed_skeleton[-1]]
    i = 0
    j = 0
    if find_pole1:
        for i in range(5):
            p1 = smoothed_skeleton[i]
            p2 = smoothed_skeleton[i + 1]
            # find the two intersection points between
            # the vectorized contour and line through pole1
            intersection_points_pole1 = line_contour_intersection(p1, p2, smoothed_contour)
            # find the interesection point with the same direction as the outward pole vector
            dxy_1 = p1 - p2
            p1_tile = np.tile(p1, (len(intersection_points_pole1), 1))
            p1dot = (intersection_points_pole1 - p1_tile) * dxy_1
            index_1 = np.where((p1dot[:, 0] > 0) & (p1dot[:, 1] > 0))[0]
            if len(index_1) > 0:
                extended_pole1 = intersection_points_pole1[index_1]
                break
    else:
        i = 1

    if find_pole2:
        for j in range(5):
            p3 = smoothed_skeleton[-1 - j]
            p4 = smoothed_skeleton[-2 - j]
            # find the two intersection points between
            # the vectorized contour and line through pole1
            intersection_points_pole2 = line_contour_intersection(p3, p4, smoothed_contour)
            # find the interesection point with the same direction as the outward pole vector
            dxy_2 = p3 - p4
            p3_tile = np.tile(p3, (len(intersection_points_pole2), 1))
            p3dot = (intersection_points_pole2 - p3_tile) * dxy_2
            index_2 = np.where((p3dot[:, 0] > 0) & (p3dot[:, 1] > 0))[0]
            if len(index_2) > 0:
                extended_pole2 = intersection_points_pole2[index_2]
                break
    else:
        j = 1
    trimmed_midline = smoothed_skeleton[i:length - j]
    return extended_pole1, extended_pole2, trimmed_midline


def extend_skeleton(smoothed_skeleton, smoothed_contour,
                    find_pole1=True, find_pole2=True, interpolation_factor=1):
    # initiate approximated tip points
    new_pole1, new_pole2, smoothed_skeleton = find_poles(smoothed_skeleton,
                                                         smoothed_contour,
                                                         find_pole1=find_pole1,
                                                         find_pole2=find_pole2)
    extended_skeleton = np.concatenate([new_pole1,
                                        smoothed_skeleton,
                                        new_pole2])
    return spline_approximation(extended_skeleton,
                                n=int(interpolation_factor * len(smoothed_skeleton)),
                                smooth_factor=1, closed=False)


# old method, inefficient looping, deprecated
def find_midpoints(smoothed_skeleton, smoothed_contour, max_dxy=1):
    d_perp = unit_perpendicular_vector(smoothed_skeleton, closed=False)
    updated_skeleton = smoothed_skeleton.copy()
    for i in range(len(smoothed_skeleton)):
        xy = line_contour_intersection(smoothed_skeleton[i],
                                       d_perp[i] + smoothed_skeleton[i],
                                       smoothed_contour)
        coords = np.average(xy, axis=0)
        if (len(coords) == 2) and (np.isnan(coords).sum() == 0):
            if distance(smoothed_skeleton[i], coords) <= 2:
                updated_skeleton[i] = coords
    return updated_skeleton


def intersect_matrix(line, contour,
                     orthogonal_vectors=None):
    if orthogonal_vectors is None:
        dxy = unit_perpendicular_vector(line, closed=False)
    else:
        dxy = orthogonal_vectors
    v1, v2 = contour[:-1], contour[1:]
    x1, y1 = v1.T
    x2, y2 = v2.T
    x3, y3 = line.T
    perp_xy = line + dxy
    x4, y4 = perp_xy.T
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    A1B2 = A1[:, np.newaxis] * B2
    A1C2 = A1[:, np.newaxis] * C2
    B1A2 = B1[:, np.newaxis] * A2
    B1C2 = B1[:, np.newaxis] * C2
    C1A2 = C1[:, np.newaxis] * A2
    C1B2 = C1[:, np.newaxis] * B2

    intersect_x = (B1C2 - C1B2) / (B1A2 - A1B2)
    intersect_y = (A1C2 - C1A2) / (A1B2 - B1A2)
    return intersect_x, intersect_y


def distance_matrix(data1, data2):
    x1, y1 = data1.T
    x2, y2 = data2.T
    dx = x1[:, np.newaxis] - x2
    dy = y1[:, np.newaxis] - y2
    dxy = np.sqrt(dx ** 2 + dy ** 2)
    return dxy


def between_contour_points(p, contour):
    dist = distance_matrix(contour, p)
    return np.argmin(dist)


def divide_contour(p1, p2, contour):
    pos1 = between_contour_points(p1, contour)
    pos2 = between_contour_points(p2, contour)
    if pos1 == pos2:
        raise ValueError('Two endpoints are identical!')
    else:
        pos1, pos2 = sorted([pos1, pos2])
        half_contour_1 = contour[pos1:min(pos2 + 1, len(contour) - 1)]
        half_contour_2 = np.concatenate([contour[pos2:-1], contour[:pos1]])
    return half_contour_1, half_contour_2


def direct_intersect_points(skeleton, contour):
    v1, v2 = contour[:-1], contour[1:]
    skel_x, skel_y = skeleton.T
    intersect_x, intersect_y = intersect_matrix(skeleton, contour)
    dx_v1 = intersect_x - v1.T[0][:, np.newaxis]
    dx_v2 = intersect_x - v2.T[0][:, np.newaxis]
    dy_v1 = intersect_y - v1.T[1][:, np.newaxis]
    dy_v2 = intersect_y - v2.T[1][:, np.newaxis]
    dx = dx_v1 * dx_v2
    dy = dy_v1 * dy_v2
    dist_x = skel_x[np.newaxis, :] - intersect_x
    dist_y = skel_y[np.newaxis, :] - intersect_y

    non_boundry_points = np.where(np.logical_and(dy > 0, dx > 0))
    dist_matrix = np.sqrt(dist_x ** 2 + dist_y ** 2)
    dist_matrix[non_boundry_points] = np.inf
    nearest_id_x = np.argsort(dist_matrix, axis=0)[:2]
    nearest_id_y = np.linspace(0, dist_matrix.shape[1] - 1, dist_matrix.shape[1]).astype(int)

    pos1_list = np.array([intersect_x[nearest_id_x[0], nearest_id_y], intersect_y[nearest_id_x[0], nearest_id_y]]).T
    pos2_list = np.array([intersect_x[nearest_id_x[1], nearest_id_y], intersect_y[nearest_id_x[1], nearest_id_y]]).T
    midpoints = (pos1_list + pos2_list) / 2
    d_midpoints = np.abs(midpoints - skeleton)
    outliers = np.where(d_midpoints > 1)
    midpoints[outliers] = skeleton[outliers]
    return midpoints

def line_contour_intersection(p1, p2, contour):
    v1, v2 = contour[:-1], contour[1:]
    x1, y1 = v1.T
    x2, y2 = v2.T
    x3, y3 = p1
    x4, y4 = p2
    xy = np.array(line_intersect(x1, y1, x2, y2, x3, y3, x4, y4)).T
    dxy_v1 = xy - v1
    dxy_v2 = xy - v2
    dxy = dxy_v1 * dxy_v2
    intersection_points = xy[np.where(np.logical_and(dxy[:, 0] < 0, dxy[:, 1] < 0))]
    if len(intersection_points) > 2:
        dist = np.sum(np.square(np.tile(p1, (len(intersection_points), 1)) - intersection_points),
                      axis=1)
        intersection_points = intersection_points[np.argsort(dist)[0:2]]
    return intersection_points


@jit(nopython=True, cache=True)
def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):  # Ax+By = C
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3
    intersect_x = (B1 * C2 - B2 * C1) / (A2 * B1 - A1 * B2)
    intersect_y = (A1 * C2 - A2 * C1) / (B2 * A1 - B1 * A2)
    return intersect_x, intersect_y


@jit(nopython=True, cache=True)
def estimate_boundary_length(boundary_coords):
    max_L = len(boundary_coords)
    dist = 0
    for i in range(len(boundary_coords)):
        for j in range(i+1,len(boundary_coords)):
            x1,y1 = boundary_coords[i]
            x2,y2 = boundary_coords[j]
            dist =  np.sqrt((x2-x1)**2+(y2-y1)**2)
            if dist > max_L:
                max_L = dist
    return max_L


def midline_approximation(skeleton, smoothed_contour,
                          tolerance=0.1, max_iteration=10):
    midline = skeleton.copy()
    n = 0
    converged = False
    while n < max_iteration:
        updated_midline = direct_intersect_points(midline, smoothed_contour)
        dxy = updated_midline - midline
        midline = spline_approximation(updated_midline, n=len(updated_midline),
                                       smooth_factor=1, closed=False)
        if dxy.max() <= tolerance:
            converged = True
            break
        n += 1
    return midline.astype(np.float), converged


# inefficient width calculation, deprecated
def measure_width(extended_skeleton, smoothed_contour):
    length = line_length(extended_skeleton)
    extended_skeleton = spline_approximation(extended_skeleton,
                                             int(round(length)),
                                             smooth_factor=0, closed=False)
    d_perp = unit_perpendicular_vector(extended_skeleton, closed=False)
    width_list = []
    for i in range(1, len(extended_skeleton)-1):
        xy = line_contour_intersection(extended_skeleton[i],
                                       d_perp[i] + extended_skeleton[i],
                                       smoothed_contour)
        coords = np.average(xy, axis=0)
        if (len(xy) == 2) and (np.isnan(coords).sum() == 0):
            width_list.append(distance(xy[0], xy[1]))
        else:
            raise ValueError('Error encountered while computing line intersection points!')
    return extended_skeleton, np.array([0]+width_list+[0]), length


def direct_intersect_distance(skeleton, contour):
    v1, v2 = contour[:-1], contour[1:]
    skel_x, skel_y = skeleton[1:-1].T
    intersect_x, intersect_y = intersect_matrix(skeleton[1:-1], contour)
    dx_v1 = intersect_x - v1.T[0][:, np.newaxis]
    dx_v2 = intersect_x - v2.T[0][:, np.newaxis]
    dy_v1 = intersect_y - v1.T[1][:, np.newaxis]
    dy_v2 = intersect_y - v2.T[1][:, np.newaxis]
    dx = dx_v1 * dx_v2
    dy = dy_v1 * dy_v2
    dist_x = skel_x[np.newaxis, :] - intersect_x
    dist_y = skel_y[np.newaxis, :] - intersect_y

    non_boundry_points = np.where(np.logical_and(dy > 0, dx > 0))
    dist_matrix = np.sqrt(dist_x ** 2 + dist_y ** 2)
    dist_matrix[non_boundry_points] = np.inf
    nearest_id_x = np.argsort(dist_matrix, axis=0)[:2]
    nearest_id_y = np.linspace(0, dist_matrix.shape[1] - 1, dist_matrix.shape[1]).astype(int)
    dists = dist_matrix[nearest_id_x[0], nearest_id_y] + dist_matrix[nearest_id_x[1], nearest_id_y]
    return np.concatenate([[0], dists, [0]])


def suppress_extreme_edge_points(edge, tolerance=1):
    dif = edge[1:] - edge[:-1]
    dist = np.sqrt(np.sum(dif ** 2, axis=1))
    extremities = np.where(dist > tolerance)[0]
    new_edge = edge.copy()
    if len(extremities) > 0:
        for i in extremities:
            new_edge[i + 1] = 2 * edge[i] - edge[i - 1]
    return new_edge


def line_length(line):
    v1 = line[:-1]
    v2 = line[1:]
    d = v2-v1
    return np.sum(np.sqrt(np.sum(d**2,axis=1)))


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


def background_subtraction(data, method='rolling_ball', **kwargs):
    if method == 'rolling_ball':
        return rolling_ball_bg_subtraction(data, **kwargs)
    else:
        raise ModuleNotFoundError('Method {} not found!'.format(method))


def touching_edge(img_shape,optimized_bbox):
    (rows, columns) = img_shape
    (x1, y1, x2, y2) = optimized_bbox
    if min(x1, y1, rows-x2-1,columns-y2-1) <= 0:
        return True
    else:
        return False


def measure_along_contour(target, contour):
    x = contour[:,0]
    y = contour[:,1]
    return bilinear_interpolate_numpy(target,x,y)


def orientation_by_eig(inertia_tensor):
    u20,u11,u02 = inertia_tensor[0,0],\
                  -inertia_tensor[0,1],\
                  inertia_tensor[1,1]
    if u20 == u02:
        if u11 == 0:
            theta = 0
        elif u11 > 0:
            theta = 0.25*np.pi
        elif u11 < 0:
            theta = -0.25*np.pi
    else:
        if u11 == 0:
            if u20<u02:
                theta = -0.5*np.pi
            elif u20>u02:
                theta = 0
        elif u20 > u02:
            theta = 0.5*np.arctan(2*u11/(u20-u02))
        elif u20 < u02:
            if u11 > 0:
                theta = 0.5*np.arctan(2*u11/(u20-u02)) + 0.5*np.pi
            elif u11 < 0:
                theta = 0.5*np.arctan(2*u11/(u20-u02)) - 0.5*np.pi
    return theta


def get_regionprop(labels, intensity_image=None):
    regionprop_properties = ['label','bbox','coords',
                             'area','convex_area','filled_area','eccentricity','solidity',
                             'major_axis_length','minor_axis_length',
                             'perimeter','equivalent_diameter',
                             'extent','orientation',
                             'inertia_tensor','moments_hu','weighted_moments_hu',
                             'mean_intensity','max_intensity','min_intensity']

    if intensity_image is not None:
        rp_table = pd.DataFrame(measure.regionprops_table(labels,
                                                      intensity_image,
                                                      properties=regionprop_properties))
    else:
        rp_table = pd.DataFrame(measure.regionprops_table(labels,
                                                          intensity_image=(labels>0)*1,
                                                          properties=regionprop_properties))
    rp_table['roundness'] = np.round(4*rp_table['area'].values/np.pi*(rp_table['major_axis_length'].values**2),3)
    rp_table['circularity'] = np.round(np.sqrt(4*np.pi*rp_table['area'].values/(rp_table['perimeter'].values**2)),3)
    rp_table['aspect_ratio'] = np.round(rp_table['minor_axis_length'].values/rp_table['major_axis_length'].values,3)


    return rp_table


def hu_log10_transform(hu_moments):
    warnings.filterwarnings("ignore")
    abs_vals = np.abs(hu_moments)
    hu_log10 = np.copysign(np.log10(abs_vals), hu_moments)
    corrected = np.zeros(hu_moments.shape)
    corrected[np.isfinite(hu_log10)] = hu_log10[np.isfinite(hu_log10)]
    return corrected


def median_over_gaussian(data,windowsize=3):
    return (ndi.median_filter(data,size=windowsize)/filters.gaussian(data,sigma=windowsize,preserve_range=True))


def ratio_of_gaussian(data, s1=1,s2=5):
    g1 = filters.gaussian(data,sigma=s1)
    g2 = filters.gaussian(data,sigma=s2)
    return g1/g2


def normalized_difference_of_gaussian(data, s1=0, s2=10):
    g1 = filters.gaussian(data, sigma=s1)
    g2 = filters.gaussian(data, sigma=s2)
    return (g1-g2)/g1

def percentile(regionmask, intensity):
    return np.percentile(intensity[regionmask], q=(25, 50, 75))


def cv(regionmask, intensity):
    return 100*np.std(intensity[regionmask])/np.mean(intensity[regionmask])


def skewness(regionmask, intensity):
    return stats.skew(intensity[regionmask])


def kurtosis(regionmask, intensity):
    return stats.kurtosis(intensity[regionmask])


def std(regionmask, intensity):
    return np.std(intensity[regionmask])


def bend_angle(data, window=3):
    p1 = np.concatenate((data[-window:],data[:-window])).T
    p2 = data.copy().T
    p3 = np.concatenate((data[window:],data[0:window])).T
    p1p2 = p1[0]*1+p1[1]*1j - (p2[0]*1+p2[1]*1j)
    p1p3 = p1[0]*1+p1[1]*1j - (p3[0]*1+p3[1]*1j)
    return np.angle(p1p3/p1p2, deg=True)


def bend_angle_open(data, window=5):
    p1 = data[:-2*window].T
    p2 = data[window:-window].T
    p3 = data[2*window:].T
    p1p2 = p1[0]*1+p1[1]*1j - (p2[0]*1+p2[1]*1j)
    p2p3 = p2[0]*1+p2[1]*1j - (p3[0]*1+p3[1]*1j)
    return np.angle(p2p3/p1p2, deg=True)

def distance(v1, v2):
    #Euclidean distance of two points
    return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2))


def in_range(val,min_val,max_val):
    if (val < min_val) or (val > max_val):
        return(False)
    else:
        return(True)


class merge_joint:

    def __init__(self, data, old_idxlist):
        self.data = [set(elem) for elem in data]
        self.taken = [False] * len(self.data)
        self.newdata = []
        self.old_idxlist = old_idxlist
        self.new_idxlist = []
        self.merge_sets()

    def dfs(self, node, idx):
        self.taken[idx] = True
        newset = node
        self.new_idxlist[-1].append(self.old_idxlist[idx])
        for i, item in enumerate(self.data):
            if not self.taken[i] and not newset.isdisjoint(item):
                newset.update(self.dfs(item, i))
        return newset

    def merge_sets(self):
        for idx, node in enumerate(self.data):
            if not self.taken[idx]:
                self.new_idxlist.append([])
                self.newdata.append(list(self.dfs(node, idx)))


def locate_nodes(skeleton):
    endpoints = []
    branch_points = []
    skeleton_path = np.where(skeleton>0)
    skeleton_length = len(skeleton_path[0])
    if skeleton_length > 5:
        for i in range(skeleton_length):
            x = skeleton_path[0][i]
            y = skeleton_path[1][i]
            if cube_nonzero(skeleton, x, y) == 1:
                endpoints.append([x, y])
            if cube_nonzero(skeleton,x, y) == 2:
                _n,  _neighbors = cube_nonzero_with_return(skeleton, x, y)
                _x0, _y0 = _neighbors[0]
                _x1, _y1 = _neighbors[1]
                dist=abs(_x0-_x1) +abs(_y0-_y1)
                if dist == 1:
                    skeleton[x, y] = 0
                    if skeleton[_x0, _y0] == 3:
                        skeleton[_x0, _y0] = 2
                    if skeleton[_x1, _y1] == 3:
                        skeleton[_x1, _y1] = 2
                    if [_x0, _y0] in branch_points:
                        branch_points.remove([_x0, _y0])
                    if [_x1, _y1] in branch_points:
                        branch_points.remove([_x1, _y1])
            if cube_nonzero(skeleton, x, y) > 2:
                branch_points.append([x, y])
                skeleton[x, y] = 3
        return endpoints,branch_points,skeleton
    else:
        return endpoints,branch_points,skeleton


@jit(nopython=True, cache=True)
def neighbor_search(input_map, endpoint, max_iterations=500):
    output_x = []
    output_y = []
    end_reached = False
    x, y = endpoint[0], endpoint[1]
    counter=0
    while not end_reached:
        if counter >= max_iterations:
            break
        n_neighbor, neighbor_list = cubecount_with_return(input_map, x, y, 2)
        if n_neighbor == 0:
            input_map[x, y] = 1
            output_x.append(x)
            output_y.append(y)
            end_reached = True
        elif n_neighbor == 1:
            input_map[x, y] = 1
            output_x.append(x)
            output_y.append(y)
            x, y = neighbor_list[-1]
        elif n_neighbor >= 2:
            input_map[x, y] = 1
            output_x.append(x)
            output_y.append(y)
            end_reached = True
        counter += 1
    return output_x, output_y


@jit(nopython=True, cache=True)
def cubesum(mask, px, py, val):
    n = 0
    for i in [0, 1, 2, 3, 5, 6, 7, 8]:
        dx = int(i / 3)
        dy = i - dx * 3
        x1, y1 = px + dx - 1, py + dy - 1
        if mask[x1, y1] == val:
            n += 1
    return n


@jit(nopython=True, cache=True)
def cube_nonzero(mask, px, py):
    n = 0
    for i in [0, 1, 2, 3, 5, 6, 7, 8]:
        dx = int(i / 3)
        dy = i - dx * 3
        x1, y1 = px + dx - 1, py + dy - 1
        if mask[x1, y1] > 0:
            n += 1
    return n


@jit(nopython=True, cache=True)
def cube_nonzero_with_return(mask, px, py):
    retlist = []
    n = 0
    for i in [0, 1, 2, 3, 5, 6, 7, 8]:
        dx = int(i / 3)
        dy = i - dx * 3
        x1, y1 = px + dx - 1, py + dy - 1
        if mask[x1, y1] > 0:
            retlist.append([x1, y1])
            n += 1
    return n, retlist


@jit(nopython=True, cache=True)
def cubecount_with_return(mask, px, py, val):
    retlist = []
    n = 0
    for i in [0, 1, 2, 3, 5, 6, 7, 8]:
        dx = int(i / 3)
        dy = i - dx * 3
        x1, y1 = px + dx - 1, py + dy - 1
        if mask[x1, y1] == val:
            retlist.append([x1, y1])
            n += 1
    return n, retlist


def skeleton_analysis(mask, pruning=False, min_branch_length=5, max_iterations=30):
    skeleton = morphology.skeletonize(mask)*2
    endpoints, branch_points, skeleton = locate_nodes(skeleton)
    anchor_points = endpoints+branch_points
    skeleton_branches = []
    n = 0
    if (len(endpoints) >= 2) and (len(branch_points) <= 10):
        while True:
            if len(anchor_points) == 0:
                break
            if n>=max_iterations:
                break
            is_real_pole = [0,0]
            startpoint = anchor_points[0]
            if startpoint in endpoints:
                is_real_pole[0] = 1
            xcoords,ycoords = neighbor_search(skeleton,startpoint)
            anchor_points.remove(startpoint)
            if len(xcoords)>=1:
                lx,ly = xcoords[-1],ycoords[-1]
                node_count, node_coords = cubecount_with_return(skeleton, lx, ly, 3)
                if (node_count == 0) and ([lx,ly] in endpoints):
                    is_real_pole[1] = 1
                    anchor_points.remove([lx, ly])
                elif (node_count == 1):
                    lx, ly = node_coords[0]
                    if [lx, ly] in anchor_points:
                        anchor_points.remove([lx, ly])
                    xcoords.append(lx)
                    ycoords.append(ly)
                else:
                    for xy in node_coords:
                        if distance([lx,ly],xy) == 1:
                            lx,ly = xy
                            break
                    if [lx, ly] in anchor_points:
                        anchor_points.remove([lx, ly])
                    xcoords.append(lx)
                    ycoords.append(ly)
                skeleton_branches.append([is_real_pole, xcoords, ycoords])
            n+=1
        branch_copy = skeleton_branches.copy()
        if pruning:
            for branch in branch_copy:
                if len(branch) <= min_branch_length:
                    skeleton_branches.remove(branch)
        if n < max_iterations:
            return skeleton_branches, skeleton
        else:
            return [], skeleton
    else:
        return [], skeleton


def interpolate_2Dmesh(data_array, smooth=1):
    x_mesh = np.linspace(0, data_array.shape[0] - 1, data_array.shape[0]).astype(int)
    y_mesh = np.linspace(0, data_array.shape[1] - 1, data_array.shape[1]).astype(int)
    return RectBivariateSpline(x_mesh, y_mesh, data_array, s=smooth)


def quadratic_maxima_approximation(q1, q2, q3):
    if q2 <= max(q1, q3):
        raise ValueError('Mid point must be local maxima!')
    else:
        return (0.5 * (q1 - q3)) / (q1 - 2 * q2 + q3)


def local_puncta_zscore(image, coords, trim_low_bound=5):

    x, y = coords
    data = np.sort((image[x - 4:x + 5, y - 4:y + 5].copy()).flatten())[trim_low_bound:]

    threshold = filters.threshold_isodata(data, nbins=64)
    threshold = 0.5 * (threshold + data[-4:].mean())

    fg = data[data > threshold]
    bg = data[data <= threshold]

    if len(fg) <= 5:
        z = 0
    else:
        z = (np.mean(fg) - np.mean(bg)) / bg.std()
    return z


def subpixel_approximation_quadratic(x, y, energy_map, max_iteration=2):

    local_maxima_found = False
    for i in range(max_iteration):
        cube = energy_map[x - 1:x + 2, y - 1:y + 2]
        if cube[1, 1] == cube.max():
            local_maxima_found = True
            vx0, vx1, vx2 = cube[0, 1], cube[1, 1], cube[2, 1]
            vy0, vy1, vy2 = cube[1, 0], cube[1, 1], cube[1, 2]
            x += quadratic_maxima_approximation(vx0, vx1, vx2)
            y += quadratic_maxima_approximation(vy0, vy1, vy2)
            break
        else:
            dx, dy = np.array(np.unravel_index(cube.argmax(), cube.shape)) - 1
            x += dx
            y += dy
    return x, y, local_maxima_found


def find_puncta(img, mask):
    img_smoothed = filters.gaussian(img, sigma=0.5)
    img_smoothed = (img_smoothed - img_smoothed.min()) / (img_smoothed.max() - img_smoothed.min())
    LoG = ndi.gaussian_laplace(img_smoothed, sigma=1)
    extrema = feature.peak_local_max(-LoG * mask, min_distance=1, threshold_rel=0.1)
    output = []

    for ex in extrema:
        x, y = ex
        newx, newy, stable_maxima = subpixel_approximation_quadratic(x, y, img_smoothed)
        if stable_maxima:
            #z_image = local_puncta_zscore(img_smoothed, ex)
            #z_LoG = local_puncta_zscore(LoG, ex)
            output.append([newx, newy, img[x, y], LoG[x, y]])
    return np.array(output)


def angle_between_vector(v1, v2):
    d1 = np.sqrt(np.sum(v1 ** 2))
    d2 = np.sqrt(np.sum(v2 ** 2))
    return np.arccos(np.dot(v1, v2) / (d1 * d2))


def measure_length(data, pixel_microns=1):
    v1,v2 = data[:-1], data[1:]
    length = np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2, axis=1)).sum()*pixel_microns
    return(length)


def puncta_projection(cell,foci):
    width=cell.width_lists[0]
    midline = cell.midlines[0]
    L = measure_length(midline)
    norm_xy = []
    for x in foci[:,:2]:
        project_l = []
        project_v = []
        for i in range(0,len(midline)-1):
            u = x-midline[i]
            v = midline[i+1]-midline[i]
            v_norm = np.sqrt(sum(v**2))
            proj_of_u_on_v = (np.dot(u, v)/v_norm**2)*v
            project_l.append(np.sqrt((proj_of_u_on_v**2).sum()))
            project_v.append(midline[i]+proj_of_u_on_v)
        s=np.argmin(project_l)
        p = np.array(project_v)[s]
        a,b = midline[s],midline[s+1]
        sign = np.sign(((b[0] - a[0]) * (x[1] - a[1]) - (b[1] - a[1]) * (x[0] - a[0])))
        dl = np.sqrt(np.square(p-midline[s]).sum())
        l=measure_length(midline[:s+1])
        w=max(0.25*np.max(width),width[s])
        dx = min(0.5,np.sqrt(np.sum((p-x)**2))/w)
        norm_x = sign*dx
        norm_xy.append([norm_x,min((l+dl)/L,1)])
    return np.array(norm_xy)


def locate_puncta(cell,
                  channel='TRITC',
                  min_cutoff=100):
    img = cell.data[channel]
    mask = cell.mask
    punc = find_puncta(img,mask)
    if len(punc) > 0:
        punc = punc[punc[:, 2] >= min_cutoff]
        if len(punc)>0:
            punc_norm = puncta_projection(cell,punc)
    if len(punc) > 0:
        return np.hstack([punc, punc_norm])
    else:
        return np.array([])

def point_line_orientation(l1,l2,p):
    return np.sign(((l2[0] - l1[0]) * (p[1] - l1[1]) - (l2[1] - l1[1]) * (p[0] - l1[0])))
