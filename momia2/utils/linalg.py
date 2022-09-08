import numpy as np
from scipy.interpolate import splprep, splev, RectBivariateSpline
from numba import jit

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
    return (Ia*wa) + (Ib*wb) + (Ic*wc) + (Id*wd)


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


@jit(nopython=True, cache=True)
def intersect_lines(x1, y1, x2, y2, x3, y3, x4, y4):  
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3
    intersect_x = (B1 * C2 - B2 * C1) / (A2 * B1 - A1 * B2)
    intersect_y = (A1 * C2 - A2 * C1) / (B2 * A1 - B1 * A2)
    return intersect_x, intersect_y


def point_line_orientation(l1,l2,p):
    """
    """
    return np.sign(((l2[0] - l1[0]) * (p[1] - l1[1]) - (l2[1] - l1[1]) * (p[0] - l1[0])))


def interpolate_2Dmesh(data_array, smooth=1):
    x_mesh = np.linspace(0, data_array.shape[0] - 1, data_array.shape[0]).astype(int)
    y_mesh = np.linspace(0, data_array.shape[1] - 1, data_array.shape[1]).astype(int)
    return RectBivariateSpline(x_mesh, y_mesh, data_array, s=smooth)


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


def angle_between_vector(v1, v2):
    d1 = np.sqrt(np.sum(v1 ** 2))
    d2 = np.sqrt(np.sum(v2 ** 2))
    return np.arccos(np.dot(v1, v2) / (d1 * d2))


def measure_length(data, pixel_microns=1):
    v1,v2 = data[:-1], data[1:]
    length = np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2, axis=1)).sum()*pixel_microns
    return(length)


def line_contour_intersection(p1, p2, contour):
    v1, v2 = contour[:-1], contour[1:]
    x1, y1 = v1.T
    x2, y2 = v2.T
    x3, y3 = p1
    x4, y4 = p2
    xy = np.array(intersect_lines(x1, y1, x2, y2, x3, y3, x4, y4)).T
    dxy_v1 = xy - v1
    dxy_v2 = xy - v2
    dxy = dxy_v1 * dxy_v2
    intersection_points = xy[np.where(np.logical_and(dxy[:, 0] < 0, dxy[:, 1] < 0))]
    if len(intersection_points) > 2:
        dist = np.sum(np.square(np.tile(p1, (len(intersection_points), 1)) - intersection_points),
                      axis=1)
        intersection_points = intersection_points[np.argsort(dist)[0:2]]
    return intersection_points


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