import numpy as np
from .generic import *
from .linalg import *
from shapely.geometry import Polygon
from centerline.geometry import Centerline
from shapely.ops import linemerge
from .skeleton import find_poles

def find_contour_marching_squares(image,
                                  binary_mask,
                                  level=0.1,
                                  dilation=True,
                                  dark_background=False,
                                  gaussian_smooth_sigma=2):
    """
    2021.04.05
    expand or shrink the contours to maximize average Sobel intensity
    :param image: reference image (phase contrast)
    :param binary_mask: binary mask for
    :param sobel: Sobel edge
    :return: initial contour
    """
    binary_mask = binary_mask.astype(int)
    if not identical_shapes([image,binary_mask]):
        raise ValueError('Input images are of different sizes!')
    if dilation:
        binary_mask = morphology.binary_dilation(binary_mask).copy()
    if not dark_background:
        intensity_mask = np.zeros(image.shape)
        intensity_mask[binary_mask==1] = invert_normalize(image[binary_mask==1])
    else:
        intensity_mask = binary_mask*image
    intensity_mask = filters.gaussian(intensity_mask, sigma=gaussian_smooth_sigma)
    contours = measure.find_contours(intensity_mask, level=level)
    return contours

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



def optimize_contour(contours,
                     target,
                     step_range=(-1.5, 1.5),
                     n_steps=20,
                     contour_length_range=(30,100),
                     smoothing='moving_window',
                     smooth_factor=2):
    """
    subpixel contour optimization
    :param contours: input contours (1-N)
    :param target: target edge map, sobel or schar
    :param step_range: range of contour enlargement/shrinkage
    :param n_steps: number of steps
    :param smoothing: initial smoothing method
    :return:
    """

    optimized_contours = []
    steps = np.linspace(step_range[0], step_range[1], n_steps)
    for contour in contours:
        if smoothing == 'spline':
            n = value_constrain(int(0.5 * len(contour)), contour_length_range[0], contour_length_range[1])
            smoothed = spline_approximation(contour, smooth_factor=smooth_factor, n=n,closed=True)
        elif smoothing == 'moving_window':
            smoothed = contour_moving_window_smoothing(contour)

        unit_perp = unit_perpendicular_vector(smoothed, closed=True)
        global_intensities = []
        for s in steps:
            c = smoothed + unit_perp * s
            x,y = c.T
            x[x<0]=0
            x[x>target.shape[0]-1]=target.shape[0]-1
            y[y<0]=0
            y[y>target.shape[1]-1]=target.shape[1]-1
            #if c.min()>=0 and c[:,0].max()<=target.shape[0] and c[:,1].max()<=target.shape[1]:
            global_intensities.append(np.median(bilinear_interpolate_numpy(target, x, y)))
        max_contour = steps[np.argmax((global_intensities))] * unit_perp + smoothed
        max_contour[-1] = max_contour[0]
        optimized_contours.append(max_contour)
    return optimized_contours


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


def contour2midlines(contour,
                        interpolation_distance=1,
                        smooth_factor=2,
                        attributes = {"id": 1, "name": "polygon", "valid": True},
                        min_fragment_length = 0.5):
    """
    Computes extended midlines from any given contour.
    """
    midline = Centerline(Polygon(contour),
                            interpolation_distance=interpolation_distance, 
                            **attributes)
    fragments = linemerge(midline)
    midline_dict = {}
    
    # filter and sort midlines
    try:    
        total_length = 0
        pole_array = []
        counter=0
        for l in fragments:
            xy = np.asarray(l.xy).T
            line_length = measure_length(xy)
            if line_length>min_fragment_length:
                xy = spline_approximation(xy,
                                      smooth_factor=smooth_factor,
                                                      closed=False)
                midline_dict[counter]=[xy,line_length]
                pole_array.append([xy[0][0],xy[0][1],counter,0])
                pole_array.append([xy[-1][0],xy[-1][1],counter,1])
                counter+=1
        pole_array = np.array(pole_array)
        dist = distance_matrix(pole_array[:,:2],pole_array[:,:2])
        dist[np.diag_indices_from(dist)]=np.inf
        extend_poles = np.sum(dist<1,axis=1)==0
        for i,l in midline_dict.items():
            find_pole1 = extend_poles[i*2]
            find_pole2 = extend_poles[i*2+1]
            p1,p2,trim_l = find_poles(l[0],contour,find_pole1=find_pole1, find_pole2=find_pole2)
            midline_coords = np.vstack([p1,trim_l,p2])
            length = measure_length(midline_coords)
            midline_dict[i]=[midline_coords,length]
            total_length+=length
    except:
        xy = np.asarray(fragments.xy).T
        xy = spline_approximation(xy,smooth_factor=smooth_factor,closed=False)
        line_length = measure_length(xy)
        p1,p2,trim_l = find_poles(xy,contour,find_pole1=True,find_pole2=True)
        midline_coords = np.vstack([p1,trim_l,p2])
        length = measure_length(midline_coords)
        midline_dict[0]=[midline_coords,length]
        total_length = length
    return midline_dict,total_length