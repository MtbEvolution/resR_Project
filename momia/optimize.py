from . import helper_image as hi
import numpy as np

def optimize_contour(contours,
                     target,
                     step_range=(-1.5, 1.5),
                     n_steps=20,
                     smoothing='moving_window'):
    """
    subpixel contour optimization
    :param contours: input contours (1-N)
    :param target: target edge map, sobel or schar
    :param step_range: degree of contour enlargement/shrinkage
    :param n_steps: number of steps
    :param smoothing: initial smoothing method
    :return:
    """

    optimized_contours = []
    steps = np.linspace(step_range[0], step_range[1], n_steps)
    for contour in contours:
        if smoothing == 'spline':
            n = hi.value_constrain(int(0.5 * len(contour)), 50, 300)
            smoothed = hi.spline_approximation(contour, smooth_factor=2, n=n)
        elif smoothing == 'moving_window':
            smoothed = hi.contour_moving_window_smoothing(contour)

        unit_perp = hi.unit_perpendicular_vector(smoothed, closed=True)
        global_intensities = []
        for s in steps:
            c = smoothed + unit_perp * s
            if c.min()>=0 and c[:,0].max()<=target.shape[0] and c[:,1].max()<=target.shape[1]:
                global_intensities.append(np.mean(hi.bilinear_interpolate_numpy(target, c[:, 0], c[:, 1])))
        max_contour = steps[np.argmax((global_intensities))] * unit_perp + smoothed
        max_contour[-1] = max_contour[0]
        optimized_contours.append(max_contour)
    return optimized_contours


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
    (x1, y1, x2, y2) = region_prop_table[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].values.T
    new_x1, new_y1, new_x2, new_y2 = x1 - edge_width, y1 - edge_width, x2 + edge_width, y2 + edge_width
    new_x1[new_x1 <= 0] = 0
    new_y1[new_y1 <= 0] = 0
    new_x2[new_x2 >= rows - 1] = rows - 1
    new_y2[new_y2 >= columns - 1] = columns - 1

    touching_edge = (((x1 <= 0.5 * edge_width) + (y1 <= 0.5 * edge_width) + \
                    (x2 >= rows - 1 + 0.5 * edge_width) + (y2 >= columns - 1 + 0.5 * edge_width))>0).astype(np.int)
    return np.array([new_x1, new_y1, new_x2, new_y2, touching_edge]).T