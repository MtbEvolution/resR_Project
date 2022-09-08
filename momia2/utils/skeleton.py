import numpy as np
from numba import jit
from skimage import morphology, feature, filters, measure
from .linalg import line_contour_intersection

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
    
    
def distance(v1, v2):
    #Euclidean distance of two points
    return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2))


def find_poles(smoothed_skeleton,
               smoothed_contour,
               find_pole1=True,
               find_pole2=True,
               gap=4,):
    # find endpoints and their nearest neighbors on a midline
    length = len(smoothed_skeleton)
    extended_pole1 = [smoothed_skeleton[0]]
    extended_pole2 = [smoothed_skeleton[-1]]
    i = 0
    j = 0
    if find_pole1:
        for i in range(5):
            p1 = smoothed_skeleton[i]
            p2 = smoothed_skeleton[i + gap]
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
            p4 = smoothed_skeleton[-1-gap - j]
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


def find_poles_simp(smoothed_skeleton,
               smoothed_contour,
               find_pole1=True,
               find_pole2=True,
                gap=3):
    # find endpoints and their nearest neighbors on a midline
    length = len(smoothed_skeleton)
    extended_pole1 = [smoothed_skeleton[0]]
    extended_pole2 = [smoothed_skeleton[-1]]
    i = 0
    j = 0
    if find_pole1:
        for i in range(5):
            p1 = smoothed_skeleton[i]
            p2 = smoothed_skeleton[i + gap]
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
            p4 = smoothed_skeleton[-1-gap - j]
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