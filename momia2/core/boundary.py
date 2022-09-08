from itertools import combinations
from numba import jit
import numpy as np
from skimage import measure
import pandas as pd
from ..utils import generic

class Boundary:
    
    def __init__(self,boundary_id,p1,p2,coords,mask_shape):
        self.id=boundary_id
        self.p1=p1
        self.p2=p2
        self.neighbors = np.array([p1,p2])
        self.p1coords=None
        self.p2coords=None
        self.stiched_coords=None
        self.stiched_mask=None
        self.stiched_regionprop=None
        self.coords=coords
        self.data=None
        self.bbox=None
        self.shape = mask_shape
        self.neighbor_regionprops=None
        
        
    def stich_neighbors(self,regionprops):
        
        
        self.p1coords,self.p2coords = regionprops.loc[self.neighbors]['$coords'].values
        self.coords = np.array(self.coords)
        self.stiched_coords = np.vstack([self.p1coords,
                                         self.p2coords,
                                         self.coords])
        self.neighbor_regionprops = regionprops.loc[self.neighbors].copy()
        x0 = max(min(self.neighbor_regionprops['$bbox-0'].min(),min(self.coords[:,0]))-1,0)
        y0 = max(min(self.neighbor_regionprops['$bbox-1'].min(),min(self.coords[:,1]))-1,0)
        x1 = min(max(self.neighbor_regionprops['$bbox-2'].max(),max(self.coords[:,0]))+1,self.shape[0])
        y1 = min(max(self.neighbor_regionprops['$bbox-3'].max(),max(self.coords[:,1]))+1,self.shape[1])
        self.bbox = (x0,x1,y0,y1)
        
        self.stiched_mask=np.zeros((x1-x0,y1-y0))
        self.stiched_mask[self.stiched_coords[:,0]-x0,
                          self.stiched_coords[:,1]-y0]=1
        self.stiched_regionprop=measure.regionprops(self.stiched_mask.astype(int))[0]
    
    def measure(self,regionprops,feature_images={}):
        
        # compute basic features
        p_length=estimate_boundary_length(self.coords)
        p_max_perimeter_fraction = p_length/np.min(self.neighbor_regionprops['perimeter'].values)
        neighbor_orientations = self.neighbor_regionprops['orientation'].values
        p_neighbor_orientation_diff = np.abs(neighbor_orientations[0]-neighbor_orientations[1])
        p_stiched_orientation_diff = np.min(np.abs(self.stiched_regionprop.orientation-neighbor_orientations))
        p_size_differences = generic.norm_ratio(self.neighbor_regionprops['area'])
        p_stiched_length_fraction = self.stiched_regionprop.major_axis_length/self.neighbor_regionprops['major_axis_length'].sum()
        p_min_neighbor_size = np.min(self.neighbor_regionprops['area'])
        p_stiched_area = self.stiched_regionprop.area
        p_stiched_convexity = p_stiched_area/self.stiched_regionprop.convex_area
        p_stiched_eccentricity = self.stiched_regionprop.eccentricity
        p_stiched_solidity = self.stiched_regionprop.solidity
        p_stiched_extent = self.stiched_regionprop.extent
        
        basic_features = [self.id,
                          p_length,
                          p_max_perimeter_fraction,
                          p_neighbor_orientation_diff,
                          p_stiched_orientation_diff,
                          p_size_differences,
                          p_min_neighbor_size,
                          p_stiched_length_fraction,
                          p_stiched_area,
                          p_stiched_convexity,
                          p_stiched_eccentricity,
                          p_stiched_solidity,
                          p_stiched_extent]
        basic_feature_columns = ['boundary_id','boundary_length','max_perimeter_fraction','neighbor_orientation_diff',
                                 'stiched_orientation_diff','neighbor_size_difference','min_neighor_size',
                                 'stiched_length_fraction','stiched_size','stiched_convexity','stiched_eccentricity',
                                 'stiched_solidity','stiched_extent']
        
        intensity_features = []
        intensity_columns = []
        if len(feature_images)>0:
            for name,val in feature_images.items():
                b_val = val[self.coords[:,0],self.coords[:,1]]
                intensity_features+=[b_val.mean(),b_val.std(),b_val.max(),b_val.min()]
                intensity_columns+=['mean_{}'.format(name),
                                    'std_{}'.format(name),
                                    'max_{}'.format(name),
                                    'min_{}'.format(name),]
        columns = basic_feature_columns+intensity_columns
        boundary_measures = basic_features+intensity_features
        self.data = pd.DataFrame([boundary_measures],columns=columns)
        
        
def boundary_neighbor_pairwise(watershed, boundary_mask, connectivity=1):
    particle_pairwise_dict = {}
    boundary_pairwise_dict = {}
    xlist, ylist = np.where(boundary_mask > 0)
    xmax, ymax = watershed.shape
    n = 0
    unique_neighbors = []
    for i in range(len(xlist)):
        x, y = xlist[i], ylist[i]
        # find unique neighbors
        if connectivity == 1:
            m = np.unique([watershed[max(x - 1, 0), y], 
                           watershed[min(x + 1, xmax - 1), y],
                           watershed[x, max(y - 1, 0)], 
                           watershed[x, min(ymax - 1, y + 1)]])
        elif connectivity == 2:
            m = np.unique(watershed[max(x - 1, 0):min(x + 1, xmax - 1), max(y - 1, 0):max(y - 1, 0)].flatten())

        # discard background pixels
        unique_neighbors = m[m>0]
        if len(unique_neighbors) > 0:
            for pair in combinations(unique_neighbors, 2):
                p1, p2 = pair
                if p1 not in particle_pairwise_dict:
                    particle_pairwise_dict[p1] = {p2: n}
                    n += 1
                elif p2 not in particle_pairwise_dict[p1]:
                    particle_pairwise_dict[p1][p2] = n
                    n += 1
                idx = particle_pairwise_dict[p1][p2]
                if idx not in boundary_pairwise_dict:
                    boundary_pairwise_dict[idx] = Boundary(idx, p1,p2,[[x,y]],watershed.shape)
                else:
                    boundary_pairwise_dict[idx].coords += [[x, y]]
    return particle_pairwise_dict, boundary_pairwise_dict


def deflection_angle(theta1, theta2):
    return min(abs(theta1 + theta2), abs(theta1 - theta2))


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