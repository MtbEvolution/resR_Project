from ..utils import contour
from scipy.ndimage import median_filter
from skimage import morphology, measure, filters, draw
import numpy as np



class Particle:
    
    def __init__(self,
                 bbox, #x1,y1,x2,y2
                 mask_coords, # shape ~(N,2)
                 pixel_microns=0.065):
        self.bbox=bbox
        self.shape = (bbox[2]-bbox[0],bbox[3]-bbox[1])
        self.mask = np.zeros(self.shape)
        x=mask_coords[:,0]-bbox[0]
        y=mask_coords[:,1]-bbox[1]
        x[x<0]=0
        x[x>self.shape[0]-1]=self.shape[0]-1
        y[y<0]=0
        y[y>self.shape[1]-1]=self.shape[1]-1
        self.mask[x,y]=1
        self.init_mask_coords = mask_coords
        self.optimized_coords = None
        self.contours = None
        self.optimized_contours = None
        self.midlines = None
        self.data = {}
        self.length = 0
        
    def find_contours(self,
                       target_image=None,
                       dark_background=False,
                       sigma=2,
                       level=0.1):
        self._smooth_mask()
        if target_image is None:
            target_image = self.mask
            dark_background=True
        elif target_image.shape != self.shape:
            target_image = target_image[self.bbox[0]:self.bbox[2],self.bbox[1]:self.bbox[3]]
        contours = contour.find_contour_marching_squares(target_image,self.mask,
                                                                  dark_background=dark_background,
                                                                  level=level,
                                                                  gaussian_smooth_sigma=sigma)
        self.contours = contours
    
    def _smooth_mask(self,
                     median_filter_radius=5,
                     opening_radius=1,
                     max_hole_size=100):
        self.mask = morphology.binary_opening(self.mask,morphology.disk(opening_radius))
        self.mask = median_filter(self.mask,median_filter_radius)>0.5
        self.mask = morphology.remove_small_holes(self.mask, max_hole_size)
        
        
    def optimize_contours(self,target,**kwargs):
        self.optimized_contours = contour.optimize_contour(self.contours,target,**kwargs)
        new_mask = draw.polygon2mask(self.shape,self.optimized_contours[0])
        _x,_y = np.where(new_mask>0)
        self.optimized_coords = np.array([_x+self.bbox[0],_y+self.bbox[1]]).T
    
    def find_midlines(self,interp_dist=4,smooth_factor=1,min_fragment_length=1):
        self.midlines,self.length = contour.contour2midlines(self.optimized_contours[0],
                                                             interpolation_distance=interp_dist,
                                                             smooth_factor=smooth_factor,
                                                             min_fragment_length=min_fragment_length)
        
    def inherit_data(self,parent_patch):
        x1,y1,x2,y2 = self.bbox
        for c,v in parent_patch.data.items():
            self.data[c] = v[x1:y1,x2:y2]
        self.ref_channel = parent_patch.ref_channel()
            
    def get_ref_image(self):
        return self.data[self.ref_channel]