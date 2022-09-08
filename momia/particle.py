from .helper_image import *


class Particle:

    def __init__(self,
                 bbox,
                 mask,
                 contours,
                 data,
                 ref_channel,
                 particle_id,
                 pixel_microns=0.065,
                 offset=(0, 0)):
        self.bbox = bbox
        self.init_contours = contours
        self.mask = mask
        self.x,self.y = np.where(mask>0)
        self.data = data
        self.ref_channel = ref_channel
        self.pixel_microns = pixel_microns
        self.id = particle_id
        self.sobel = None
        self.optimized_contours = {}
        self.skeleton = None
        self.skeleton_coords = {}
        self.branched = False
        self.discarded = False
        self.midlines = []
        self.lengths = []
        self.width_lists = []
        self.config = None
        self.offset = offset
        self.intensity_data={}
        self.basic_feature_attributes = ['area', 'eccentricity', 'solidity', 'touching_edge',
                                         'convex_area', 'filled_area', 'eccentricity', 'solidity',
                                         'major_axis_length', 'minor_axis_length', 'perimeter',
                                         'equivalent_diameter', 'extent','circularity', 'aspect_ratio',
                                         'moments_hu-0', 'moments_hu-1', 'moments_hu-2', 'moments_hu-3',
                                         'moments_hu-4', 'moments_hu-5', 'moments_hu-6', 'aspect_ratio',
                                         'bending_std', 'bending_90', 'bending_10', 'bending_max',
                                         'bending_min', 'bending_skewness', 'bending_median', 'touching_edge']
        self.basic_features = []

    def load_basic_features(self, parent_regionprop, basic_attributes):
        self.basic_feature_attributes = basic_attributes
        self.basic_features = parent_regionprop[basic_attributes].values

    def get_ref_image(self):
        return self.data[self.ref_channel]

    def optimize_contours(self, reinitiate=False):
        if reinitiate:
            self.init_contours = find_contour_marching_squares(self.get_ref_image(),
                                                               self.mask)
        sobel = filters.sobel(self.get_ref_image())
        self.optimized_contours = optimize_contour(contours=self.init_contours,
                                                   target=sobel)
        self.mask = draw.polygon2mask(self.mask.shape, self.optimized_contours[0])

    def extract_skeleton(self):
        try:
            self.skeleton_coords, self.skeleton = skeleton_analysis(self.mask)
            if len(self.skeleton_coords) == 0:
                self.discarded = True
            if len(self.skeleton_coords) > 1:
                self.branched = True
        except:
            self.discarded = True

    def find_midline(self, fast_mode=False):
        """
        Our current midline algorithm doesn't work very well for branched objects
        future versions may include a graph embedded midline system that accounts for
        the various shapes
        :param fast_mode: extend the termini of the morphological skeleton to the poles and call it a day

        """
        if not self.branched and not self.discarded:
            pole1, pole2 = self.skeleton_coords[0][0]
            xcoords = self.skeleton_coords[0][1]
            ycoords = self.skeleton_coords[0][2]
            if len(xcoords) <= 3:
                self.discarded = True
            else:
                if len(xcoords) <= 20:
                    interpolation_factor = 1
                else:
                    interpolation_factor = 2
                skel_coords = np.array([xcoords, ycoords]).T
                #
                smooth_skel = spline_approximation(skel_coords,
                                                   n=int(len(skel_coords)/interpolation_factor),
                                                   smooth_factor=5, closed=False)
                if not fast_mode:
                    smooth_skel, _converged = midline_approximation(smooth_skel,
                                                                    self.optimized_contours[0])
                    midline = extend_skeleton(smooth_skel, self.optimized_contours[0],
                                              find_pole1=pole1,
                                              find_pole2=pole2,
                                              interpolation_factor=interpolation_factor)

                else:
                    midline = extend_skeleton(smooth_skel, self.optimized_contours[0],
                                              find_pole1=pole1, find_pole2=pole2,
                                              interpolation_factor=1)

                width = direct_intersect_distance(midline, self.optimized_contours[0])
                length = line_length(midline)
                self.lengths.append(length*self.pixel_microns)
                self.midlines.append(midline)
                self.width_lists.append(width)


