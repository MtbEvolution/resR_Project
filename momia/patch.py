__author__ = "jz-rolling"

import numpy as np
import tifffile
import nd2reader as nd2
from .helper_image import *
import pickle as pk
from .segmentation import *
from .optimize import *
from .particle import Particle

Version = '0.2.2'


class Patch:

    def __init__(self):

        # make sure that the number of images match the number of channels provided.
        # modularize Image class
        self.id = 0
        self.data = {}
        self.channels = []
        self.ref_channel = ''
        self.config = None
        self.pixel_microns = 0.065
        self.shape = (2048, 2048)
        self.binary = None
        self.contours = {}
        self.cell_dict = {}
        self._cluster_categories = None
        self.RoG = None
        self.Frangi = None
        self.DoG = None
        self.shapeindex = None
        self.bbox=()
        self.predictions = None
        self.ridge=None
        self._annotated_seeds = None
        self._seed_watershed = None
        self._binary_from_seed = None
        self._pixel_annotation = None
        self._pixel_features = None
        self._pixel_features_proba = None
        self._masked_coords = None

    def load_data(self,
                  image_id,
                  image_dict,
                  ref_channel=-1):
        self.id = image_id
        self.data = image_dict
        self.channels = list(image_dict.keys())
        if isinstance(ref_channel, int):
            self.ref_channel = self.channels[ref_channel]
        elif isinstance(ref_channel, str):
            self.ref_channel = ref_channel
        else:
            raise ValueError('reference channel should be integer or string.')
        self.shape = self.data[self.ref_channel].shape

    def load_config(self,
                    config):
        self.config=config

    def inherit(self, image_id, parent_obj, bbox, binary):
        self.id = image_id
        self.bbox = bbox
        self.binary = binary
        self.ref_channel = parent_obj.ref_channel
        self.data = parent_obj._get_roi_data(bbox)
        self.pixel_microns = parent_obj.pixel_microns
        self.shape = self.binary.shape
        self.config = parent_obj.config

        x1, y1, x2, y2 = bbox
        if parent_obj.RoG is not None:
            self.RoG = parent_obj.RoG[x1:x2, y1:y2].copy()
        if parent_obj.DoG is not None:
            self.DoG = parent_obj.DoG[x1:x2, y1:y2]
        if parent_obj.Frangi is not None:
            self.Frangi = parent_obj.Frangi[x1:x2, y1:y2]
        if parent_obj.shapeindex is not None:
            self.shapeindex = parent_obj.shapeindex[x1:x2, y1:y2]
        if parent_obj.config is not None:
            self.config = parent_obj.config

    def crop_edge(self):
        from skimage import registration
        cropped = float(self.config['image']['crop_edge'])
        offset_correction = bool(int(self.config['image']['offset_correction']))
        max_drift = float(self.config['image']['maximum_xy_drift'])

        if 0 <= cropped < 0.4:
            crop_width = int(cropped * self.shape[1])
            crop_height = int(cropped * self.shape[0])
            w1, w2 = crop_width, self.shape[1] - crop_width
            h1, h2 = crop_height, self.shape[0] - crop_height
            self.shape = (h2 - h1, w2 - w1)
        else:
            raise ValueError('Edge fraction should be no higher than 0.4 (40% from each side)!')

        if offset_correction:
            reference_image = self.data[self.ref_channel]
            reference_image = 100 + reference_image.max() - reference_image
            for channel, data in self.data.items():
                if channel != self.ref_channel:
                    shift, error, _diff = registration.phase_cross_correlation(reference_image, data,
                                                                               upsample_factor=10)
                    if max(np.abs(shift)) <= max_drift:
                        offset_image = shift_image(data, shift)
                        self.data[channel] = offset_image[h1:h2, w1:w2]
                    else:
                        self.data[channel] = self.data[channel][h1:h2, w1:w2]
                else:
                    self.data[channel] = self.data[channel][h1:h2, w1:w2]

        else:
            for channel, data in self.data.items():
                self.data[channel] = self.data[channel][h1:h2, w1:w2]

    def enhance_brightfield(self, normalize=True, gamma=1.0, adjust_gamma=False):

        """
        suppresses signal aberrations in phase contrast image using FFT bandpass filters
        :param normalize: adjust exposure of brightfield image
        :param gamma: user specified gamma correction value, default = 1

        """
        perform_bandpass_correction = bool(int(self.config['image']['bandpass']))
        ref_img = self.data[self.ref_channel].copy()
        if perform_bandpass_correction:

            ref_fft = fft(ref_img, subtract_mean=True)
            fft_filters = bandpass_filter(pixel_microns=self.pixel_microns,
                                          img_width=self.shape[1], img_height=self.shape[0],
                                          high_pass_width=float(self.config['image']['bandpass_high']),
                                          low_pass_width=float(self.config['image']['bandpass_low']))
            ref_img = fft_reconstruction(ref_fft, fft_filters)
            if normalize:
                ref_img = adjust_image(ref_img, adjust_gamma=adjust_gamma, gamma=gamma)
            self.data[self.ref_channel] = ref_img
            del ref_fft, fft_filters, ref_img

    def enhance_fluorescence(self,
                             normalize=False,
                             adjust_gamma=False,
                             gamma=1.0,
                             method='rolling_ball'):
        """
        remove fluorescence background using the rolling ball method
        :param normalize: adjust exposure and data depth
        :param adjust_gamma: apply user specified gamma correction if True
        :param gamma: user specified gama correction value, default = 1
        :return:
        """
        subtract_background = bool(int(self.config['image']['subtract_background']))
        if subtract_background:
            for channel, img in self.data.items():
                if channel != self.ref_channel:
                    img = background_subtraction(img,method=method)
                    if normalize:
                        img = adjust_image(img, adjust_gamma=adjust_gamma, gamma=gamma)
                    self.data[channel] = (filters.gaussian(img, sigma=0.5) * 65535).astype(np.uint16)
                    del img

    def segmentation_basic(self):
        method, kwarg = self._get_binary_method()
        self.binary, \
        self.cluster_labels, \
        self.regionprop_table = patch_segmentation_basic(self.data[self.ref_channel],
                                                         method=method, **kwarg)
        self.regionprop_table = self.regionprop_table.set_index('label')
        self.regionprop_table[['opt-x1','opt-y1','opt-x2','opt-y2','touching_edge']] = optimize_bbox_batch(self.shape,
                                                                                           self.regionprop_table)

    def segmentation_shapeindex(self,
                                watershed_line=True):
        threshold = (int(self.config['image']['shapeindex_low_bound']),
                     int(self.config['image']['shapeindex_high_bound']))
        disk_radius = int(self.config['image']['opening_radius'])
        min_seed_size = int(self.config['image']['min_seed_size'])
        use_ridge = bool(int(self.config['image']['use_ridge']))
        sigmas = tuple([float(x) for x in self.config['segmentation']['sato_sigmas'].split(',')])
        if self.binary is None:
            method = self._get_binary_method()
            self.binary = make_mask(self.get_ref_image(),method=method, min_size=40)
        shapeindex_sigma = float(self.config['image']['shapeindex_sigma'])
        self.shapeindex = shape_index_conversion(self.get_ref_image(),
                                                 shapeindex_sigma=shapeindex_sigma)

        if use_ridge:
            self.ridge = filters.sato(self.get_ref_image(),
                                      sigmas= sigmas,
                                      mode='constant',
                                      black_ridges=False)
            target = self.ridge
        else:
            target = self.get_ref_image()

        self.cluster_labels, \
        self.regionprop_table = patch_segmentation_shapeindex(target,
                                                              self.binary,
                                                              self.shapeindex,
                                                              threshold,
                                                              disk_radius,
                                                              min_seed_size,
                                                              watershed_line,
                                                              ridge=self.ridge)
        self.regionprop_table = self.regionprop_table.set_index('label')
        self.regionprop_table[['opt-x1','opt-y1','opt-x2','opt-y2','touching_edge']] = optimize_bbox_batch(self.shape,
                                                                                            self.regionprop_table)

    def find_contours(self, level=0.1, dilation=True):
        labels = self.regionprop_table.index
        n_contours = []
        for label in labels:
            x1,y1,x2,y2 = self.regionprop_table.loc[label][['opt-x1','opt-y1','opt-x2','opt-y2']].values.astype(int)
            mask = (self.cluster_labels[x1:x2,y1:y2]==label).astype(np.int32)
            data = self.get_ref_image()[x1:x2,y1:y2]
            contours = find_contour_marching_squares(data, mask,
                                                     level=level,
                                                     dilation=dilation)
            n_contours.append(len(contours))
            self.contours[label] = contours
        self.regionprop_table['n_contours'] = n_contours

    def get_ref_image(self):
        return self.data[self.ref_channel]

    def _measure_RoG(self,s1=0.5,s2=5):
        basic_measures = ['weighted_moments_hu','mean_intensity',
                          'min_intensity','max_intensity']
        if self.RoG is None:
            self.RoG = ratio_of_gaussian(self.get_ref_image(),s1,s2)
        rog_measures = pd.DataFrame(measure.regionprops_table(self.cluster_labels,
                                                              intensity_image=self.RoG,
                                                              properties=basic_measures,
                                                              extra_properties=(std, skewness, kurtosis))).values
        columns = ['RoG_weighted_hu-{}'.format(x) for x in range(0,7)] +\
                  ['RoG_mean','RoG_min','RoG_max','RoG_std','RoG_skewness','RoG_kurtosis']
        self.regionprop_table[columns] = rog_measures

    def _measure_contour(self):
        contour_data=[]
        for label, contours in self.contours.items():
            contour_angles = bend_angle(contours[0], window=3)
            contour_data.append([np.std(contour_angles),
                                 np.percentile(contour_angles,90),
                                 np.percentile(contour_angles,10),
                                 np.max(contour_angles),
                                 np.min(contour_angles),
                                 stats.skew(contour_angles),
                                 np.median(contour_angles)])
        self.regionprop_table[['bending_std','bending_90','bending_10',
                               'bending_max','bending_min','bending_skewness',
                               'bending_median']]=contour_data

    def _measure_Frangi(self):
        basic_measures = ['weighted_moments_hu', 'mean_intensity',
                          'min_intensity', 'max_intensity']
        if self.Frangi is None:
            self.Frangi = filters.frangi(self.get_ref_image(), scale_step=5)
        frangi_measures = pd.DataFrame(measure.regionprops_table(self.cluster_labels,
                                                                 intensity_image=self.Frangi,
                                                                 properties=basic_measures,
                                                                 extra_properties=(std, skewness, kurtosis))).values
        columns = ['Frangi_weighted_hu-{}'.format(x) for x in range(0, 7)] + \
                  ['Frangi_mean', 'Frangi_min', 'Frangi_max',
                   'Frangi_std', 'Frangi_skewness', 'Frangi_kurtosis']
        self.regionprop_table[columns] = frangi_measures

    def _measure_shapeindex(self, shapeindex_sigma=1):
        basic_measures = ['weighted_moments_hu', 'mean_intensity',
                          'min_intensity', 'max_intensity']
        if self.shapeindex is None:
            self.shapeindex = shape_index_conversion(self.get_ref_image(), shapeindex_sigma=shapeindex_sigma)
        shapeindex_measures = pd.DataFrame(measure.regionprops_table(self.cluster_labels,
                                                                     intensity_image=self.shapeindex,
                                                                     properties=basic_measures,
                                                                     extra_properties=(std, skewness, kurtosis))).values
        columns = ['Shapeindex_weighted_hu-{}'.format(x) for x in range(0, 7)] + \
                  ['Shapeindex_mean', 'Shapeindex_min', 'Shapeindex_max',
                   'Shapeindex_std', 'Shapeindex_skewness', 'Shapeindex_kurtosis']
        self.regionprop_table[columns] = shapeindex_measures

    def _measure_DoG(self,s1=0.5,s2=10):
        basic_measures = ['weighted_moments_hu', 'mean_intensity',
                          'min_intensity', 'max_intensity']
        if self.DoG is None:
            self.DoG = normalized_difference_of_gaussian(self.get_ref_image(), s1, s2)
        dog_measures = pd.DataFrame(measure.regionprops_table(self.cluster_labels,
                                                              intensity_image=self.DoG,
                                                              properties=basic_measures,
                                                              extra_properties=(std, skewness, kurtosis))).values
        columns = ['DoG_weighted_hu-{}'.format(x) for x in range(0, 7)] + \
                  ['DoG_mean', 'DoG_min', 'DoG_max', 'DoG_std', 'DoG_skewness', 'DoG_kurtosis']
        self.regionprop_table[columns] = dog_measures

    def _hu_moments_log_transform(self):
        for c in self.regionprop_table.columns.values:
            if 'hu' in c:
                self.regionprop_table[c] = hu_log10_transform(self.regionprop_table[c].values)

    def cluster_classification(self, classifier):
        self.regionprop_table['prediction'] = classifier.predict(self.regionprop_table)

    def get_cluster_roi(self,label):
        x1, y1, x2, y2 = self.regionprop_table.loc[label][['opt-x1', 'opt-y1', 'opt-x2', 'opt-y2']].values.astype(int)
        return (x1,y1,x2,y2)

    def get_cluster_data(self,label,channel):
        x1,y1,x2,y2 = self.get_cluster_roi(label)
        return self.data[channel][x1:x2,y1:y2]

    def _get_roi_data(self,roi):
        x1, y1, x2, y2 = roi
        return {c:d[x1:x2, y1:y2] for c,d in self.data.items()}

    def _get_cluster_mask(self,label):
        x1, y1, x2, y2 = self.get_cluster_roi(label)
        mask = (self.cluster_labels[x1:x2, y1:y2] == label) * 1
        return mask

    def _predict(self, classifier=None):
        if classifier is not None:
            self.predictions = classifier.predict(self.regionprop_table).astype(int)
        else:
            self.predictions = np.zeros(len(self.regionprop_table))

    def _get_class_index(self, class_label=0):
        return self.regionprop_table.index[np.where(self.predictions==class_label)].values

    def _get_binary_method(self):
        if int(self.config['segmentation']['binary_method']) == 0:
            method='isodata'
        elif int(self.config['segmentation']['binary_method']) == 1:
            method='sauvola'
        elif int(self.config['segmentation']['binary_method']) == 2:
            method='legacy'
        elif int(self.config['segmentation']['binary_method']) == 3:
            method='combined'

        kwarg = {'window_size':int(self.config['segmentation']['sauvola_window_size']),
                 'k':float(self.config['segmentation']['sauvola_k']),
                 'min_size':float(self.config['segmentation']['min_size']),
                 'block_sizes':np.array(self.config['segmentation']['block_sizes'].split(',')).astype(float),
                 'binary_opening':bool(int(self.config['segmentation']['binary_opening']))}
        return method, kwarg

    def _adopt_seeds(self,seed):
        if seed.shape != self.shape:
            raise ValueError("Seed image shape doesn't match")
        self._annotated_seeds = measure.label(seed)

    def _seed2annotation(self, overwrite=True):
        for i, k in enumerate(self.regionprop_table.index):
            x, y = np.array(self.regionprop_table.loc[k, 'coords']).T
            members = np.unique(self._annotated_seeds[x, y])
            if members[0] == 0:
                if len(members) == 1:
                    self.predictions[i] = 2
                elif len(members) > 2:
                    self.predictions[i] = 1
                elif len(members) == 2:
                    self.predictions[i] = 0
            else:
                if len(members) == 1:
                    self.predictions[i] = 0
                elif len(members) > 1:
                    self.predictions[i] = 1
        if overwrite:
            self.regionprop_table['annotation'] = self.predictions
