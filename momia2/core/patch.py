import os, glob
import numpy as np
import matplotlib.pyplot as plt
from ..utils import bandpass, generic, Config, correction, contour
from ..segmentation import MakeMask, BinarizeLegend
from ..classify import prediction2foreground,prediction2seed
from ..gui import Labeler
from scipy import ndimage as ndi
from skimage.filters import sato
from skimage.morphology import binary_dilation, binary_erosion, binary_opening
from skimage.segmentation import watershed
from skimage.measure import label, regionprops_table
from ..metrics import multiscale_image_feature, local_stat, get_regionprop
import pandas as pd
import warnings
import PIL,io

class Patch:

    def __init__(self,store_backup = False):
        # 
        # make sure that the number of images match the number of channels provided.
        # modularize Image class
        self.id = 0
        self.data = {}
        self._data = {}
        self.seed = None
        self.labeled_seeds = None
        self.labeled_mask = None
        self.config = None
        self.mask = None
        self.prob_mask = None
        self.labels = None
        self.pixel_microns = 0.065
        self.shape = (2048, 2048)
        self.channels = []
        self.ref_channel = ''
        self.contours = {}
        self.bbox=()
        self.cache = PatchCache()
        self.store_backup = store_backup
        self.regionprops = pd.DataFrame()
        self.boundaries = None
        
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
        if self.store_backup:
            self._data = self.data.copy()
        
    def revert(self):
        if self.store_backup:
            self.data = self._data.copy()
        else:
            print('No raw data found. Set "store_backup" to True to backup raw data.')

    def load_config(self, config):
        if 'ConfigParser' in str(type(config)):
            self.config = config
        else:
            self.config = Config(config).config
            
    def preprocess_images(self, optimizers=[]):
        for opt in optimizers:
            opt.optimize(self)
    
    def load_seed(self,seed):
        if seed.shape != self.shape:
            raise ValueError("The shapes of seed image and target images don't match!")
        self.seed = seed
       
    def crop_edge(self):
        cropped = float(self.config['image']['crop_edge'])

        if 0 <= cropped < 0.4:
            crop_width = int(cropped * self.shape[1])
            crop_height = int(cropped * self.shape[0])
            w1, w2 = crop_width, self.shape[1] - crop_width
            h1, h2 = crop_height, self.shape[0] - crop_height
            self.shape = (h2 - h1, w2 - w1)
        else:
            raise ValueError('Edge fraction should be no higher than 0.4 (40% from each side)!')
        
        # crope edge
        for channel, data in self.data.items():
            self.data[channel] = self.data[channel][h1:h2, w1:w2]
    
    def correct_xy_drift(self, reference_channel = 'default'):
        from skimage import registration
        
        # define max drift in pixel unit
        max_drift = float(self.config['image']['maximum_xy_drift'])
        invert_ref = bool(int(self.config['image']['invert_reference']))
        
        if reference_channel == 'default':
            reference_channel = self.ref_channel
            reference_image = self.get_ref_image()
        elif reference_channel in list(self.data.keys()):
            reference_image = self.data[reference_channel]
        else:
            raise ValueError('Channel {} not found!'.format(reference_channel))
        
        # invert image if necessary
        if invert_ref:
            reference_image = 100 + reference_image.max() - reference_image
        
        # correct xy drift by phase cross correlation
        for channel, data in self.data.items():
            if channel != reference_channel:
                shift, error, _diff = registration.phase_cross_correlation(reference_image,
                                                                           data,
                                                                           upsample_factor=10)
                if max(np.abs(shift)) <= max_drift:
                    offset_image = correction.shift_image(data, shift)
                    self.data[channel] = offset_image
    
    def generate_mask(self, model='default', channels=[], **kwargs):
        if len(channels)>len(self.channels):
            raise ValueError('Patch image had {} channels but {} channels were provided'.format(len(self.channels),
                                                                                                len(channels)))
        if len(channels)==0:
            input_images = [self.get_ref_image()]
    
        elif len(channels)>=1:
            stack = []
            for c in channels:
                if c in self.data:
                    stack.append(self.data[c])
                else:
                    raise ValueError('Patch image does not have channel "{}".'.format(c))
            input_images = [np.array(stack)]

        if isinstance(model, str):
            if model=='default':
                mask_method = int(self.config['segmentation']['binary_method'])
                model = BinarizeLegend(method=mask_method, config=self.config)
            else:
                raise ModuleNotFoundError('Mask method {} not found!'.format(model))
        mask = MakeMask(model, **kwargs).predict(input_images)[0]
        self.mask = mask
        
    def pixel_classification(self,
                             pixel_classifier,
                             cache_in = False,
                             seed_prob_min=0.5,
                             edge_prob_max=0.8,
                             min_seed_size=5,
                             background_prob_max=0.3,
                             foreground_erosion_radius=0):
        classifier_output = pixel_classifier.predict(self.get_ref_image(),
                                                     return_feature_images=cache_in)
    
        self.prob_mask = classifier_output['Probability_map']
        if cache_in:
            self.cache.feature_images = classifier_output['Feature_images']
        self.seed = prediction2seed(self.prob_mask,
                                    seed_min=seed_prob_min,
                                    edge_max=edge_prob_max,
                                    min_seed_size=min_seed_size)
        self.mask = prediction2foreground(self.prob_mask,
                                          threshold=background_prob_max,
                                          erosion_radius=foreground_erosion_radius)
        self.seed *= self.mask
        self.labeled_mask = watershed(image=self.prob_mask[:,:,1],
                                      mask=self.mask,
                                      markers=self.seed,
                                      connectivity=1,
                                      compactness=0.1,
                                      watershed_line=True)
        
    def locate_particles(self, 
                         use_intensity=True, 
                         precompute_contours=False, 
                         contour_kwargs={'level':0.1, 'dilation':True}):
        if self.labeled_mask.max()>0:
            if use_intensity:
                self.regionprops=get_regionprop(self.labeled_mask, intensity_image=self.get_ref_image())
            else:
                self.regionprops=get_regionprop(self.labeled_mask, intensity_image=None)
            self.regionprops.set_index('$label', inplace=True)
            self.regionprops['$image_id'] = [self.id]*len(self.regionprops)
            self.regionprops[['$opt-x1','$opt-y1','$opt-x2','$opt-y2','$touching_edge']] = correction.optimize_bbox_batch(self.shape,
                                                                                                                          self.regionprops)
            if precompute_contours:
                self._find_contours(**contour_kwargs)

            
    def render_image_features(self, model='default', mode='default'):
        if isinstance(model, str):
            if (model=='default') & (len(self.cache.feature_columns) == 0):
                self._image2pixelfeatures(mode=mode)
        else:
            model.predict(self)
        return None
            
    def classify_particles(self, model=None):
        return None
        
    def subdivde_particles(self,model):
        return None
        
    def annot2feature(self):
        use_Gaussian = bool(int(self.config['classification_pixel']['use_Gaussian']))
        pixel_feature_selem = generic.config2selem(str(self.config['classification_pixel']['pixel_feature_selem']))
        self.cache.extract_local_features(use_Gaussian=use_Gaussian, selem=pixel_feature_selem)
        
    def mask2feature(self, ):
        coords = np.where(self.mask>0)
        self.cache.pixel_data['xcoord'] = coords[0]
        self.cache.pixel_data['ycoord'] = coords[1]
        self.annot2feature()
        
    def pixel2label(self, classifier):
        
        # ridge
        ridge_sigma = float(self.config['classification_pixel']['ridge_sigma'])
        opening = bool(int(self.config['classification_pixel']['open_before_label']))
        ridge = sato(self.get_ref_image(),
                     sigmas=(ridge_sigma),
                     black_ridges=False)
        
        xcoord = self.cache.pixel_data['xcoord']
        ycoord = self.cache.pixel_data['ycoord'] 
        canvas = np.zeros(self.shape)
        prediction = classifier.predict(self.pixel_data())
        canvas[xcoord,ycoord] = (prediction==2)*1
        canvas *= self.mask
        if opening:
            canvas = binary_opening(canvas)
        
        self.labels = watershed(ridge,
                                markers=label(canvas),
                                mask=self.mask,
                                watershed_line=True,
                                compactness=1)
        
    def pixel_data(self):
        return self.cache.pixel_data
    
    def get_ref_image(self):
        return self.data[self.ref_channel]
    
    def get_channel_data(self,channel):
        return self.data[channel]
    
    def _sample_window(self,
                       window_size=(256,256),
                       filename=None):
        return None
    
    def _find_contours(self, level=0.1, dilation=True):
        if len(self.regionprops)>0:
            contour_list = []
            n_contours = []
            if '$opt-x1' not in self.regionprops:
                self.regionprops[['$opt-x1','$opt-y1','$opt-x2','$opt-y2','$touching_edge']] = correction.optimize_bbox_batch(self.shape,
                                                                                                                         self.regionprops)
            for label in self.regionprops.index:
                x1,y1,x2,y2 = self.regionprops.loc[label][['$opt-x1','$opt-y1','$opt-x2','$opt-y2']].values.astype(int)
                mask = (self.labeled_mask[x1:x2,y1:y2]==label).astype(np.int32)
                mask = ndi.median_filter(mask,2)>0.5
                data = self.get_ref_image()[x1:x2,y1:y2]
                contours = contour.find_contour_marching_squares(data, 
                                                                 mask,
                                                                 level=level,
                                                                 dilation=dilation)
                n_contours.append(len(contours))
                contour_list.append(contours)
            self.regionprops['n_contours'] = n_contours
            self.regionprops['$contours'] = contour_list
    
    def _image2pixelfeatures(self,mode='default'):
        if mode == 'default':
            sigmas = (0.5, 1.5, 5)
            use_RoG = True
            use_Ridge = True
            use_Sobel = False
            use_Gaussian = True
            use_Shapeindex = True
            num_workers=12
        elif mode == 'use_config':
            # adopt parameters from configurations
            sigmas = np.array(self.config['classification_pixel']['sigmas'].split(',')).astype(float)
            use_RoG = bool(int(self.config['classification_pixel']['use_RoG']))
            use_Ridge = bool(int(self.config['classification_pixel']['use_Ridge']))
            use_Sobel = bool(int(self.config['classification_pixel']['use_Sobel']))
            use_Gaussian = bool(int(self.config['classification_pixel']['use_Gaussian']))
            use_Shapeindex = bool(int(self.config['classification_pixel']['use_Shapeindex']))
            num_workers = int(self.config['classification_pixel']['num_workers'])
        elif mode == 'fast':
            sigmas = (float(self.config['classification_pixel']['ridge_sigma']))
            use_RoG = True
            use_Ridge = True
            use_Sobel = False
            use_Gaussian = False
            use_Shapeindex = True
            num_workers=12
        else:
            raise ValueError('Illegal pixel feature extraction mode "{}", use "default", "fast", "use_config" instead.'.format(mode))
        # feature images
        self.cache.prepare_feature_images(self.get_ref_image(),
                                          num_workers=num_workers,
                                          sigmas = sigmas,
                                          rog=use_RoG,
                                          ridge=use_Ridge,
                                          sobel=use_Sobel,
                                          shapeindex=use_Shapeindex)
        
    def _seed2annot(self):
        from skimage.segmentation import watershed
        from skimage.measure import regionprops_table,label
        
        # label_seed
        self.labeled_seeds = label(self.seed)
        
        # adopt parameters from configurations
        ridge_sigma = float(self.config['classification_pixel']['ridge_sigma'])
        min_seed_fraction = float(self.config['classification_pixel']['min_seed_fraction'])
        erosion_selem = generic.config2selem(str(self.config['classification_pixel']['seeded_mask_erosion_selem']))
        dilation_selem = generic.config2selem(str(self.config['classification_pixel']['seeded_mask_dilation_selem']))
        
        # watershed segmentation
        ridge = sato(self.get_ref_image(),
                     sigmas=(ridge_sigma),
                     black_ridges=False)
        labeled_mask = watershed(ridge,markers=self.labeled_seeds,
                                 mask=self.mask,
                                 watershed_line=True,
                                 compactness=1)
        rp = pd.DataFrame(regionprops_table(labeled_mask,
                                            intensity_image=(self.seed>0)*1,
                                            properties=['coords','mean_intensity']))
        if rp['mean_intensity'].max()<min_seed_fraction:
            x,y = np.vstack(rp[rp['mean_intensity']<min_seed_fraction]['coords'].values).T
            labeled_mask[x,y]=0
        self.labeled_mask = labeled_mask
        
        # extract relavent pixel coordinates 
        seeded_mask = (labeled_mask>0)*1
        core = binary_erosion(seeded_mask,erosion_selem)*1
        dilated = binary_dilation(seeded_mask,dilation_selem)*1
        residual = ((self.mask-dilated)==1)*1
        edge = dilated-core
        
        # mask to coords
        core_coords = np.array(np.where(core>0))
        edge_coords = np.array(np.where(edge>0))
        residual_coords = np.array(np.where(residual>0))
        
        # concat coords and annotations
        coords = np.hstack([core_coords, edge_coords,residual_coords])
        annot = [2]*len(core_coords[0])+\
                [1]*len(edge_coords[0])+\
                [0]*len(residual_coords[0])
        self.mask = np.zeros(self.mask.shape)
        self.mask[edge==1]=1
        self.mask[core==1]=2

        self.cache.pixel_data['xcoord'] = coords[0]
        self.cache.pixel_data['ycoord'] = coords[1]
        self.cache.pixel_data['annotation'] = annot
    
    def annotate_particles(self,
                       percentile_norm_params=(0,100,2000,10000),
                       show_contour=False,
                       contour_color='r',
                       classes = {1:'single cell', 2:'microcolony', 3:'others'},
                       saturation=0.85):
        norm = (generic.percentile_normalize(self.get_ref_image(),
                                             percentile_norm_params[0],
                                             percentile_norm_params[1],
                                             percentile_norm_params[2],
                                             percentile_norm_params[3])*int(saturation*255)).astype(np.uint8)
        thumbnails = []
        if '$annotation' in self.regionprops.columns:
            annotation = self.regionprops['$annotation'].values
        else:
            annotation = np.zeros(len(self.regionprops))
        for (x0,y0,x1,y1,c) in self.regionprops[['$opt-x1','$opt-y1','$opt-x2','$opt-y2','$opt-contours']].values:
            if show_contour:
                fig=plt.figure(figsize=(4,4))
                plt.imshow(self.get_ref_image()[x0:x1,y0:y1],cmap='gist_gray')
                plt.plot(c[0][:,1],c[0][:,0],color='r',lw=0.8)
                plt.axis('off')
                img_buf = io.BytesIO()
                plt.savefig(img_buf,bbox_inches='tight')
                thumbnails.append(PIL.Image.open(img_buf))
                plt.close()
            else:
                thumbnails.append(PIL.Image.fromarray(norm[x0:x1,y0:y1]))
        Labeler(thumbnails, label_dict=classes, annot=annotation,)
        self.regionprops['$annotation']=annotation
                
class PatchCache:
    
    def __init__(self):
        self.gaussians = {}
        self.feature_images = {}
        self.feature_columns = []
        self.pixel_data = pd.DataFrame()
        
    def prepare_feature_images(self, img, **kwargs):
        self.gaussians, self.feature_images = multiscale_image_feature(img, **kwargs)
        
    def extract_local_features(self, use_Gaussian=True, **kwargs):
        if len(self.pixel_data) == 0:
            raise ValueError('Pixel coordinates not defined!')
        xcoord,ycoord = self.pixel_data[['xcoord','ycoord']].values.T
        feature_images  = self.feature_images
        if use_Gaussian:
            for k,g in self.gaussians.items():
                feature_images['Gaussian_{}'.format(k)]=g
        local_pixel_features = local_stat(feature_images,xcoord,ycoord,**kwargs)
        self.feature_columns = local_pixel_features.columns
        self.pixel_data[self.feature_columns] = local_pixel_features.values