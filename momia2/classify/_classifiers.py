from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle as pk
from ..metrics import multiscale_image_feature, local_stat
from ..utils import generic,dual_bandpass
from ..gui import Labeler
from ..core.boundary import boundary_neighbor_pairwise, Boundary
from skimage import filters,morphology,feature
import read_roi as rr
from matplotlib import pyplot as plt
import os,glob,cv2,datetime

class PixelClassifier:
     
    def __init__(self,
                 pixel_microns=0.065,
                 bandpass_struct_size=(0.01,10),
                 percentile_normalization_params = (0,99.5,500,None),
                 sigmas=(0.8,1,1.5,3.5),
                 mask_dilation_radius=5,
                 mask_hole_size_threshold=1000,
                 mask_mode = 'isodata',
                 use_gabor=False,
                 use_sobel=True,
                 use_shapeindex=True,
                 use_RoG=True,
                 ext_kernels={},
                 use_ridge=True,
                 local_stat_radius=0,
                 verbose=False,
                 patch_cache=True,
                 drop_before_dump=True,
                 prob2binary=False):
        """
        MOMIA2 implements both a conventional Weka-like semantic segmentaion tools (classify.PixelClassifier) as well as 
        several unet-based deep models (classify.DeepPixelClassifier, huge thanks to @DigitalSreeni and his marvelous youtube channel). 
        Input: manually annotated binary- or multilabel-masks and corresponding single-channel intensity images, 
        both under the same source folder.
        """
        self.pixel_microns = pixel_microns
        self.bandpass_struct = bandpass_struct_size
        self.sigmas = sigmas
        self.dilation_radius = mask_dilation_radius
        self.hole_size = mask_hole_size_threshold
        self.use_gabor = use_gabor
        self.use_sobel = use_sobel
        self.use_shapeindex = use_shapeindex
        self.use_RoG = use_RoG
        self.use_ridge = use_ridge
        self.verbose = verbose
        self.update_patch_cache = patch_cache
        self.date_created = generic.current_datetime()
        self.mask_mode = mask_mode
        self.training_images = []
        self.training_masks = []
        self.training_dataframe = []
        self._secondary_training_dataframe = []
        self.training_masklabels = []
        self.drop_before_dump = drop_before_dump
        self.perc_norm_params = percentile_normalization_params
        self.extra_kernels = ext_kernels
        self.local_stat_radius = local_stat_radius
        self.model=None
        self._seed_model_binary=None
        self._foreground_model_binary=None
        self.prob2binary=prob2binary 
        self._default_lgb_multiclass_params = {'learning_rate':0.02,
                                               'boosting_type':'gbdt',
                                               'objective':'multiclass',
                                               'metric':'multi_logloss',
                                               'num_leaves':200,
                                               'max_depth':30,
                                               'num_class':3,
                                               'n_iter':200,
                                               'num_threads':8,
                                               'num_boost_round':200}
        self._default_lgb_binary_params = {'learning_rate':0.02,
                                            'boosting_type':'gbdt',
                                            'objective':'binary',
                                            'metric':'binary_logloss',
                                            'num_leaves':100,
                                            'max_depth':20,
                                            'num_class':1,
                                            'n_iter':200,
                                            'num_boost_round':100}
        self._default_random_forrest_params = {'criterion':'entropy',
                                              'n_jobs':-1}
    
    def load_trainingset(self,
                         src_folder,
                         mask_identifier='_multilabel.tif',
                         image_identifier='.tif'):
        import glob,os
        mask_filenames = sorted(glob.glob('{}*{}'.format(src_folder,mask_identifier)))
        if self.verbose:
            print('{} mask files ending with "{}" are found in\n"{}".'.format(len(mask_filenames),
                                                                              mask_identifier,
                                                                              src_folder))
        for f in mask_filenames:
            # load data
            header = f.split('/')[-1].split(mask_identifier)[0]
            img=plt.imread(src_folder+'{}.tif'.format(header)).astype(float)
            mask = plt.imread(f)
            # extract features
            if self.mask_mode not in ['full','isodata']:
                user_xy = np.where(morphology.binary_dilation(mask>0,morphology.disk(self.dilation_radius))>0)
            else:
                user_xy = ()
            x,y,feature_df = self._image2features(img,user_xy=user_xy)
            self.training_dataframe.append(feature_df)
            self.training_masklabels.append(mask[x,y])
            self.training_masks.append(mask)
            self.training_images.append(img)
            
        if self.verbose:
            print('{} target images are found in\n"{}".'.format(len(mask_filenames), src_folder))
            if len(self.training_dataframe)>0:
                print("{} image features will be used to train the pixel classification model.".format(len(self.training_dataframe[0].columns)))
            
    def train_classifier(self,
                         params=None,
                         method='lgb',
                         test_size=0.4,
                         random_state=42):
        """
        
        """
        masks = np.hstack(self.training_masklabels)
        feature_df = pd.concat(self.training_dataframe)
        data = feature_df.values
        annotations = LabelEncoder().fit_transform(masks.reshape(-1))
        self.X_train, \
        self.X_test, \
        self.y_train, \
        self.y_test = train_test_split(data, annotations, test_size=test_size, random_state=random_state)
        
        if method in ['lgb','lightGBM','lGBM','LGBM']:
            if params is None:
                params = self._default_lgb_multiclass_params
            self.model = lgb.LGBMModel(random_state=random_state,
                                       verbose=int(self.verbose),
                                       **params)
            self._predict_func = self.model.predict
        elif method in ['RandomForrest','random_forrest','rf','RF']:
            if params is None:
                params = self._default_random_forrest_params
            self.model = RandomForestClassifier(random_state=random_state,
                                                verbose=int(self.verbose),
                                                **params)
            self._predict_func = self.model.predict_proba
        else:
            print('Method {} is currently not supported.'.format(method))
        if self.verbose:
            print('Model description:')
            print(self.model)
            print('Training...')
        t1 = datetime.datetime.now()
        self.model.fit(self.X_train,self.y_train)
        t2 = datetime.datetime.now()
        if self.verbose:
            print('Training time: {}'.format(t2-t1))
            
        # train additional classifierer to convert probablistic output into binary masks and seeds
        if self.prob2binary:
            self._train_binary_classifiers()
        
    def _train_binary_classifiers(self,method='lgb',
                                 test_size=0.4,
                                 random_state=42):
        foreground_labels = []
        seed_labels = []
        for i,mask in enumerate(self.training_masks): 
            user_xy = np.where(morphology.binary_dilation(mask>0,morphology.disk(self.dilation_radius)))
            img = self.training_images[i]
            canvas = np.zeros(list(img.shape)+[3])
            canvas[:,:,0]=1
            x,y,df = self._image2features(img,return_feature_images=False,user_xy=user_xy)
            canvas[x,y,:]=self._predict_func(df.values)
            _x,_y,bg_df=self._image2features(canvas[:,:,0],user_xy=user_xy,_force_ignore_selem=True)
            _x,_y,edge_df=self._image2features(canvas[:,:,1],user_xy=user_xy,_force_ignore_selem=True)
            _x,_y,core_df=self._image2features(canvas[:,:,2],user_xy=user_xy,_force_ignore_selem=True)
            self._secondary_training_dataframe.append(pd.concat([bg_df,edge_df,core_df],axis=1))
            foreground_labels += list(((mask>0)*1)[_x,_y])
            seed_labels += list(((mask==mask.max())*1)[_x,_y])
        self._secondary_training_dataframe = pd.concat(self._secondary_training_dataframe)
        fg_xtrain,fg_xtest,fg_ytrain,fg_ytest = train_test_split(self._secondary_training_dataframe.values,
                                                                 np.array(foreground_labels).reshape(-1,1),
                                                                 test_size=test_size, random_state=random_state)
        seed_xtrain,seed_xtest,seed_ytrain,seed_ytest = train_test_split(self._secondary_training_dataframe.values,
                                                                 np.array(seed_labels).reshape(-1,1),
                                                                 test_size=test_size, random_state=random_state)
        params = self._default_lgb_binary_params
    
        self._seed_model_binary=lgb.LGBMModel(random_state=random_state,
                                              verbose=int(self.verbose),
                                              **params)
        self._foreground_model_binary=lgb.LGBMModel(random_state=random_state,
                                              verbose=int(self.verbose),
                                              **params)
        if self.verbose:
            print('Binary seed pixel classification model:')
            print(self._seed_model_binary)
            print('Training...')
        t1 = datetime.datetime.now()
        self._seed_model_binary.fit(seed_xtrain,seed_ytrain)
        t2 = datetime.datetime.now()
        if self.verbose:
            print('Binary foreground pixel classification model:')
            print(self._seed_model_binary)
            print('Training...')
        self._foreground_model_binary.fit(fg_xtrain,fg_ytrain)
        t3 = datetime.datetime.now()
        if self.verbose:
            print('Training time: {}'.format(t3-t2))
            
    def predict(self,target,return_feature_images=False):
        canvas = np.zeros(list(target.shape)+[3])
        canvas[:,:,0]=1
        x,y,df,norm_input,feature_images = self._image2features(target, return_feature_images=True)
        canvas[x,y,:]=self._predict_func(df.values)

        output_dict = {'Probability_map':canvas}
        if return_feature_images:
            output_dict['Feature_images']=feature_images
        if self.prob2binary:
            fg_canvas = np.zeros(target.shape)
            seed_canvas = np.zeros(target.shape)
            user_xy = (x,y)
            _x,_y,bg_df=self._image2features(canvas[:,:,0],user_xy=user_xy,_force_ignore_selem=True)
            _x,_y,edge_df=self._image2features(canvas[:,:,1],user_xy=user_xy,_force_ignore_selem=True)
            _x,_y,core_df=self._image2features(canvas[:,:,2],user_xy=user_xy,_force_ignore_selem=True)
            _df= pd.concat([bg_df,edge_df,core_df],axis=1)
            fg_canvas[x,y] = self._foreground_model_binary.predict(_df.values)
            seed_canvas[x,y] = self._seed_model_binary.predict(_df.values)
            output_dict['Foreground']=fg_canvas
            output_dict['Seed']=seed_canvas
        return output_dict
    
    def save(self,dst):
        import pickle as pk
        if self.drop_before_dump:
            self.X_train, \
            self.X_test, \
            self.y_train, \
            self.y_test, \
            self.training_images, \
            self.training_masklabels, \
            self.training_dataframe, \
            self.training_masks = [None]*8
        if self.verbose:
            print('All traininging data will be deleted before saving the trained model to free up space.\n'+\
                  'If you want to preserve the training dataset, set "drop_before_dump" to True.')
            print('The trained model will be saved to the following path:\n'+\
                  '"{}"'.format(dst))
        pk.dump(self,open(dst,'wb'))
    
    def _image2features(self,
                        img,
                        return_feature_images=False,
                        user_xy=(),
                        _force_ignore_selem=False):
        # dual-bandpass filter
        img=dual_bandpass(img, 
                          pixel_microns=self.pixel_microns,
                          min_structure_scale=self.bandpass_struct[0],
                          max_structure_scale=self.bandpass_struct[1])
        
        # percentile normalization
        img=generic.percentile_normalize(img,
                                         self.perc_norm_params[0],
                                         self.perc_norm_params[1],
                                         self.perc_norm_params[2],
                                         self.perc_norm_params[3])
        
        # generate pixel-wise feature images
        feature_images=multiscale_image_feature(img, 
                                                sigmas=self.sigmas,
                                                rog=self.use_RoG,
                                                ridge=self.use_ridge,
                                                sobel=self.use_sobel,
                                                shapeindex=self.use_shapeindex,
                                                ext_kernels=self.extra_kernels)[1]
            
            # define pixels of interest
        if self.mask_mode=='full':
            x,y = np.where(img>-1)
        elif len(user_xy)==2:
            x,y = user_xy
        else:
            isodata_mask = (img<filters.threshold_isodata(img))*1
            isodata_mask = morphology.binary_dilation(isodata_mask,
                                                      morphology.disk(self.dilation_radius))
            isodata_mask = morphology.remove_small_holes(isodata_mask,
                                                         area_threshold=self.hole_size)
            x,y = np.where(isodata_mask>0)
            
        
        # local feature stat extraction
        selem=None
        if isinstance(self.local_stat_radius,int):
            if self.local_stat_radius>0:
                selem=morphology.disk(self.local_stat_radius)
        if _force_ignore_selem:
            selem=None
        if return_feature_images:
            return x,y,local_stat(feature_images,x,y,selem=selem),img,feature_images
        else:
            return x,y,local_stat(feature_images,x,y,selem=selem)      

        
        
        
        
        
        
        
class GenericClassifier:

    def __init__(self, method='RandomForest', existing_classifier=None, **kwargs):
        self.model = None
        if method == 'RandomForest':
            self.model = RandomForestClassifier(**kwargs)
        if method == 'PretrainedModel' and existing_classifier is not None:
            self.model = existing_classifier(**kwargs)
        self.features = []
        self.method = method
        self.normalizer = StandardScaler()
        self.z_transform = False

    def train(self, training_df, test_size=0.4, random_state=33, verbose=True, normalize=False, **kwargs):
        _missing_features = []
        _features = []
        for f in self.features:
            if f not in training_df.columns:
                _missing_features.append(f)
            else:
                _features.append(f)
        if len(_missing_features) > 0 and verbose:
            print(
                'The following features are not found in the training dataset: {}'.format(','.join(_missing_features)))
            self.features = _features
        if 'annotation' not in training_df.columns:
            raise ValueError("Dataframe contains no 'annotation' for classification.")
        self.data = training_df[self.features].values
        self.data[np.where(np.isnan(self.data))] = 0
        if normalize:
            self.z_transform = True
            self.normalizer.fit(self.data)
            self.data = self.normalizer.transform(self.data)
        self.annotations = training_df['annotation'].values.reshape(-1, 1)
        self.X_train, \
        self.X_test, \
        self.Y_train, \
        self.Y_test = train_test_split(self.data, self.annotations, test_size=test_size, random_state=random_state)
        self.model.fit(self.X_train, self.Y_train.ravel())
        self.classification_score = self.model.score(self.X_test, self.Y_test.ravel())
        if verbose:
            print('Classification score using {} model is {}'.format(self.method, self.classification_score))

    def predict(self, target_df,probability = False):
        _missing_features = []
        for f in self.features:
            if f not in target_df.columns:
                _missing_features.append(f)
        if len(_missing_features) > 0:
            raise ValueError(
                'The following features are not found in the training dataset: {}'.format(','.join(_missing_features)))
        if len(target_df) == 0:
            raise ValueError('No data found!')
        data = target_df[self.features].values
        data = np.nan_to_num(data, copy=True, nan=0.0, posinf=0, neginf=0)
        if len(data) == 1:
            data = data.reshape(1, -1)
        if self.z_transform:
            data = self.normalizer.transform(data)
        if probability:
            return self.model.predict_proba(data)
        else:
            return self.model.predict(data)

    def update(self):
        return None

    
class BoundaryClassifier:
    
    def __init__(self,
                 advanced=False,
                 feature_sigma=0.8):
        self.advanced=advanced
        self.sigmas=float(feature_sigma)
        self.training_dataframe = []
        self.training_thumbnails = []
        self.training_annot = None
        self.basic_feature_columns = ['boundary_length','max_perimeter_fraction','neighbor_orientation_diff',
                                      'stiched_orientation_diff','neighbor_size_difference','min_neighor_size',
                                      'stiched_length_fraction','stiched_size','stiched_convexity','stiched_eccentricity',
                                      'stiched_solidity','stiched_extent']
        self.intensity_columns =['{}_{}'.format(stat,name) for stat in ['mean','std','min','max'] for name in ['background','edge','seed','shapeindex','ridge']]
        self.classes = {1:'Good boundary',2:'Bad boundar'}
        self.classifier_basic = GenericClassifier(**{'criterion':'entropy',
                                                   'n_jobs':-1,'max_features':10,'random_state':42,
                                                    'class_weight':'balanced','verbose':True})
        self.classifier_basic.features = self.basic_feature_columns
        self.classifier_with_intensity = GenericClassifier(**{'criterion':'entropy',
                                                   'n_jobs':-1,'max_features':15,'random_state':42,
                                                    'class_weight':'balanced','verbose':True})
        self.classifier_with_intensity.features = self.basic_feature_columns+self.intensity_columns
        
    def load_trainingset(self,patch):
        from PIL import Image
        basic_mode,boundaries,concat_dataframe=self._load_from_patch(patch)
        norm_img = generic.percentile_normalize(patch.get_ref_image(),
                                                0,99.5,None,np.percentile(patch.get_ref_image(),99.5)*1.2)
        
        for k,v in boundaries.items():
            x0,x1,y0,y1 = v.bbox
            crop = norm_img[x0:x1,y0:y1]
            boundary_overlay = crop.copy()
            boundary_overlay[v.coords[:,0]-x0,v.coords[:,1]-y0]=1
            h,w = crop.shape
            if w>h:
                crop=crop.T
                boundary_overlay=boundary_overlay.T
            stiched = np.hstack([crop,boundary_overlay])
            stiched = (stiched*255).astype(np.uint8)
            self.training_thumbnails.append(Image.fromarray(stiched))
        if len(self.training_dataframe)==0:
            self.training_dataframe=concat_dataframe
        else:
            self.training_dataframe=pd.concat([self.training_dataframe,
                                               concat_dataframe])
        if self.training_annot is None:
            self.training_annot = np.zeros(len(concat_dataframe))
        else:
            self.training_annot = np.hstack([self.training_annot,np.zeros(len(concat_dataframe))])
        
    def label_trainingset(self):
        Labeler(self.training_thumbnails, label_dict=self.classes, annot=self.training_annot)
        
    def train(self,test_size=0.2,random_state=33):
        df = self.training_dataframe.copy()
        df['annotation']=self.training_annot
        df_filtered = df[df['annotation']!=0]
        self.classifier_basic.train(df_filtered,test_size=0.2,random_state=33)
        try:
            self.classifier_with_intensity.train(df_filtered,test_size=0.2,random_state=33)
        except:
            print('Intensity features not found!')
            
    def predict(self,patch,force_basic_mode=True):
        basic_mode,boundaries,concat_dataframe = self._load_from_patch(patch)
        if force_basic_mode:
            basic_model=True
        if basic_mode:
            return boundaries,concat_dataframe,self.classifier_basic.predict(concat_dataframe)
        else:
            try:
                return boundaries,concat_dataframe,self.classifier_with_intensity.predict(concat_dataframe)
            except:
                print('Intensity features not found! Will use basic features instead.')
                return boundaries,concat_dataframe,self.classifier_basic.predict(concat_dataframe)
    
    def _load_from_patch(self,patch):
        basic_mode=False
        _boundary_ref_dict,boundaries = boundary_neighbor_pairwise(patch.labeled_mask,
                                                                   patch.mask-(patch.labeled_mask>0))
        if patch.prob_mask is not None:
            feature_images = {'background':patch.prob_mask[:,:,0],
                              'edge':patch.prob_mask[:,:,1],
                              'seed':patch.prob_mask[:,:,2]}
        else:
            basic_mode=True
        if self.advanced and not basic_mode:
            shape_index_name = 'Sigma_{}-shapeindex'.format(round(self.sigmas[0],1))
            ridge_name = 'Sigma_{}-ridge'.format(round(self.sigmas[0],1))
            if not shape_index_name in patch.cache.feature_images:
                patch.cache.feature_images[shape_index_name] = feature.shape_index(patch.get_ref_image(),
                                                                                   sigma=self.sigmas[0])
            if not ridge_name in patch.cache.feature_images:
                patch.cache.feature_images[ridge_name] = filters.sato(patch.get_ref_image(),
                                                                      sigmas=(self.sigmas[0]),
                                                                      black_ridges=False)
            feature_images['shapeindex'] = patch.cache.feature_images[shape_index_name]
            feature_images['ridge'] = patch.cache.feature_images[ridge_name]
        
        concat_dataframe=[]
        for k,v in boundaries.items():
            v.stich_neighbors(patch.regionprops)
            v.measure(patch.regionprops,feature_images=feature_images)
            concat_dataframe.append(v.data)
        concat_dataframe=pd.concat(concat_dataframe)
        return basic_mode,boundaries,concat_dataframe
    
    def save(self,dst):
        print('Saving boundary prediction model to:\n{}'.format(dst))
        pk.dump(self,open(dst,'wb'))
    
"""
class particle_classifier_legend:
    
    def __init__(self, 
                 use_RoG = True,  
                 use_contour_stat=True,
                 **kwargs):
        self.features = None
        self.use_RoG = use_RoG
        self.use_contour_stat = use_contour_stat
        self.model = GenericClassifier(method='RandomForest', **kwargs)
        
    def measure(self,patch):
        self.features = [c for c in patch.regionprops.columns if '$' not in c]
        if len(patch.regionprops)>0:
            if self.use_RoG:
                self.features+=particle_RoG_profile(patch)
            if self.use_contour_stat:
                self.features+=particle_contour_profile(patch)
        self.model.features = self.features
    
    def load_training(self):
        return None

    def drop_features(self,exclude = []):
        return None

    def init_GUI(self):
        return None
"""


def save_model(model, pickle_filename):
    pk.dump(model, open(pickle_filename, 'wb'))


def load_model(pickle_filename):
    return pk.load(open(pickle_filename, 'rb'))


