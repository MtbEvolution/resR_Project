from .helper_image import *
from .segmentation import *
from .optimize import *
from .config import *
from .classification import *
from .patch import *
import os, glob

class Project:

    def __init__(self):
        self.config = None
        self.patch_classifier = None
        self.particle_classifier = None
        self.pixel_classifier = None
        self.date_created = None
        self.random_state = None
        self.log = None
        self.training_patches = {}
        self.training_boundaries = {}

    def load_config(self, configfile):
        self.config = Preset_configurations_default(configfile).config

    def load_trainingset(self,
                         src_folder):
        """

        :param src_folder:
        :param seed_identifier:
        :return:
        """

        seed_identifier = str(self.config['classification_feature']['seed_identifier'])

        if not os.path.isdir(src_folder):
            raise ValueError('{} is not a valid folder!'.format(src_folder))
        seeds = sorted(glob.glob(src_folder + '*{}'.format(seed_identifier)))
        if len(seeds) == 0:
            raise ValueError('No {} file found in {}'.format(seed_identifier, src_folder))
        for s in seeds:
            img_id = s.split('/')[-1].split('_seed')[0]
            src_image = plt.imread(src_folder + '{}.tif'.format(img_id))
            seed_image = plt.imread(s)
            self.update_training_data(src_image, seed_image, img_id)

    def update_training_data(self, src_image, seed_image, img_id):
        self.training_patches[img_id] = make_patch(src_image, img_id, self.config, seed_image=seed_image)

    def train_patch_classifier(self, **kwargs):
        self.patch_classifier = GenericClassifier(**kwargs)
        self.patch_classifier.features = preset_patch_features(self.config)
        self.patch_classifier.train(self._merge_patch_data(), verbose=True)

    def train_pixel_classifier(self,
                               method='RandomForest',
                               existing_classifier=None,
                               normalize=True, **kwargs):
        """

        :param kwargs:
        :return:
        """
        self.pixel_classifier = GenericClassifier(method=method,
                                                  existing_classifier=existing_classifier,
                                                  **kwargs)
        pixel_data = self._merge_pixel_data()
        self.pixel_classifier.features = [x for x in pixel_data.columns if x != 'annotation']
        self.pixel_classifier.train(pixel_data,
                                    normalize=normalize, **kwargs)

    def train_pixel_proba_classifier(self, selem=morphology.disk(2), **kwargs):
        """

        :param selem:
        :param kwargs:
        :return:
        """
        self.pixel_proba_classifier = GenericClassifier(**kwargs)
        pixel_proba_data = self._merge_pixel_proba_data(selem=selem)
        self.pixel_proba_classifier.features = [x for x in pixel_proba_data.columns if x != 'annotation']
        self.pixel_proba_classifier.train(pixel_proba_data, verbose=True)
        return None

    def _merge_patch_data(self):
        """

        :return:
        """
        dataframes = [p.regionprop_table for k, p in self.training_patches.items()]
        merged_df = pd.concat(dataframes)
        return merged_df

    def _merge_pixel_data(self):
        """

        :return:
        """
        dataframes = [p._pixel_features for k, p in self.training_patches.items()]
        merged_df = pd.concat(dataframes)
        return merged_df

    def _merge_pixel_proba_data(self):
        """

        :return:
        """
        dataframes = []
        for k,p in self.training_patches.items():
            dataframes.append(render_pixel_feature_proba(p, self.pixel_classifier))
        merged_df = pd.concat(dataframes)
        return merged_df

    def _reset_patch_features(self, features):
        self.patch_classifier.features = features


def make_patch(src_image, img_id, config,
               seed_image=None):
    if isinstance(src_image, dict):
        data = src_image
    else:
        data = {'ref': src_image}
    scale = float(config['image']['rescale'])
    data = {k:rescale_img(v, scale) for k,v in data.items()}
    p = Patch()
    p.load_config(config)
    p.load_data(img_id, data)
    p.segmentation_basic()

    s1,s2 = np.array(config['image']['RoG_sigmas'].split(',')).astype(float)
    p._measure_RoG(s1=s1,s2=s2)
    p.predictions = np.zeros(len(p.regionprop_table))

    if bool(int(config['classification_feature']['precompute_contour'])):
        p.find_contours()

    if seed_image is not None:
        p._adopt_seeds(rescale_img(seed_image, scale)>0)
        p._seed2annotation(overwrite=True)
        p._binary_from_seed = _annotation2labelmap(p)
        p.ridge = filters.sato(p.get_ref_image(),
                               sigmas=np.array(config['segmentation']['sato_sigmas'].split(',')).astype(float),
                               black_ridges=False, mode='constant')
        p._seed_watershed = segmentation.watershed(p.ridge,
                                                   p._annotated_seeds,
                                                   mask=p._binary_from_seed,
                                                   compactness=0,
                                                   watershed_line=True)
        erosion_selem = config2selem(config['classification_feature']['mask_erosion_selem'])
        open_selem = config2selem(config['classification_feature']['mask_opening_selem'])
        use_fullimage = bool(int(config['classification_feature']['fullimage']))

        p._masked_coords,\
        p._pixel_annotation = seed2pixel_annotation(p._binary_from_seed,
                                                    p._seed_watershed,
                                                    erosion_selem=erosion_selem,
                                                    open_selem=open_selem,
                                                    full=use_fullimage)
    else:
        p._masked_coords = np.where(p.binary > 0)
    p._pixel_features = render_pixel_feature(p, config, as_dataframe=True)
    p._pixel_features['annotation'] = p._pixel_annotation
    p.regionprop_table['image_id'] = [p.id] * len(p.regionprop_table)
    return p


def seed2pixel_annotation(mask, watershed_labels,
                          erosion_selem=morphology.disk(1),
                          open_selem=morphology.square(3),
                          full=False):
    """

    :param mask:
    :param watershed_labels:
    :param erosion_selem:
    :param open_selem:
    :return:
    """

    if erosion_selem is not None:
        fg = morphology.binary_erosion(watershed_labels > 0, erosion_selem)
    else:
        fg = morphology.binary_erosion(watershed_labels > 0)
    if open_selem is not None:
        fg = morphology.binary_opening(fg, open_selem)
    else:
        fg = morphology.binary_opening(fg)
    if not full:
        bg = ((fg == 0) & (mask > 0))
    else:
        bg = (fg==0)*1

    x1, y1 = np.where(fg == 1)
    x2, y2 = np.where(bg == 1)
    labels = np.array([1] * len(x1) + [2] * len(x2))
    x_concat = np.concatenate([x1, x2])
    y_concat = np.concatenate([y1, y2])

    return np.array([x_concat, y_concat]), labels


def prediction2labelmap(labels,
                        predictions,
                        regionprop_table,
                        bitcode={0: 1, 1: 2, 2: 3}):
    """

    :param labels:
    :param predictions:
    :param regionprop_table:
    :param bitcode:
    :return:
    """
    canvas = np.zeros(labels.shape)
    for i in np.unique(predictions):
        x, y = np.vstack(regionprop_table[predictions == i]['coords'].values).T
        canvas[x, y] = bitcode[i]
    return canvas


def _annotation2labelmap(patch,
                         bitcode={0: 1, 1: 1, 2: 0}):
    """

    :param patch:
    :param bitcode:
    :return:
    """
    return prediction2labelmap(patch.cluster_labels,
                               patch.regionprop_table['annotation'],
                               patch.regionprop_table,
                               bitcode=bitcode)


def render_pixel_feature(patch,
                         config,
                         as_dataframe=True):
    sigmas = np.array(config['segmentation']['sato_sigmas'].split(',')).astype(float)
    num_workers = int(config['classification_feature']['num_workers'])
    use_RoG = bool(int(config['classification_feature']['use_RoG']))
    use_Ridge = bool(int(config['classification_feature']['use_Ridge']))
    use_Sobel = bool(int(config['classification_feature']['use_Sobel']))
    use_Shapeindex = bool(int(config['classification_feature']['use_Shapeindex']))
    selem = config2selem(config['classification_feature']['pixel_feature_selem'])

    img = patch.get_ref_image()
    _gaussian_filtered, image_features = multiscale_image_feature(img,
                                                                  sigmas=sigmas,
                                                                  num_workers=num_workers,
                                                                  ridge=use_Ridge,
                                                                  shapeindex=use_Shapeindex,
                                                                  sobel=use_Sobel)
    concat = []
    for s, dlist in image_features.items():
        concat += dlist
    if use_RoG and patch.RoG is not None:
        concat += [patch.RoG]
    x, y = patch._masked_coords
    pixel_feature_type = bool(int(config['classification_feature']['pixel_feature_rawdata']))
    pixel_measures = pixel_feature(concat, x, y, selem=selem, raw=pixel_feature_type)
    if as_dataframe:
        pixel_measures = pd.DataFrame(pixel_measures, columns=np.arange(pixel_measures.shape[1]))
    return pixel_measures


def pixel_feature(target_images, x_coords, y_coords,
                  selem=morphology.disk(1), raw = False):
    if raw:
        return _pixel_feature_flat(target_images,x_coords, y_coords, selem)
    else:
        return _pixel_feature_stat(target_images, x_coords, y_coords, selem)

def _pixel_feature_stat(target_images, x_coords, y_coords, selem=morphology.disk(2)):
    """

    :param target_images:
    :param x_coords:
    :param y_coords:
    :param selem:
    :return:
    """
    x_mesh, y_mesh = coord2mesh(x_coords, y_coords, selem,
                                target_images[0].shape[0],
                                target_images[0].shape[1])
    stacked_measures = []
    for img in target_images:
        mesh_data = img[x_mesh, y_mesh]
        mesh_measures = np.array([img[x_coords, y_coords],
                                  np.min(mesh_data, axis=1),
                                  np.max(mesh_data, axis=1),
                                  np.std(mesh_data, axis=1),
                                  np.mean(mesh_data, axis=1)]).T
        stacked_measures.append(mesh_measures)
    return np.hstack(stacked_measures)


def _pixel_feature_simp(target_images, x_coords, y_coords):
    stacked_measures = []
    for img in target_images:
        stacked_measures.append(img[x_coords, y_coords])
    return np.vstack(stacked_measures).T


def render_pixel_feature_proba(patch,
                               pixel_classifier,
                               selem=morphology.disk(2),
                               as_dataframe=True):
    """

    :param patch:
    :param pixel_classifier:
    :param selem:
    :param as_dataframe:
    :return:
    """
    x,y = patch._masked_coords
    predictions = pixel_classifier.predict(patch._pixel_features,probability=True)
    canvas_fg,canvas_bg = np.zeros(patch.binary.shape),\
                          np.zeros(patch.binary.shape)
    canvas_fg[x,y] = predictions[:,0]
    canvas_bg[x,y] = predictions[:,1]
    pixel_proba = _pixel_feature_flat([canvas_fg,canvas_bg],x,y,selem=selem)
    if as_dataframe:
        pixel_proba = pd.DataFrame(pixel_proba, columns=np.arange(pixel_proba.shape[1]))
    pixel_proba['annotation']=patch._pixel_annotation
    return pixel_proba


def coord2mesh(x, y, selem,
               xmax=10000000,
               ymax=10000000):
    """

    :param x:
    :param y:
    :param selem:
    :param xmax:
    :param ymax:
    :return:
    """
    dx, dy = np.where(selem)
    dx -= int(selem.shape[0] / 2)
    dy -= int(selem.shape[1] / 2)
    x_mesh = x[:, np.newaxis] + dx[np.newaxis, :]
    x_mesh[x_mesh < 0] = 0
    x_mesh[x_mesh > xmax - 1] = xmax - 1
    y_mesh = y[:, np.newaxis] + dy[np.newaxis, :]
    y_mesh[y_mesh < 0] = 0
    y_mesh[y_mesh > ymax - 1] = ymax - 1
    return x_mesh, y_mesh


def _singlescale_image_feature(image,
                               sigma,
                               mode='reflect',
                               order='rc',
                               shapeindex=True,
                               sobel=True,
                               ridge=True,
                               cval=0,
                               user_func = [],
                               user_func_kwargs = []):
    """

    :param image:
    :param sigma:
    :param mode:
    :param order:
    :param shapeindex:
    :param sobel:
    :param ridge:
    :param cval:
    :return:
    """
    gaussian_filtered = filters.gaussian(image, sigma=sigma,
                                         mode=mode,
                                         cval=cval)
    gradients = np.gradient(gaussian_filtered)
    axes = range(image.ndim)
    if order == 'rc':
        axes = reversed(axes)

    H_elems = [np.gradient(gradients[ax0], axis=ax1)
               for ax0, ax1 in combinations_with_replacement(axes, 2)]

    # correct for scale
    H_elems = [(sigma ** 2) * e for e in H_elems]
    l1, l2 = feature.hessian_matrix_eigvals(H_elems)

    image_features = [l1, l2]
    if shapeindex:
        # shape index
        image_features.append((2.0 / np.pi) * np.arctan((l2 + l1) / (l2 - l1)))
    if sobel:
        # sobel edge
        image_features.append(filters.sobel(gaussian_filtered))

    if ridge:
        image_features.append(filters.sato(image, sigmas=(0.1, sigma), black_ridges=False, mode='constant'))

    if len(user_func)>0:
        for i,func in enumerate(user_func):
            kwarg = user_func_kwargs[i]
            image_features.append(func(gaussian_filtered, **kwarg))
    return gaussian_filtered, image_features


def multiscale_image_feature(image,
                             sigmas=(0.5, 0.8, 1.5, 3.5),
                             num_workers=None,
                             mode='reflect',
                             order='rc',
                             shapeindex=True,
                             sobel=True,
                             ridge=True,
                             cval=0):
    """

    :param image:
    :param sigmas:
    :param num_workers:
    :param mode:
    :param order:
    :param shapeindex:
    :param sobel:
    :param ridge:
    :param cval:
    :return:
    """
    if num_workers is None:
        num_workers = len(sigmas)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        out_sigmas = list(
            ex.map(
                lambda s: _singlescale_image_feature(image, s,
                                                     mode=mode,
                                                     order=order,
                                                     shapeindex=shapeindex,
                                                     sobel=sobel,
                                                     ridge=ridge,
                                                     cval=cval),
                sigmas,
            )
        )
    multiscale_gaussian = {}
    multiscale_features = {}
    for i,pair in enumerate(out_sigmas):
        gaussian_filtered, image_features = pair
        multiscale_gaussian[sigmas[i]] = gaussian_filtered
        multiscale_features[sigmas[i]] = image_features

    return multiscale_gaussian, multiscale_features


def _pixel_feature_flat(target_images, x_coords, y_coords,
                        selem=morphology.disk(1)):
    """

    :param target_images:
    :param x_coords:
    :param y_coords:
    :param selem:
    :return:
    """
    x_mesh, y_mesh = coord2mesh(x_coords, y_coords, selem,
                                target_images[0].shape[0],
                                target_images[0].shape[1])
    stacked_measures = []
    for img in target_images:
        mesh_data = img[x_mesh, y_mesh]
        stacked_measures.append(mesh_data)
    return np.hstack(stacked_measures)


def preset_patch_features(config):

    selected_features = []

    morphological_features = ['area', 'eccentricity', 'solidity', 'touching_edge',
                              'convex_area', 'filled_area', 'eccentricity', 'solidity',
                              'major_axis_length', 'minor_axis_length',
                              'perimeter', 'equivalent_diameter',
                              'extent', 'circularity', 'aspect_ratio',
                              'moments_hu-0', 'moments_hu-1',
                              'moments_hu-2', 'moments_hu-3',
                              'moments_hu-4', 'moments_hu-5', 'moments_hu-6']

    orientation_features = ['orientation', 'inertia_tensor-0-0', 'inertia_tensor-0-1',
                            'inertia_tensor-1-0', 'inertia_tensor-1-1']

    weighted_hu = ['weighted_moments_hu-0', 'weighted_moments_hu-1', 'weighted_moments_hu-2',
                   'weighted_moments_hu-3', 'weighted_moments_hu-4', 'weighted_moments_hu-5',
                   'weighted_moments_hu-6', 'RoG_weighted_hu-0', 'RoG_weighted_hu-1', 'RoG_weighted_hu-2',
                   'RoG_weighted_hu-3', 'RoG_weighted_hu-4', 'RoG_weighted_hu-5', 'RoG_weighted_hu-6']

    contour_features = ['n_contours', 'bending_std','bending_90',
                        'bending_10', 'bending_max', 'bending_min',
                        'bending_skewness', 'bending_median']

    intensity_features = ['mean_intensity', 'max_intensity', 'min_intensity',
                          'RoG_mean', 'RoG_min', 'RoG_max',
                          'RoG_skewness', 'RoG_kurtosis', 'RoG_std']

    if bool(int(config['classification_feature']['patch_morphology'])):
        selected_features += morphological_features
    if bool(int(config['classification_feature']['patch_orientation'])):
        selected_features += orientation_features
    if bool(int(config['classification_feature']['patch_weighted_hu'])):
        selected_features += weighted_hu
    if bool(int(config['classification_feature']['patch_contour'])) &\
       bool(int(config['classification_feature']['precompute_contour'])):
        selected_features += contour_features
    if bool(int(config['classification_feature']['patch_intensity'])):
        selected_features += intensity_features

    return selected_features


def rescale_img(image,scale):
    from skimage.transform import rescale
    if scale == 1:
        return image
    elif scale != 0:
        return rescale(image,scale)
    else:
        raise ValueError('Invalid scale parameter')