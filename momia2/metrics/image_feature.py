import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from skimage import filters, measure
from skimage.feature import hessian_matrix_eigvals
from concurrent.futures import ThreadPoolExecutor
from skimage.morphology import disk
from scipy import stats
from ..utils.contour import bend_angle
import cv2

"""
class ImageProfiler:
    
    def __init__(self, )
"""


def preset_features(config):

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


def _uniscale_image_feature(image,
                            sigma,
                            mode='reflect',
                            order='rc',
                            shapeindex=True,
                            sobel=True,
                            ridge=True,
                            ext_kernels={},
                            cval=0):
    from itertools import combinations_with_replacement
    from skimage.feature import hessian_matrix_eigvals
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
    gaussian_filtered = filters.gaussian(image, 
                                         sigma=sigma,
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
    l1, l2 = hessian_matrix_eigvals(H_elems)
    
    image_features = {'H_eig_1':l1, 'H_eig_2':l2}
    if shapeindex:
        # shape index
        shpid =  (2.0 / np.pi) * np.arctan((l2 + l1) / (l2 - l1))
        shpid[np.isnan(shpid)]=0
        image_features['shapeindex'] = shpid
    if sobel:
        # sobel edge
        image_features['sobel'] = filters.sobel(gaussian_filtered)

    if ridge:
        image_features['ridge'] = filters.sato(image, sigmas=(0.1, sigma), black_ridges=False, mode='constant')            

    if len(ext_kernels)>0:
        for name,kernel in ext_kernels.items():
            image_features[name] = cv2.filter2D(image,
                                                -1,
                                                kernel)
    return gaussian_filtered, image_features


def multiscale_image_feature(image,
                             sigmas=(0.5, 0.8, 1.5, 3.5),
                             rog_sigmas = (0.5, 5), 
                             num_workers=None,
                             mode='reflect',
                             order='rc',
                             shapeindex=True,
                             sobel=True,
                             ridge=True,
                             rog=True,
                             cval=0,
                             ext_kernels={},
                             header=''):
    from concurrent.futures import ThreadPoolExecutor
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
                lambda s: _uniscale_image_feature(image, s,
                                                     mode=mode,
                                                     order=order,
                                                     shapeindex=shapeindex,
                                                     sobel=sobel,
                                                     ridge=ridge,
                                                     cval=cval,
                                                     ext_kernels=ext_kernels),
                sigmas,
            )
        )
    multiscale_gaussian = {}
    multiscale_features = {}
    for i,pair in enumerate(out_sigmas):
        gaussian_filtered, image_features = pair
        multiscale_gaussian[sigmas[i]] = gaussian_filtered
        multiscale_features[sigmas[i]] = image_features
    reformated_features = {'Sigma_{}-{}'.format(sigma,k):v for sigma, dic in multiscale_features.items() for k,v in dic.items()}   
    
    # compute rog
    if rog:
        if rog_sigmas[0] in sigmas:
            g1 = multiscale_gaussian[rog_sigmas[0]]
        else:
            g1 = filters.gaussian(image, rog_sigmas[0])
        if rog_sigmas[1] in sigmas:
            g2 = multiscale_gaussian[rog_sigmas[1]]
        else:
            g2 = filters.gaussian(image, rog_sigmas[1])
        reformated_features['RoG']=g2/g1
   
    return multiscale_gaussian, reformated_features


def local_stat(target_images, 
               x_coords, 
               y_coords, 
               selem=disk(2),
               as_dataframe=True):
    """

    :param target_images:
    :param x_coords:
    :param y_coords:
    :param selem:
    :return:
    """
    stacked_measures = []
    col_names = []
    
    if selem is None:
        for name, img in target_images.items():
            col_names.append('{}-intensity'.format(name))
            stacked_measures.append(img[x_coords, y_coords].reshape(-1,1))
    else:
        x_mesh, y_mesh = coord2mesh(x_coords, y_coords, selem,
                                    target_images[list(target_images.keys())[0]].shape[0],
                                    target_images[list(target_images.keys())[0]].shape[1])

        for name, img in target_images.items():
            mesh_data = img[x_mesh, y_mesh]
            mesh_measures = np.array([img[x_coords, y_coords],
                                      np.min(mesh_data, axis=1),
                                      np.max(mesh_data, axis=1),
                                      np.std(mesh_data, axis=1),
                                      np.mean(mesh_data, axis=1)]).T
            col_names += ['{}-{}'.format(name, stat) for stat in ['intensity','min','max','std','mean']]
            stacked_measures.append(mesh_measures)
    if as_dataframe:
        return pd.DataFrame(np.hstack(stacked_measures),columns=col_names)
    else:
        return np.hstack(stacked_measures)


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


def ratio_of_gaussian(img, s1=0.5, s2=5):
    g1 = filters.gaussian(data,sigma=s1)
    g2 = filters.gaussian(data,sigma=s2)
    return g1/g2


def hu_log10_transform(hu_moments):
    warnings.filterwarnings("ignore")
    abs_vals = np.abs(hu_moments)
    hu_log10 = np.copysign(np.log10(abs_vals), hu_moments)
    corrected = np.zeros(hu_moments.shape)
    corrected[np.isfinite(hu_log10)] = hu_log10[np.isfinite(hu_log10)]
    return corrected


def median_over_gaussian(data,windowsize=3):
    return (ndi.median_filter(data,size=windowsize)/filters.gaussian(data,sigma=windowsize,preserve_range=True))


def normalized_difference_of_gaussian(data, s1=0, s2=10):
    g1 = filters.gaussian(data, sigma=s1)
    g2 = filters.gaussian(data, sigma=s2)
    return (g1-g2)/g1


def get_regionprop(labeled_image, intensity_image=None):
    regionprop_properties = ['label','bbox','coords',
                             'area','convex_area','filled_area','eccentricity','solidity',
                             'major_axis_length','minor_axis_length',
                             'perimeter','equivalent_diameter',
                             'extent','orientation',
                             'inertia_tensor','moments_hu','weighted_moments_hu',
                             'mean_intensity','max_intensity','min_intensity']

    if intensity_image is not None:
        rp_table = pd.DataFrame(measure.regionprops_table(labeled_image,
                                                          intensity_image,
                                                          properties=regionprop_properties))
    else:
        rp_table = pd.DataFrame(measure.regionprops_table(labeled_image,
                                                          intensity_image=(labeled_image>0)*1,
                                                          properties=regionprop_properties))
    #rp_table['roundness'] = np.round(4*rp_table['area'].values/np.pi*(rp_table['major_axis_length'].values**2),3)
    #rp_table['circularity'] = np.round(np.sqrt(4*np.pi*rp_table['area'].values/(rp_table['perimeter'].values**2)),3)
    rp_table['aspect_ratio'] = np.round(rp_table['minor_axis_length'].values/rp_table['major_axis_length'].values,3)
    rp_table.rename(columns={'bbox-0':'$bbox-0',
                             'bbox-1':'$bbox-1',
                             'bbox-2':'$bbox-2',
                             'bbox-3':'$bbox-3',
                             'coords':'$coords',
                             'label':'$label'},
                    inplace=True)
    return rp_table


def particle_RoG_profile(patch, s1=0.5,s2=5):
    basic_measures = ['weighted_moments_hu','mean_intensity',
                          'min_intensity','max_intensity']
    try:
        RoG = patch.cache.feature_images['RoG']
    except:
        RoG = ratio_of_gaussian(self.get_ref_image(),s1,s2)
        
    rog_measures = pd.DataFrame(measure.regionprops_table(patch.labeled_mask,
                                                          intensity_image=RoG,
                                                          properties=basic_measures,
                                                          extra_properties=(masked_std, masked_skewness, masked_kurtosis))).values
    columns = ['RoG_weighted_hu-{}'.format(x) for x in range(0,7)] +\
              ['RoG_mean','RoG_min','RoG_max','RoG_std','RoG_skewness','RoG_kurtosis']
    patch.regionprops[columns] = rog_measures
    return columns


def particle_contour_profile(patch):
    contour_data=[]
    columns = ['bending_std','bending_90','bending_10',
               'bending_max','bending_min','bending_skewness',
               'bending_median']
    for contours in patch.regionprops['$contours'].values:
        contour_angles = bend_angle(contours[0], window=3)
        contour_data.append([np.std(contour_angles),
                             np.percentile(contour_angles,90),
                             np.percentile(contour_angles,10),
                             np.max(contour_angles),
                             np.min(contour_angles),
                             stats.skew(contour_angles),
                             np.median(contour_angles)])
    patch.regionprops[columns]=contour_data
    return columns


def masked_percentile(regionmask, intensity):
    return np.percentile(intensity[regionmask], q=(25, 50, 75))


def masked_cv(regionmask, intensity):
    return 100*np.std(intensity[regionmask])/np.mean(intensity[regionmask])


def masked_skewness(regionmask, intensity):
    return stats.skew(intensity[regionmask])


def masked_kurtosis(regionmask, intensity):
    return stats.kurtosis(intensity[regionmask])


def masked_std(regionmask, intensity):
    return np.std(intensity[regionmask])


def gabor_kernel_bank(n_theta=4,
                      sigmas=[2],
                      lamdas = [np.pi/4],
                      gammas = [0.5],
                      kernel_size=5):
    gabor_kernels = {}
    counter=0
    for i in range(n_theta):
        theta = (i/n_theta)*np.pi
        for sigma in sigmas:
            for lamda in lamdas:
                for gamma in gammas:
                    kernel = cv2.getGaborKernel((kernel_size,kernel_size),sigma,theta,lamda,gamma,0,ktype=cv2.CV_64F)
                    gabor_kernels['Gabor-{}'.format(counter)]=kernel
                    counter+=1
    return gabor_kernels