import numpy as np
from skimage import measure, morphology

def roi2multilabel(roi_file,image_shape,dst,erosion_radius=2):
    import read_roi as rr
    import tifffile as tf
    from skimage import draw,morphology
    roi = rr.read_roi_zip(roi_file)
    canvas = np.zeros(image_shape)
    core = np.zeros(image_shape)
    for k,v in roi.items():
        xy = np.array([v['y'],v['x']]).T
        mask = draw.polygon2mask(image_shape,xy)
        eroded = morphology.binary_erosion(mask,morphology.disk(erosion_radius))
        core[eroded==1]=1
    dilated = morphology.binary_dilation(core,morphology.disk(erosion_radius+1))
    canvas[(dilated-core)>0]=125
    canvas[core>0]=254
    tf.imsave(dst,canvas.astype(np.uint8),imagej=True)
    
    
def prediction2seed(pred_multilabel_mask,seed_min=0.3,edge_max=0.75,min_seed_size=20,opening_radius=1):
    seed=(pred_multilabel_mask[:,:,2]>seed_min)*(pred_multilabel_mask[:,:,1]<edge_max)*1
    if isinstance(opening_radius,int) and opening_radius>0:
        seed=morphology.binary_opening(seed,morphology.disk(opening_radius))
    seed=morphology.remove_small_objects(seed,min_seed_size)
    return measure.label(seed)


def prediction2foreground(pred_multilabel_mask,channel=0,
                          threshold=0.4,
                          erosion_radius=1):
    fg=(pred_multilabel_mask[:,:,channel]<threshold)*1
    #fg=morphology.binary_closing(fg,morphology.disk(1))
    if isinstance(erosion_radius,int) and erosion_radius>0:
        fg=morphology.binary_erosion(fg,morphology.disk(erosion_radius))
    return fg*1