from matplotlib import cm
from skimage import measure, feature, filters, transform, morphology,segmentation
import numpy as np
import pandas as pd
import read_roi as rr
import matplotlib.pyplot as plt
import glob,os
import pickle as pk
import tifffile

class Colony:
    
    def __init__(self,images,masks,edges,dates):
        self.images = images
        self.masks = masks
        self.edges = edges
        self.scale_rule = shift_scaler()
        self.labels = []
        self.colored_labels = []
        self.colony_growths = {}
        self.is_touching_edge = {}
        self.dates = dates
        self.touched = {}
        self.high_contrast_images = []
        
    def link_colonies(self):
        self.labels=[measure.label(x) for x in self.masks]
        for i in range(len(self.masks)-1):
            label1 = self.labels[i]
            label2 = self.labels[i+1]
            edge = self.edges[i+1]
            if label2.max()==0:
                self.labels[i] = label2
            elif label1.max()>0:
                if ((label1*label2)>0).sum()/(label1>0).sum()<0.7:
                    new_label1, new_label2=forward_match(label1,label2,edge,self.scale_rule)
                else:
                    new_label1, new_label2=pixel_match(label1,label2,edge)
                self.labels[i]=new_label1
                self.labels[i+1]=new_label2
        self.labels = np.array(self.labels)
        
    def measure_growth(self):
        for l in np.unique(self.labels)[1:]:
            self.colony_growths[l] = np.array([(x==l).sum() for x in self.labels])
            self.is_touching_edge[l] = np.array([touching_edge(x,l) for x in self.labels])
            
    def fill_holes(self, area_threshold=100):
        for i in range(1,len(self.masks)):
            m1,m2 = self.masks[i-1],self.masks[i]
            lost_a = ((m1>0)&(m2==0))*1
            lost_labeled = measure.label(lost_a)
            rp = pd.DataFrame(measure.regionprops_table(lost_labeled,properties=['area','label']))
            for l,a in rp[['label','area']].values:
                if a>area_threshold:
                    self.masks[i][lost_labeled==l]=1
                    
    def render_highcontrast_images(self, p1=65,p2=180):
        for i in range(len(self.images)):
            image = self.images[i].copy().astype(float)
            mask = self.masks[i]
            if mask.sum()>100:
                p2 = np.percentile(image[mask>0].flatten(),97.5)
            image = (image-p1)/(p2-p1)
            image[image<0]=0
            image[image>1]=1
            self.high_contrast_images.append((image*255).astype(np.uint8))
            
    def label_colonies(self,alpha=0.5):
        colors = {i:cm.get_cmap('Set2')(i%8) for i in range(1,500)}
        for i,l in enumerate(self.labels):
            image = self.high_contrast_images[i].copy()
            colored = np.zeros(list(l.shape)+[4])
            bg_x,bg_y = np.where(l==0)
            colored[bg_x,bg_y,:3] = image[bg_x,bg_y,np.newaxis]/255
            colored[bg_x,bg_y,3] = 1

            for j in np.unique(self.labels)[1:]:
                growth = self.colony_growths[j]
                by_edge = self.is_touching_edge[j]
                x,y = np.where(l==j)
                colored[x,y,:] = np.array(colors[j])*alpha
                colored[x,y,:3] += (1-alpha)*image[x,y,np.newaxis]/255
                colored[x,y,3] = 1                 
            self.colored_labels.append(colored)
            
    def filter_colonies(self):
        drop_list = []
        for l, v in self.is_touching_edge.items():
            if 1 in v:
                if np.where(v==1)[0][0] == np.where(self.colony_growths[l]>0)[0][0]:
                    drop_list.append(l)
            if not filter_by_growth(self.colony_growths[l]):
                drop_list.append(l)
        for l in np.unique(drop_list):
            self.labels[self.labels==l]=0
            g = self.colony_growths.pop(l)
            t = self.is_touching_edge.pop(l)
            
    def inspect_colonies(self):
        n,h,w = self.labels.shape
        for l in np.unique(self.labels)[1:]:
            has_prox_neighbors = []
            for s in self.labels:
                c_label = (s==l)*1
                if c_label.max()==0:
                    has_prox_neighbors.append(0)
                else:
                    dilated = morphology.binary_dilation(c_label,morphology.disk(2))
                    x,y = np.where((dilated==1)&(c_label==0))
                    neighbor_vals = s[x,y]
                    frac_bg = (neighbor_vals==0).sum()/len(neighbor_vals)
                    has_prox_neighbors.append(1-frac_bg)
            self.touched[l] = np.array(has_prox_neighbors)
    
def touching_edge(labeled_image,label_id,edge_width=3):
    if label_id in labeled_image:
        max_x,max_y = np.array(labeled_image.shape)-1
        x,y = np.where(labeled_image==label_id)
        return 1-int((x.min()>edge_width)*(x.max()<max_x-edge_width)*(y.min()>edge_width)*(y.max()<max_y-edge_width))
    else:
        return 0
            
def shift_scaler():
    rule = []
    for dx in np.linspace(-10,10,20):
        for dy in np.linspace(-10,10,20):
            for sx in np.linspace(0.98,1.02,5):
                for sy in np.linspace(0.98,1.02,5):
                    rule.append([dx,dy,sx,sy])
    return np.array(rule)

def dist_matrix(m1,m2,axis=1):
    return np.sqrt(np.sum(np.square(m1[:,np.newaxis,:]-m2[np.newaxis,:,:]),axis=2))

def forward_match(label1,label2,watershed_ref,scale_rule,max_init_dist=50, min_pixel_dist=1):

    r1=pd.DataFrame(measure.regionprops_table(label1,properties=['centroid','area','bbox','label','solidity','eccentricity','coords']))
    r2=pd.DataFrame(measure.regionprops_table(label2,properties=['centroid','area','bbox','label','solidity','eccentricity','coords']))
    
    newlabel_1 = np.zeros(label1.shape)
    newlabel_2 = np.zeros(label2.shape)
    cent1 = r1[['centroid-0','centroid-1']].values
    cent2 = r2[['centroid-0','centroid-1']].values
    
    filtered_cent1id = np.where(np.min(dist_matrix(cent1,cent2),axis=1)<max_init_dist)[0]
    filtered_cent1 = cent1[filtered_cent1id]
    
    shifted = (filtered_cent1[np.newaxis,:,:]*scale_rule[:,2:][:,np.newaxis])+scale_rule[:,:2][:,np.newaxis]
    min_total_distance = np.array([np.min(dist_matrix(x,cent2),axis=1).sum() for x in shifted])
    
    min_rule = scale_rule[np.argmin(min_total_distance)]
    new_cent1 = shifted[np.argmin(min_total_distance)]
    
    coord2 = np.vstack(np.where(label2>0)).T
    
    nearest_pixel = np.min(dist_matrix(new_cent1,coord2),axis=1)
    nearest_pixel_arg = np.argmin(dist_matrix(new_cent1,np.vstack(np.where(label2>0)).T),axis=1)
    nearest_pixel_label = label2[coord2[nearest_pixel_arg,0],
                                 coord2[nearest_pixel_arg,1]]
    filtered_cent1id = filtered_cent1id[np.where(nearest_pixel<min_pixel_dist)]
    new_cent1 = new_cent1[np.where(nearest_pixel<min_pixel_dist)]
    nearest_pixel_label = nearest_pixel_label[np.where(nearest_pixel<min_pixel_dist)]
    filtered_r1 = r1.iloc[filtered_cent1id].reset_index(drop=True)
    for (l,coord) in filtered_r1[['label','coords']].values: 
        newlabel_1[coord[:,0],coord[:,1]] = l
    
    # link colonies
    frame2_dict = {i:[] for i in r2['label'].values}
    for i,l2 in enumerate(nearest_pixel_label):
        l1 = filtered_r1.iloc[i]['label']
        frame2_dict[l2] += [[new_cent1[i],l1]]
        
    base=1
    
    max_x,max_y = np.array(label2.shape)-1
    
    for l2,l1_list in frame2_dict.items():
        if len(l1_list)>1:
            mask = (label2==l2)*1
            seeds = np.zeros(mask.shape)
            for c1,l1 in l1_list:
                x=min(max(0,int(np.round(c1[0],0))),max_x)
                y=min(max(0,int(np.round(c1[1],0))),max_y)
                seeds[x,y]=l1
                
            seeds = morphology.dilation(seeds,morphology.disk(3))*mask
            watersheded = segmentation.watershed(image=watershed_ref,
                                                 markers=seeds,
                                                 mask=mask,
                                                 watershed_line=True,
                                                 compactness=0,connectivity=1)


            for c1,l1 in l1_list:
                newlabel_2[watersheded==l1] = l1
        elif len(l1_list)==1:
            newlabel_2[label2==l2]=l1_list[0][1]
        else:
            x,y = np.where(label2==l2)
            newlabel_2[x,y]=newlabel_1.max()+base
            base+=1
    return newlabel_1.astype(int), newlabel_2.astype(int)

def pixel_match(label1,label2,watershed_ref,min_overlap=0.2):
    r1=pd.DataFrame(measure.regionprops_table(label1,properties=['centroid','area','bbox','label','solidity','eccentricity','coords'])).set_index('label')
    r2=pd.DataFrame(measure.regionprops_table(label2,properties=['centroid','area','bbox','label','solidity','eccentricity','coords'])).set_index('label')
    newlabel_1 = label1.copy()
    newlabel_2 = np.zeros(label2.shape)


    frame2_dict = {i:[] for i in r2.index}

    # remove false r1:
    retained = []
    base = 1
    for i in r1.index:
        coord = r1.loc[i,'coords']
        found_in_r2 = label2[coord[:,0],coord[:,1]]
        if np.count_nonzero(found_in_r2)>min_overlap*len(coord):
            overlap = found_in_r2[np.nonzero(found_in_r2)]
            unique_counts = np.array([[x,(overlap==x).sum()] for x in np.unique(overlap)])
            if len(unique_counts)>1:
                possible_daughters = unique_counts[np.argsort(unique_counts[:,1])[-2:]]
                if (possible_daughters[0,1]/possible_daughters[1,1])<0.8:
                    daughter = possible_daughters[1,0]
                else:
                    cent1 = r1.loc[i,['centroid-0','centroid-1']].values.astype(float)
                    cent2 = r2.loc[possible_daughters[:,0],['centroid-0','centroid-1']].values.astype(float)
                    dists = dist_matrix(np.array([cent1]),cent2)[0]
                    daughter = possible_daughters[np.argmin(dists),0]
            else:
                daughter = unique_counts[0,0]
            frame2_dict[daughter] += [i]
            retained.append(i)
        else:
            newlabel_1[coord[:,0],coord[:,1]] = 0

    for l,parents in frame2_dict.items():
        x,y = r2.loc[l,'coords'].T
        if len(parents)>1:

            seeds = np.zeros(label1.shape)
            for p in parents:
                pcoord = r1.loc[p,'coords']
                seeds[pcoord[:,0],pcoord[:,1]]=p

            mask = (label2==l)*1
            seeds = match_seeds2mask(seeds,mask)
            watersheded = segmentation.watershed(image=watershed_ref,
                                                 markers=seeds,
                                                 mask=mask,
                                                 watershed_line=True,
                                                 compactness=0,connectivity=1)
            for p in parents:
                newlabel_2[watersheded==p] = p
        elif len(parents)==1:
            newlabel_2[x,y]=parents[0]
        else:
            newlabel_2[x,y]=newlabel_1.max()+base
            base+=1
    return newlabel_1.astype(int), newlabel_2.astype(int)
                
def nonzeros(array):
    array = np.sort(array)
    if 0 in array:
        return array[1:]
    else:
        return array
    
def match_seeds2mask(seeds,mask):
    steps = np.arange(-10,10,2)
    new_seeds = np.zeros(seeds.shape)
    x,y = np.meshgrid(steps,steps)
    x=x.flatten()
    y=y.flatten()
    seed_x,seed_y = np.where(seeds>0)
    overlap_ratio = []
    h,w = seeds.shape
    n_seed_pix = len(seed_x)
    for (dx,dy) in zip(x,y):
        newx,newy = seed_x+dx,seed_y+dy
        if (newx.max()<h) and (newy.max()<w) and min(newx.min(),newy.min())>0:
            overlap_ratio.append(mask[newx,newy].sum()/n_seed_pix)
        else:
            overlap_ratio.append(0)
    best_match = np.argmax(overlap_ratio)
    new_seeds[seed_x+x[best_match],seed_y+y[best_match]]=seeds[seed_x,seed_y]
        
    #shifted_seeds = seed_coords[:,np.newaxis,:]+xy[np.newaxis,:,:]
    return new_seeds

def sort_gridpoints(points):
    coords = []
    sorted_coords = np.zeros((7,7,2))
    for k in points.keys():
        v = points[k]
        x=v['x'][0]
        y=v['y'][0]
        coords.append([x,y])
    xc,yc = np.array(coords).T
    yc = yc[np.argsort(xc)].reshape((7,7))
    xc = np.sort(xc).reshape((7,7))
    for i in range(7):
        row_x = xc[i]
        row_y = yc[i]
        sorted_coords[i,:,0]=row_x[np.argsort(row_y)]
        sorted_coords[i,:,1]=np.sort(row_y)
    sorted_coords = sorted_coords.reshape(49,2)
    return sorted_coords

def projection(image,
               target_key_points,
               reference_key_points, 
               output_name=None,
               output_shape=(2000,2000),
               plot=True,
               save=False):
    target_aspect = target_key_points[-1]-target_key_points[0]
    ref_aspect = reference_key_points[-1]-reference_key_points[0]
        
    rescaled_ref = (reference_key_points-reference_key_points[0][np.newaxis,:])*((target_aspect/ref_aspect)[np.newaxis,:])+reference_key_points[0][np.newaxis,:]
    tform = transform.PiecewiseAffineTransform()
    tform.estimate(rescaled_ref,target_key_points)
    out = transform.warp(image, tform, output_shape=np.array(output_shape)+800)
    x,y = np.where(out>0)
    cropped = out[x.min():x.max(),y.min():y.max()]
    resized=min_max(transform.resize(cropped,(2000,2000),anti_aliasing=True))
    transformed = (resized*255).astype(np.uint8).T
    if save:
        tifffile.imsave(output_name,transformed,imagej=True)
    if plot:
        tifffile.imshow(transformed, cmap='gist_gray')

def min_max(data):
    return (data-data.min())/(data.max()-data.min())


def filter_by_growth(growth,min_gradient=-20,max_gradient=1000,min_dgrowth=25):
    accepted = False
    with np.errstate(divide='ignore',invalid='ignore'):
        g=np.array(growth)
        rel_dgrowth = (g[1:]-g[:-1])/g[:-1]
        filtered_rel_dgrowth = rel_dgrowth[(np.isnan(rel_dgrowth)|np.isinf(rel_dgrowth))==False]
        gradient = g[1:]-g[:-1]
    if ((g.max()-g.min())>min_dgrowth):
        if (filtered_rel_dgrowth.max()<6) and (filtered_rel_dgrowth.min()>-0.5):
            accepted=True
    return accepted
    
def exponential(x,a,r):
    return a*(1+r)**x


def exponential_fit(xdata,ydata,plot=True):
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    p0 = [0.1,0.1] # this is an mandatory initial guess
    try:
        popt, pcov = curve_fit(exponential, 
                               xdata, 
                               ydata, 
                               p0,
                               method='trf',
                               bounds=[(0.01,0.1),
                                       (10,1)],
                               ftol=5e-8,)
    except:
        popt, pcov = curve_fit(exponential, 
                               xdata, 
                               ydata, 
                               p0,
                               method='trf',
                               bounds=[(0.01,0.1),
                                       (10,1)],
                               ftol=5e-2,)
    a,r=popt
    interp_x=np.linspace(0,25,200)
    interp_y=exponential(interp_x,a,r)
    if plot:
        fig=plt.figure(figsize=(3,3))
        plt.scatter(xdata,ydata)
        plt.plot(interp_x,interp_y)
    return interp_x,interp_y,a,r

from scipy.optimize import curve_fit
from scipy.stats import linregress

def sigmoid(x, L ,x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return (y)

def fit_sigmoid(xdata,ydata,plot=True):
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    est_x0 = xdata[ydata<0.5*ydata.max()][-1]
    p0 = [ydata.max()+1,est_x0,2] # this is an mandatory initial guess
    try:
        popt, pcov = curve_fit(sigmoid, 
                               xdata, 
                               ydata, 
                               p0,
                               method='trf',
                               bounds=[(p0[0]/2,5,0.5),
                                       (p0[0]*10,27,3)],
                               ftol=1e-8,)
    except:
        popt, pcov = curve_fit(sigmoid, 
                               xdata, 
                               ydata, 
                               p0,
                               method='trf',
                               bounds=[(p0[0]/2,5,0.5),
                                       (p0[0]*10,27,3)],
                               ftol=5e-1,)
    L,x0, k=popt
    interp_x=np.linspace(8,28,200)
    interp_y=sigmoid(interp_x, L,x0, k)
    if plot:
        fig=plt.figure(figsize=(3,3))
        plt.scatter(xdata,ydata)
        plt.plot(interp_x,interp_y)
    return interp_x,interp_y,L,x0, k