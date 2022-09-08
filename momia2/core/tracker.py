from scipy.ndimage import median_filter
import trackpy
from ..classify import prediction2foreground,prediction2seed
from ..utils import generic,skeleton
from ..metrics import image_feature
from skimage import filters, morphology, measure, feature, segmentation
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
    
class CellTracker:
    
    def __init__(self,
                 sorted_patches, #pre-loaded patches as list
                 time_points,
                 pixel_classifier,
                 seed_prob_min=0.2,
                 edge_prob_max=0.9,
                 background_prob_max=0.2,
                 min_overlap_threshold = 0.5,
                 min_overlap_area=100,
                 min_size_similarity=0.75,
                 min_cell_size=200,
                 min_seed_size=10,
                 max_iter=5,
                 cache_image_features=False,
                 verbose = True,
                 backtrack_generations=4):
        
        if len(time_points) != len(sorted_patches):
            raise ValueError("The number of time points given doesn't match the number of image patches!")
        self.timepoints = time_points
        self.frames = sorted_patches
        self.pixel_classifier = pixel_classifier
        self.seed_prob_min = seed_prob_min
        self.edge_prob_max = edge_prob_max
        self.background_prob_max = background_prob_max
        self.min_seed_size = min_seed_size
        self.cache_in = cache_image_features
        
        self.min_overlap_threshold = min_overlap_threshold
        self.min_overlap_area = min_overlap_area
        self.min_size_similarity = min_size_similarity
        self.min_cell_size = min_cell_size
        self.backtrack_generations=backtrack_generations
        self.max_iter = max_iter
        
        self.regionprops = None
        self.overlap_stats = None
        self.to_be_merged = [-1]
        self.to_be_split = [-1]
        self.verbose = verbose
        
    def run_segmentation(self):
        for i,p in enumerate(self.frames):
            if p.labeled_mask is None:
                if self.verbose:
                    print('Running segmentation on frame {}...'.format(i))
                p.pixel_classification(self.pixel_classifier,
                           seed_prob_min=self.seed_prob_min,
                           edge_prob_max=self.edge_prob_max,
                           background_prob_max=self.background_prob_max,
                           cache_in=self.cache_in,
                           min_seed_size=self.min_seed_size)
        
    def locate_spurious_events(self):
        to_be_split,to_be_merged = _locate_spurious_events(self.regionprops)        
        self.to_be_split=to_be_split
        self.to_be_merged=to_be_merged
    
    def update_masks(self):
        if len(self.to_be_merged)>0:
            for c in self.to_be_merged:
                self._merge_cells(c)
        if len(self.to_be_split)>0:
            for c in self.to_be_split:
                self._split_cells(c)
    
    def update_frames(self):
        _regionprops=[]
        for i,p in enumerate(self.frames):
            p.locate_particles()
            p.regionprops['$cell']=['{}_{}'.format(i,l) for l in p.regionprops.index]
            p.regionprops['$time']=i
            _regionprops.append(p.regionprops)
        self.regionprops = pd.concat(_regionprops).set_index('$cell')
    
    def _update_regionprops(self):
        _regionprops=[]
        for i,p in enumerate(self.frames):
            _regionprops.append(p.regionprops)
            p.regionprops['$cell']=['{}_{}'.format(i,l) for l in p.regionprops.index]
        _regionprops = pd.concat(_regionprops).set_index('$cell')
        for col in _regionprops.columns:
            if col not in self.regionprops.columns:
                self.regionprops[col] = _regionprops.loc[self.regionprops.index][col]
        del _regionprops
            
    def trace_by_overlap(self):
        counter = 0
        while counter<=self.max_iter:
            self.update_frames()
            self.link_cells_by_overlap()
            self.locate_spurious_events()
            self.update_masks()
            if self._end_iter():
                self.update_frames()
                self.link_cells_by_overlap()
                break
            counter+=1
            
    def link_cells_by_overlap(self):
        self.regionprops,self.overlap_stats = overlap_match(self.frames,
                                                             self.regionprops,
                                                             min_cell_size=self.min_cell_size,
                                                             min_size_similarity=self.min_size_similarity,
                                                             min_overlap_threshold=self.min_overlap_threshold,
                                                             min_overlap_area = self.min_overlap_area)
        
    def _end_iter(self):
        end = False
        if (len(self.to_be_merged)+len(self.to_be_split))==0:
            end = True
        return end
    
    def _merge_cells(self, cell_list):

        coords = self.regionprops.loc[np.array(cell_list)]['$coords']
        dilated_coords = [dilate_by_coords(x,self.frames[0].shape,morphology.disk(1)) for x in coords]
        unique_coords,pix_count = np.unique(np.vstack(dilated_coords),return_counts=True,axis=0)
        border_coords = unique_coords[pix_count>=2]
        merged_coords = np.vstack(list(coords)+[border_coords])
        time = np.unique([int(x.split('_')[0]) for x in cell_list])[0]
        new_label = np.min([int(x.split('_')[1]) for x in cell_list])
        self.frames[time].labeled_mask[merged_coords[:,0],merged_coords[:,1]]=new_label
    
    def _split_cells(self, cell_ref):
        cell, n_daughter = cell_ref
        time,label = np.array(cell.split('_')).astype(int)
        x,y = self.regionprops.loc[cell]['$coords'].T
        cropped_prob = np.zeros(self.frames[time].prob_mask.shape)
        cropped_mask = np.zeros(self.frames[time].mask.shape)
        cropped_mask[x,y]=1
        cropped_prob[x,y,:] = self.frames[time].prob_mask[x,y,:]
        for th in np.linspace(self.seed_prob_min,1,10):
            seed = prediction2seed(cropped_prob,
                                   seed_min=th,
                                   edge_max=self.edge_prob_max,
                                   min_seed_size=self.min_seed_size)

            if seed.max()>=n_daughter:
                break
        if seed.max()!= n_daughter:
            last_th = th
            for th in np.linspace(self.seed_prob_min,last_th,40):
                seed = prediction2seed(cropped_prob,
                                       seed_min=th,
                                       edge_max=self.edge_prob_max,
                                       min_seed_size=self.min_seed_size)

                if seed.max()==n_daughter:
                    break
        if seed.max()==n_daughter:
            seed[seed==1]=label
            seed[seed==2]=self.frames[time].labeled_mask.max()+1
            cropped_watershed = segmentation.watershed(image=cropped_prob[:,:,1],
                                                          mask=cropped_mask,
                                                          markers=seed,
                                                          connectivity=1,
                                                          compactness=0.1,
                                                          watershed_line=True)
            new_labeled_mask = self.frames[time].labeled_mask.copy()
            new_labeled_mask[x,y]=cropped_watershed[x,y]
            self.frames[time].labeled_mask = new_labeled_mask
        
    def plot_cell(self,cell_id):
        show_highlight(cell_id,self.frames)
        
    def trace_lineage(self):
        rev_lineage = {}
        init_cell_counter = 1
        for c in self.regionprops.index:
            daughters = self.regionprops.loc[c]['daughter(s)']
            if c not in rev_lineage:
                rev_lineage[c]=str(init_cell_counter)
                init_cell_counter+=1
            if len(daughters)==1:
                rev_lineage[daughters[0]]=rev_lineage[c]
            else:
                for i,d in enumerate(daughters):
                    rev_lineage[d] = '{}.{}'.format(rev_lineage[c],i+1)
        lineage = [rev_lineage[c] for c in self.regionprops.index]
        self.regionprops['cell_lineage'] = lineage
        
def dilate_by_coords(coords,image_shape,
                     selem=morphology.disk(2)):
    h,w = image_shape
    if len(selem)==1:
        return coords
    elif len(selem)>1:
        x,y=image_feature.coord2mesh(coords[:,0],
                                     coords[:,1],
                                     selem=selem)
        x[x<0]=0
        x[x>h-1]=h-1
        y[y<0]=0
        y[y>w-1]=w-1
        xy = np.unique(np.array([x.ravel(),y.ravel()]).T,axis=0)
        return xy

def back_track(regionprops,
               cell_id,
               n_generations=3):
    # the backtrack algorithm is based on the naive hypothesis that an fragments of an oversegmented cell
    # should have a common ancester tracing back n generations whereas an undersegmented cell should not.
    
    time,label = np.array(cell_id.split('_')).astype(int)
    elderlist = []
    current_cell = np.array([cell_id])
    while True:
        if n_generations==0 or time ==0:
            break
        mothers = np.unique(np.hstack(regionprops.loc[current_cell]['mother(s)'].values))
        current_cell = mothers
        n_generations-=1
        time -= 1
        if len(mothers)>0:
            elderlist.append(mothers)
        else:
            break
    return elderlist

def area_ratio(areas):
    a1,a2=areas
    if a1>a2:
        ratio = a2/a1
    else:
        ratio = a1/a2
    return ratio

def show_highlight(cell_id,frames):
    t,label = np.array(cell_id.split('_')).astype(int)
    mask = (frames[t].labeled_mask>0)*1
    mask[frames[t].labeled_mask==label]=3
    fig=plt.figure(figsize=(10,10))
    plt.imshow(mask)
    
def overlap_match(frames,regionprops,
                  min_overlap_threshold = 0.5,
                  min_overlap_area=100,
                  min_size_similarity=0.75,
                  min_cell_size=200):
    
    # link cells by overlap
    init_stat = {k:[[],[]] for k in regionprops.index}
    mother_list = []
    daughter_list = []
    rp = regionprops.copy()
    overlap_stat = []
    for t in range(0,len(frames)-1):
        df = _calculate_overlap_matrix(frames[t].labeled_mask,
                                       frames[t+1].labeled_mask,t,t+1)
        filtered_df = df[np.max(df[['frame1_overlap_frac','frame2_overlap_frac']].values,axis=1)>min_overlap_threshold]
        for (cell1,cell2) in filtered_df[['frame1_id','frame2_id']].values:
            init_stat[cell1][1] += [cell2]
            init_stat[cell2][0] += [cell1]
        overlap_stat.append(df)
    
    overlap_stat=pd.concat(overlap_stat)
    
    # find missing mother(s) for cells that moved a bit
    orphans_candidates = rp[(rp['$time']>0)&(rp['$touching_edge']==0)].index
    for o in orphans_candidates:
        if len(init_stat[o][0])==0 and len(init_stat[o][1])>0:
            subset = overlap_stat[overlap_stat['frame2_id']==o].values
            missing_mother=[]
            for s in subset:
                cond1 = s[2]>min_overlap_area
                cond2 = area_ratio((s[3],s[4]))>min_size_similarity
                cond3 = min(s[3],s[4])>min_cell_size
                if cond1*cond2*cond3:
                    missing_mother += [s[0]]
                    init_stat[s[0]][1] += [o]
            init_stat[o][0]=missing_mother            
    rp[['mother(s)','n_mother']]=[[np.unique(init_stat[k][0]),len(np.unique(init_stat[k][0]))] for k in rp.index]
    rp[['daughter(s)','n_daughter']]=[[np.unique(init_stat[k][1]),len(np.unique(init_stat[k][1]))] for k in rp.index]
    
    
    return rp,overlap_stat

def _calculate_overlap_matrix(frame1_labeled_mask,
                              frame2_labeled_mask,
                              frame1_label,
                              frame2_label):
    f1,f2 = frame1_labeled_mask.ravel(),frame2_labeled_mask.ravel()
    f1f2 = np.array([f1,f2]).T[f1*f2 !=0]
    f1_counts = _unique2dict1D(f1)
    f2_counts = _unique2dict1D(f2)
    neighbors,overlap = np.unique(f1f2,return_counts=True,axis=0)
    f1_areas = np.array([f1_counts[i] for i in neighbors[:,0]])
    f2_areas = np.array([f2_counts[i] for i in neighbors[:,1]])
    
    f1_id = np.array(['{}_{}'.format(frame1_label,i) for i in neighbors[:,0]])
    f2_id = np.array(['{}_{}'.format(frame2_label,i) for i in neighbors[:,1]])
    overlap_matrix = np.hstack([f1_id.reshape(-1,1),
                                f2_id.reshape(-1,1),
                                overlap.reshape(-1,1),
                                f1_areas.reshape(-1,1),
                                f2_areas.reshape(-1,1),
                                (overlap/f1_areas).reshape(-1,1),
                                (overlap/f2_areas).reshape(-1,1),
                                (overlap/(f1_areas+f2_areas-overlap)).reshape(-1,1)])
    overlap_df = pd.DataFrame()
    overlap_df['frame1_id']=f1_id
    overlap_df['frame2_id']=f2_id
    overlap_df['overlap_area']=overlap
    overlap_df['frame1_area']=f1_areas
    overlap_df['frame2_area']=f2_areas
    overlap_df['frame1_overlap_frac']=overlap/f1_areas
    overlap_df['frame2_overlap_frac']=overlap/f2_areas
    overlap_df['iou']=overlap/(f1_areas+f2_areas-overlap)
    return overlap_df

def _unique2dict1D(array,nonzero=True):
    copied = array.copy()
    if nonzero:
        copied=copied[copied!=0]
    vals,counts = np.unique(copied,return_counts=True)
    return {v:c for v,c in zip(vals,counts)}

def _locate_spurious_events(regionprops,
                            n_generations=4,
                            verbose=True):
    to_be_merged = []
    to_be_split = []
    for cell in regionprops[regionprops['n_mother']>=2].index:
        back_track_record = back_track(regionprops,cell,n_generations=n_generations)
        if len(back_track_record[-1])==1:
            for entry in back_track_record:
                if len(entry)>=2:
                    if verbose:
                        print('Merge cells: {}'.format(', '.join(list(entry))))
                    to_be_merged.append(list(entry))
        else:
            if verbose:
                print('Split cell {} into {} fragments.'.format(cell,regionprops.loc[cell,'n_mother']))
            to_be_split.append([cell,regionprops.loc[cell,'n_mother']])
    return to_be_split, to_be_merged