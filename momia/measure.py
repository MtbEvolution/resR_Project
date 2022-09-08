from .helper_image import *
import os, glob
import pickle as pk

class Data:

    def __init__(self, record_id=0):
        self.cells = {}
        self.intensity_profiles = None
        self.count = None
        self.id = record_id
        self.data_df = pd.DataFrame()
        self.line_profiles = {}
        self.mesh_profiles = {}

    def load_cells_from_folder(self, src_folder):
        pickled_cells = sorted(glob.glob(src_folder+'*cells.pk'))
        if len(pickled_cells)>0:
            for f in pickled_cells:
                cell_dict = pk.load(open(f,'rb'))
                if len(cell_dict)>0:
                    self.cells.update(cell_dict)

    def _correct_xy_drift(self, max_drift=3, ref_channel='FITC'):
        """
        XY-drift correction at cellular level, may introduce numeric instability
        :return:
        """
        for k, c in self.cells.items():
            ref_img = c.get_ref_image()
            ref_img = 100+ref_img.max()-ref_img

            if ref_channel==None:
                for channel, img in c.data.items():
                    if channel != c.ref_channel:
                        shift, error, _diff = registration.phase_cross_correlation(ref_img,
                                                                                   img,
                                                                                   upsample_factor=10)
                        if np.max(np.abs(shift)) <= max_drift:
                            c.data[channel] = shift_image(img, shift)
            else:
                shift, error, _diff = registration.phase_cross_correlation(ref_img,
                                                                           c.data[ref_channel],
                                                                           upsample_factor=10)
                if np.max(np.abs(shift)) <= max_drift:
                    for channel, img in c.data.items():
                        if channel != c.ref_channel:
                            c.data[channel] = shift_image(img, shift)
            self.cells[k]=c

    def measure(self,
                morphology=True,
                intensity_basic=True,
                intensity_crosscorrelation=True):
        basic_data = []
        for k, c in self.cells.items():
            if len(c.midlines) > 0 and c.basic_features[-1] == 0:
                columns = []
                cell_data = []
                if morphology:
                    morph_col, morph_data = cell_morphology_basic(c)
                    columns += morph_col
                    cell_data += morph_data
                if intensity_basic:
                    intensity_col, intensity_data = cell_intensity_basic(c)
                    columns += intensity_col
                    cell_data += intensity_data
                if intensity_crosscorrelation:
                    corr_col, corr_data = cell_intensity_cross_correlation(c)
                    columns += corr_col
                    cell_data += corr_data
                basic_data.append(cell_data)
        self.data_df = pd.DataFrame(basic_data, columns=columns)

    def profile_midline(self):
        for i in self.data_df['Cell_id']:
            line_profile = {}
            cell = self.cells[i]
            midline = cell.midlines[0]
            for c, d in cell.data.items():
                if c is not cell.ref_channel:
                    line_profile[c] = measure_along_strip(midline, d, width=2)
            self.line_profiles[i] = line_profile

    def profile_orthogonal_mesh(self,scale=2):
        for k,cell in self.cells.items():
            datadict = {}
            try:
                mx,my = orthogonal_profiling(cell, scale)
                for channel, data in cell.data.items():
                    if channel != cell.ref_channel:
                        datadict[channel] = bilinear_interpolate_numpy(data,mx,my)
            except:
                pass
            self.mesh_profiles[k] = datadict

    def get_cell(self, idx):
        if isinstance(idx,str):
            return self.cells[idx]
        elif isinstance(idx,int):
            return self.cells[list(self.cells.keys())[idx]]
        else:
            raise ValueError('{} is not a valid index, should be either a'.format(idx) +\
                             'cell_id string or its numeric index')


def cell_morphology_basic(cell):
    if len(cell.midlines) > 0 and cell.basic_features[-1] == 0:
        features = cell.basic_features
        length = np.sum(np.array(cell.lengths))
        widths = np.concatenate(cell.width_lists)
        width_median = np.median(widths)
        width_std = np.std(widths[1:-1])
        width_max = np.max(widths)
        width_cv = width_std / width_median
        midline_sinuosity = sinuosity(cell.midlines[0])
        cell_data = [cell.id, length, width_median, width_std, width_cv, width_max, midline_sinuosity] + \
                    list(features[:-1])
        columns = ['Cell_id', 'length', 'width_median', 'width_std', 'width_cv', 'width_max', 'sinuosity'] + \
                  list(cell.basic_feature_attributes[:-1])
        return columns, cell_data
    else:
        return [], []


def intensity_profiling(data):
    from scipy.stats import skew, kurtosis
    mean = np.mean(data)
    median = np.median(data)
    max_v, min_v = np.max(data), np.min(data)
    std = np.std(data)
    cv_median = std / median
    cv_mean = std / mean
    skewness, kurtosis = skew(data), kurtosis(data)
    return [mean, median, max_v, min_v, std, cv_median, cv_mean, skewness, kurtosis]


def cell_intensity_basic(cell):
    attributes = ['mean', 'median', 'max', 'min', 'std', 'cv_median',
                  'cv_mean', 'skewness', 'kurtosis']
    coords = np.where(cell.mask > 0)
    channel_data = []
    columns = []
    cell.intensity_data = {}
    for c, d in cell.data.items():
        if c != cell.ref_channel:
            intensity = d[coords]
            channel_data += intensity_profiling(intensity)
            columns += ['{}_{}'.format(c, x) for x in attributes]
            cell.intensity_data[c] = intensity
    return columns, channel_data


def cell_intensity_cross_correlation(cell):
    from itertools import combinations
    from scipy.stats import pearsonr
    channel_pairs = list(combinations(cell.intensity_data.keys(), 2))
    corr_columns = []
    corr_data = []
    for c1, c2 in channel_pairs:
        d1, d2 = cell.intensity_data[c1], cell.intensity_data[c2]
        r, p = pearsonr(d1, d2)
        corr_data += [r, round(p, 5)]
        corr_columns += ['{}_{}_pearsonr'.format(c1, c2),
                         '{}_{}_pearsonp'.format(c1, c2)]
    return corr_columns, corr_data

def sinuosity(midline):
    if (midline[0]-midline[-1]).sum() == 0:
        raise ValueError('Midline coordinates appear to be closed!')
    end_to_end_dist = distance(midline[0],midline[-1])
    length = line_length(midline)
    ret = round(length/end_to_end_dist, 3)
    if ret < 1:
        ret = 1.0
    return ret

def measure_along_strip(line, img, width = 1, subpixel = 0.5):
    warnings.filterwarnings("ignore")
    unit_dxy = unit_perpendicular_vector(line, closed=False)
    width_normalized_dxy = unit_dxy * subpixel
    copied_img = img.copy()
    data = bilinear_interpolate_numpy(copied_img, line.T[0], line.T[1])
    for i in range(1, 1+int(width * 0.5 / subpixel)):
        dxy = width_normalized_dxy * i
        v1 = line + dxy
        v2 = line - dxy
        p1 = bilinear_interpolate_numpy(copied_img, v1.T[0], v1.T[1])
        p2 = bilinear_interpolate_numpy(copied_img, v2.T[0], v2.T[1])
        data = np.vstack([p1, data, p2])
    return np.average(data, axis=0)


def orthogonal_profiling(cell,scale=2):
    midline = cell.midlines[0]
    contour = cell.optimized_contours[0]
    width = cell.width_lists[0]
    edge_points = []
    subpixel_midline = spline_approximation(midline,n=int(scale*(len(midline))),
                                            smooth_factor=1,closed=False)
    unit_perp=unit_perpendicular_vector(subpixel_midline)
    for i in range(1,len(subpixel_midline)-1):
        # make core mesh
        p=line_contour_intersection(subpixel_midline[i],
                                subpixel_midline[i]+unit_perp[i],contour)
        flip = point_line_orientation(subpixel_midline[i],subpixel_midline[i+1],p[0])
        if flip==-1:
            p=np.flip(p,axis=0)
        edge_points.append(p.flatten())
    median_width = np.median(width)
    width_segments = int(scale*median_width)
    edge_points=np.array(edge_points)
    x,y = edge_points[:,0],edge_points[:,1]
    dxy = (edge_points[:,2:]-edge_points[:,:2])/width_segments
    mat = np.tile(np.arange(width_segments+1),(len(edge_points),1))
    mat_x = x[:,np.newaxis]+mat*dxy[:,0][:,np.newaxis]
    mat_y = y[:,np.newaxis]+mat*dxy[:,1][:,np.newaxis]
    return np.array([mat_x,mat_y])