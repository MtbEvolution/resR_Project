from .particle import Particle
from .patch import Patch
from .helper_image import *
import os,glob,shutil
import pickle as pk


class Image:

    def __init__(self):
        self.patch = Patch()
        self.cells = {}
        self.selected_features = ['area', 'eccentricity', 'solidity', 'touching_edge',
                                  'convex_area', 'filled_area', 'eccentricity', 'solidity', 'major_axis_length',
                                  'minor_axis_length', 'perimeter', 'equivalent_diameter',
                                  'extent', 'circularity', 'aspect_ratio',
                                  'moments_hu-0', 'moments_hu-1', 'moments_hu-2', 'moments_hu-3', 'moments_hu-4',
                                  'moments_hu-5', 'moments_hu-6', 'aspect_ratio',
                                  'bending_std', 'bending_90', 'bending_10', 'bending_max',
                                  'bending_min', 'bending_skewness', 'bending_median','touching_edge']

    def load_data(self,
                  image_id,
                  image_dict,
                  config,
                  ref_channel=-1):
        self.patch.load_data(image_id=image_id,
                             image_dict=image_dict,
                             ref_channel=ref_channel)
        self.patch.load_config(config=config)

    def _autoRun(self, classifier):
        self.patch.crop_edge()
        self.patch.enhance_fluorescence()
        self.patch.segmentation_basic()
        self.patch.find_contours()
        self.patch._measure_contour()
        self.patch._measure_RoG(s1=0.5, s2=10)
        self.patch._hu_moments_log_transform()
        self.patch._predict(classifier)
        for i in self.patch._get_class_index(0):
            bbox = self.patch.get_cluster_roi(i)
            binary = self.patch._get_cluster_mask(i)
            data = self.patch._get_roi_data(bbox)
            contours = self.patch.contours[i]
            cell = Particle(bbox, binary, contours, data, self.patch.ref_channel,
                               '{}_{}_0'.format(self.patch.id, i),
                               self.patch.pixel_microns)
            cell.load_basic_features(self.patch.regionprop_table.loc[i], self.selected_features)
            cell.optimize_contours()
            cell.extract_skeleton()
            cell.find_midline()
            self.cells[cell.id] = cell
        for j in self.patch._get_class_index(1):
            bbox = self.patch.get_cluster_roi(j)
            binary = self.patch._get_cluster_mask(j)
            miniPatch = Patch()
            miniPatch.inherit(j, self.patch, bbox, binary)
            miniPatch.segmentation_shapeindex()
            if len(miniPatch.regionprop_table) > 1 and self.patch.regionprop_table.loc[j]['touching_edge'] == 0:
                miniPatch.find_contours(level=0.1)
                miniPatch._measure_RoG(s1=0.5, s2=10)
                miniPatch._measure_contour()
                miniPatch._hu_moments_log_transform()
                miniPatch._predict(classifier)
                for k in miniPatch._get_class_index(0):
                    babyBbox = miniPatch.get_cluster_roi(k)
                    babyMask = miniPatch._get_cluster_mask(k)
                    babyContours = miniPatch.contours[k]
                    babyData = miniPatch._get_roi_data(babyBbox)
                    cell = Particle(babyBbox, babyMask, babyContours, babyData,
                                       self.patch.ref_channel,
                                       '{}_{}_{}'.format(self.patch.id, j, k),
                                       miniPatch.pixel_microns)
                    cell.load_basic_features(miniPatch.regionprop_table.loc[k], self.selected_features)
                    cell.optimize_contours()
                    cell.extract_skeleton()
                    cell.find_midline()
                    cell.offset = bbox[:2]
                    self.cells[cell.id] = cell

    def plotSegmentation(self, dst_folder, dpi=160, image_size=(15, 15),
                         contour_color='orange',
                         midline_color='orange',
                         warning_color='red', lw=0.5,
                         savefig=True):
        fig = plt.figure(figsize=image_size)
        plt.imshow(self.patch.get_ref_image(), cmap='gist_gray', vmax=30000, vmin=1000)
        for cell in self.cells.values():
            x1, y1, x2, y2 = cell.bbox
            offset = cell.offset
            contour = cell.optimized_contours[0]
            if len(cell.midlines) > 0:
                midline = cell.midlines[0]
                plt.plot(contour[:, 1] + y1 + offset[1], contour[:, 0] + x1 + offset[0], color=contour_color, lw=lw)
                plt.plot(midline[:, 1] + y1 + offset[1], midline[:, 0] + x1 + offset[0], color=midline_color, lw=lw)
            else:
                plt.plot(contour[:, 1] + y1 + offset[1], contour[:, 0] + x1 + offset[0], color=warning_color, lw=lw)
        plt.axis('off')
        if savefig:
            plt.savefig('{}{}_segmentation.png'.format(dst_folder, self.patch.id), dpi=dpi, bbox_inches='tight')
            plt.close()

    def dump(self, dst_folder):
        pk.dump(self.cells, open('{}{}_cells.pk'.format(dst_folder, self.patch.id),'wb'))
