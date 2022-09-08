from itertools import combinations
from .helper_image import *
from matplotlib.gridspec import GridSpec as gs
class Boundary:

    def __init__(self, p1, p2, xcoords, ycoords):
        self.p1 = p1
        self.p2 = p2
        self.x = xcoords
        self.y = ycoords
        self.false_boundary = False
        self.data = {}

    def measure(self, patch):
        x, y = np.array(self.x), np.array(self.y)
        if patch.ridge is not None:
            intensity_headers = ['RoG', 'Ridge', 'Shapeindex']
            values = [patch.RoG[x, y], patch.ridge[x, y], patch.shapeindex[x, y]]
        else:
            intensity_headers = ['RoG', 'Shapeindex']
            values = [patch.RoG[x, y], patch.shapeindex[x, y]]
        for i, v in enumerate(values):
            self.data['{}_mean'.format(intensity_headers[i])] = v.mean()
            self.data['{}_std'.format(intensity_headers[i])] = v.std()
            self.data['{}_max'.format(intensity_headers[i])] = v.max()
            self.data['{}_min'.format(intensity_headers[i])] = v.min()
        self.data['length'] = len(self.x)
        ang1, ang2 = patch.regionprop_table.loc[[self.p1, self.p2], 'orientation']
        self.data['angle_between_neighbors'] = min_angle(ang1, ang2)

    def _plot_boundary(self, patch, filename=None):
        _boundary_neighbors = patch.regionprop_table.loc[[self.p1, self.p2]]
        x1 = min(_boundary_neighbors['opt-x1'])
        y1 = min(_boundary_neighbors['opt-y1'])
        x2 = max(_boundary_neighbors['opt-x2'])
        y2 = max(_boundary_neighbors['opt-y2'])
        x = np.array(self.x)
        y = np.array(self.y)
        shape = (x2 - x1, y2 - y1)

        fig = plt.figure(figsize=(shape[1] * 0.06, shape[0] * 0.03))
        grids = gs(1, 2, wspace=0.1)
        ax1 = fig.add_subplot(grids[:, 0])
        ax2 = fig.add_subplot(grids[:, 1])
        ax1.axis('off')
        ax2.axis('off')
        mask = ((patch.cluster_labels == self.p1) + (patch.cluster_labels == self.p2)) * 1
        mask[x, y] = 2
        ax1.imshow(mask[x1:x2, y1:y2], cmap='gist_gray')
        ax2.imshow(patch.get_ref_image()[x1:x2, y1:y2], cmap='gist_gray')
        ax2.scatter(y - y1, x - x1, s=2, fc='salmon', alpha=0.8)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()


@jit()
def _adjust_coords(x, y, xmax, ymax):
    x1 = x - 1
    x1[x1 < 0] = 0
    x2 = x + 1
    x2[x2 > xmax - 1] = xmax - 1
    y1 = y - 1
    y1[y1 < 0] = 0
    y2 = y + 1
    y2[y2 > ymax - 1] = ymax - 1
    return x1, x2, y1, y2


@jit(nopython=True)
def _find_unique_neighbors_connect1(labels, x, y, x1, x2, y1, y2):
    unique_neighbors = []
    xlist = []
    ylist = []
    for i in range(len(x)):
        values = [labels[x1[i], y[i]], labels[x2[i], y[i]],
                  labels[x[i], y1[i]], labels[x[i], y2[i]]]
        vlist = [v for v in values if v > 0]
        vlist = sorted(list(set(vlist)))
        if len(vlist) > 1:
            unique_neighbors.append(vlist)
            xlist.append(x[i])
            ylist.append(y[i])
    return xlist, ylist, unique_neighbors


@jit(nopython=True)
def _find_unique_neighbors_connect2(labels, x, y, x1, x2, y1, y2):
    unique_neighbors = []
    xlist = []
    ylist = []
    for i in range(len(x)):
        values = [labels[x1[i], y1[i]], labels[x[i], y1[i]], labels[x2[i], y1[i]],
                  labels[x1[i], y[i]], labels[x2[i], y[i]],
                  labels[x1[i], y2[i]], labels[x[i], y2[i]], labels[x2[i], y2[i]]]

        vlist = [v for v in values if v > 0]
        vlist = sorted(list(set(vlist)))
        if len(vlist) > 1:
            unique_neighbors.append(vlist)
            xlist.append(x[i])
            ylist.append(y[i])
    return xlist, ylist, unique_neighbors


def _boundary2neighbors(xlist, ylist, unique_neighbors):
    particle_pairwise_dict = {}
    boundary_pairwise_dict = {}
    n = 0
    idx = 0
    for i, neighbors in enumerate(unique_neighbors):
        x, y = xlist[i], ylist[i]
        for pairs in combinations(neighbors, 2):
            p1, p2 = pairs
            if p1 not in particle_pairwise_dict:
                particle_pairwise_dict[p1] = {p2: n}
                n += 1
            elif p2 not in particle_pairwise_dict[p1]:
                particle_pairwise_dict[p1][p2] = n
                n += 1
            idx = particle_pairwise_dict[p1][p2]
            if idx not in boundary_pairwise_dict:
                boundary_pairwise_dict[idx] = Boundary(p1, p2, [x], [y])
            else:
                boundary_pairwise_dict[idx].x += [x]
                boundary_pairwise_dict[idx].y += [y]
    return particle_pairwise_dict, boundary_pairwise_dict


def annotate_boundaries(watershed,
                        mask,
                        connectivity=2):
    x, y = np.where((mask > 0) & (watershed == 0))
    xmax, ymax = mask.shape
    x1, x2, y1, y2 = _adjust_coords(x, y, xmax, ymax)
    if connectivity == 1:
        xlist, ylist, unique_neighbors = _find_unique_neighbors_connect1(watershed, x, y, x1, x2, y1, y2)
    elif connectivity == 2:
        xlist, ylist, unique_neighbors = _find_unique_neighbors_connect2(watershed, x, y, x1, x2, y1, y2)
    return _boundary2neighbors(xlist, ylist, unique_neighbors)


def min_angle(theta1, theta2):
    return min(abs(theta1 + theta2), abs(theta1 - theta2))