from skimage import draw, measure
import numpy as np
from matplotlib import pyplot as plt, cm, colors
from matplotlib.gridspec import GridSpec as gs

def standardize_punclist(punc):
    if len(punc)==0:
        return np.array([None]*6)
    else:
        return punc
    
def create_canvas(width=101, height=600):
    canvas = np.zeros((width, height))
    r = int(width / 2)
    rr1, cc1 = draw.ellipse(r, int(r * 1.6), r, int((r) * 1.6))
    rr2, cc2 = draw.ellipse(r, int(height - 1.6 * r), r, int((r) * 1.6))
    rr3, cc3 = draw.rectangle(start=(1, int(r * 1.6)), end=(width - 2, height - int(r * 1.6)),
                                        shape=canvas.shape)
    canvas[rr3, cc3] = 1
    canvas[rr1, cc1] = 1
    canvas[rr2, cc2] = 1
    l = len(np.nonzero(np.sum(canvas, axis=0))[0])
    counter = 0
    canvas = canvas.T
    return canvas


def create_mesh(canvas):
    xt, yt = np.nonzero(canvas)
    l, m = np.sum(canvas, axis=0), np.sum(canvas, axis=1)
    norm_yt = np.zeros(xt.shape)
    norm_xt = (xt - xt.min()) / (l.max() - 1)
    count = 0
    for i in range(len(xt)):
        r = m.max() / m[xt[i]]
        norm_yt[i] = count * r / (m.max() - 1)
        if i != len(xt) - 1:
            if xt[i + 1] == xt[i]:
                count += 1
            else:
                count = 0
        else:
            count += 1
    return (xt, yt, norm_xt, norm_yt)


def project_image(xt, yt, norm_xt, norm_yt, canvas, data):
    paint = np.zeros(canvas.shape)
    xid = norm_xt.copy()
    yid = norm_yt.copy()
    xid *= data.shape[0] - 1
    yid *= data.shape[1] - 1
    interpolated = bilinear_interpolate_numpy(data, xid, yid)
    paint[xt, yt] = interpolated
    return (paint)

def initiate_projection():
    width = 75
    heights = [250, 350, 450, 550, 650]
    pad = np.zeros((50, width))
    gap = np.zeros((750, 10))
    padded = [gap]
    xt_list, yt_list, nxt_list, nyt_list = [], [], [], []
    for i in range(len(heights)):
        a = create_canvas(width=width, height=heights[i])
        half_pad = np.tile(pad, (len(heights) - i, 1))
        m_pad = np.concatenate([half_pad, a, half_pad], axis=0)
        m_pad = np.concatenate([m_pad, gap], axis=1)
        xt, yt, norm_xt, norm_yt = create_mesh(m_pad)
        xt_list.append(xt)
        yt_list.append(yt)
        nxt_list.append(norm_xt)
        nyt_list.append(norm_yt)
        padded.append(m_pad)
    contours = measure.find_contours(np.concatenate(padded, axis=1), level=0)
    optimized_outline = []
    for contour in contours:
        optimized_outline.append(spline_approximation(contour, n=2 * len(contour)))
    return padded, xt_list, yt_list, nxt_list, nyt_list, optimized_outline


def bilinear_interpolate_numpy(im, x, y):
    """
    bilinear interpolation 2D
    :param im: target image
    :param x: x coordinates
    :param y: y coordinates
    :return: interpolated data
    """
    h,l = im.shape
    padded = np.zeros((h+1,l+1))
    padded[:h,:l] += im
    im = padded
    x0 = x.astype(int)
    x1 = x0 + 1
    y0 = y.astype(int)
    y1 = y0 + 1
    Ia = im[x0,y0]
    Ib = im[x0,y1]
    Ic = im[x1,y0]
    Id = im[x1,y1]
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    return np.round((Ia*wa) + (Ib*wb) + (Ic*wc) + (Id*wd), 4)


def spline_approximation(init_contour, n=200, smooth_factor=1, closed=True):
    from scipy.interpolate import splprep, splev, RectBivariateSpline
    """
    spline func.
    :param init_contour:
    :param n:
    :param smooth_factor:
    :param closed:
    :return:
    """
    if closed:
        tck, u = splprep(init_contour.T, u=None, s=smooth_factor, per=1)
    else:
        tck, u = splprep(init_contour.T, u=None, s=smooth_factor)
    u_new = np.linspace(u.min(), u.max(), n)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.array([x_new, y_new]).T


def standardize_punclist(punc):
    if len(punc)==0:
        return np.array([None]*6)
    else:
        return punc
    
def filter_pundata(punc,threshold):
    if None in punc:
        return punc
    else:
        return standardize_punclist(punc[punc[:,2]>=threshold])
    
    
def punc_string_parser(punc_data):
    punc=str(punc_data)
    new_list = []
    counter = 0
    substring = ''
    for x in str(punc):
        if x not in '1234567890e-+.':
            if counter != 0:
                new_list.append(substring)
                substring=''
                counter=0
        else:
            substring+=x
            counter+=1
    new_list = np.array(new_list).reshape(int(len(new_list)/6),6).astype(float)
    return new_list


def adjust_relative_width(x,y,normx,normy):
    p1=np.sum((normx[np.newaxis,:]-y[:,np.newaxis])<0,axis=1)
    p0=p1-1
    w = normy[p0]+(normy[p1]-normy[p0])*(y-normx[p0])/(normx[p1]-normx[p0])
    return x*w

def puncta_heatmaps(df,optimized_outline,padded,coords,coord2w_list,
                    channels=['TRITC','CY5'],targets=['1','2'],cutoffs=[300,100]):
    from scipy.stats import gaussian_kde
    mRNA_loc = plt.figure(figsize=(4*len(channels)+0.2, 11.2))
    grids = gs(2,len(channels))
    for k,c in enumerate(channels):
        target = targets[k]
        cutoff = cutoffs[k]
        ax1 = mRNA_loc.add_subplot(grids[0,k])
        ax2 = mRNA_loc.add_subplot(grids[1,k])
        fg = np.concatenate(padded, axis=1)
        
        bg_scatter,bg_heatmap = np.zeros(fg.shape), np.zeros(fg.shape)
        ax1.imshow(bg_scatter, cmap='Greys', aspect='auto')
        ax1.set_xlim(0, bg_scatter.shape[1]-1)
        for outline in optimized_outline:
            ax1.plot(outline.T[1], outline.T[0], c="black", alpha=1,lw=0.5)
            ax2.plot(outline.T[1], outline.T[0], c="white", alpha=1,lw=0.5)
        median_lengths = []
        for i,frac in enumerate([0,0.2,0.4,0.6,0.8]):
            l1 = df['length'].quantile(frac)
            l2 = df['length'].quantile(frac+0.2)
            subset = df[(df['length']<l2)&(df['length']>=l1)]
            median_lengths.append(subset['length'].median())
            punc_list = []
            for punc in subset['{}_puncta_data'.format(c)]:
                punc = str(punc)
                if 'None' not in punc:
                    punc_list.append(punc_string_parser(punc))
            if len(punc_list)>2:
                punc_list = np.vstack(punc_list)
                if len(punc_list)>2:
                    punc_list = punc_list[punc_list[:,2]>=cutoff]
            if len(punc_list)>2:
                v,x,y = punc_list[:,np.array([2,4,5])].T
                norm_v = (v)/np.percentile(v,85)
                norm_v[norm_v<0]=0
                norm_v[norm_v>1]=1
                normx,normy=coord2w_list[i]
                x0, y0, l, w = coords[4-i]
            
                x = adjust_relative_width(x,y,normx,normy)
                # plot scatter
                ax1.scatter(x*w+x0,(y-0.5)*l+y0,s=norm_v*20,alpha=0.5,fc=cm.get_cmap('Blues')(norm_v))
            
                # plot heatmap
                data = np.array([x*w+x0,(y-0.5)*l+y0]).T
                xgrid,ygrid,X,Y = construct_grids(x0,w,y0,l,1)
                kernel = gaussian_kde(data.T,weights=v)
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(kernel(positions).T, X.shape)
            
                # normalize density
                Z /= Z.max()
                bg_heatmap[Y,X]=Z
        bg_heatmap[fg==0]=0
        ax2.imshow(bg_heatmap, cmap='magma', aspect='auto')
        ax1.set_xticks(np.array(coords)[:,0])
        ax1.set_xticklabels(np.flip(np.round(median_lengths,1)),fontsize=14)
        ax1.set_xlabel('median length [μm]',fontsize=14)
        ax2.set_xticks(np.array(coords)[:,0])
        ax2.set_xticklabels(np.flip(np.round(median_lengths,1)),fontsize=14)
        ax2.set_xlabel('median length [μm]',fontsize=14)
        ax1.set_title('{}-$\it{}$'.format(c,target),fontsize=14)
        ax1.set_yticks([])
        ax2.set_yticks([])
    return mRNA_loc

def construct_grids(x0,w,y0,l,gridsize=1):
    xmin = int(x0-w*0.5)
    xmax = int(x0+w*0.5)
    ymin = int(y0-l*0.5)
    ymax = int(y0+l*0.5)
    xgrid = np.arange(xmin,xmax,gridsize)
    ygrid = np.arange(ymin,ymax,gridsize)
    X, Y = np.meshgrid(xgrid,ygrid)
    return xgrid,ygrid,X,Y