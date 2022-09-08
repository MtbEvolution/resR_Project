from matplotlib import pyplot as plt
import tifffile
from matplotlib import cm
from .helper_generic import *


def _plot_contour(ax,contour,**kwargs):
    lw = set_default_by_kwarg('lw',kwargs,1)
    color = set_default_by_kwarg('color',kwargs,'orange')
    ls = set_default_by_kwarg('ls',kwargs,'-')
    ax.plot(contour[:,1],contour[:,0],lw=lw,ls=ls,color=color, **kwargs)
    return ax