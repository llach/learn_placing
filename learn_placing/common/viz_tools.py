import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from learn_placing.common import upscale_repeat, merge_mm_samples, get_mean_force_xy

def plot_line(ax, theta, point=None, **kw):
    # axes in mpl are flipped, hence the PI offset

    # if no point (in data coordinates) is given, we just use the axis' center
    if point is None:
        point = [.5,.5]
        kw |= {"transform", ax.transAxes}

    ax.axline(point, slope=np.tan(np.pi-theta), **kw)

def models_theta_plot(mm_imgs, noise_thresh, lines, ax, fig, scale=1):
    mmm = merge_mm_samples(mm_imgs, noise_tresh=noise_thresh)
    mmimg = upscale_repeat(mm_imgs[0], factor=scale)

    means = scale*get_mean_force_xy(mmm)
    im = ax.imshow(mmimg)

    # plot lines at means. NOTE means are estimates, lines will be slightly off!
    for (th, label, color) in lines: plot_line(ax, th, point=means, label=label, c=color, lw=2)
    
    ax.legend(loc="lower right")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)