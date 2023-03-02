import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from learn_placing.common import upscale_repeat, merge_mm_samples, get_mean_force_xy

def cr_plot_setup(fsize=53):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = f"{fsize}"
    plt.rcParams["font.weight"] = "500"

def plot_line(ax, theta, point=None, **kw):
    # axes in mpl are flipped, hence the PI offset

    # if no point (in data coordinates) is given, we just use the axis' center
    if point is None:
        point = [.5,.5]
        kw |= {"transform", ax.transAxes}

    ax.axline(point, slope=np.tan(np.pi-theta), **kw)

def models_theta_plot(mm_imgs, noise_thresh, lines, ax, fig, scale=1, lloc="upper right"):
    mmm = merge_mm_samples(mm_imgs, noise_tresh=noise_thresh)
    mmimg = upscale_repeat(mmm, factor=scale)

    means = scale*get_mean_force_xy(mmm)
    im = ax.imshow(mmimg, cmap="magma")

    # plot lines at means. NOTE means are estimates, lines will be slightly off!
    for (th, label, color) in lines: plot_line(ax, th, point=means, label=label, c=color, lw=9)
    
    ax.legend(loc=lloc)

    divider = make_axes_locatable(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_ticks([])
    # ax.set_ticklabels([])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)