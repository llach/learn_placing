import numpy as np

def plot_line(ax, theta, point=None, **kw):
    # axes in mpl are flipped, hence the PI offset

    # if no point (in data coordinates) is given, we just use the axis' center
    if point is None:
        point = [.5,.5]
        kw |= {"transform", ax.transAxes}

    ax.axline(point, slope=np.tan(np.pi-theta), **kw)