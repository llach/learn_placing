import os
import numpy as np

from PIL import Image
from learn_placing.common.data import load_dataset
from learn_placing.common.myrmex_processing import preprocess_myrmex, mm2img, upscale_repeat

import matplotlib.pyplot as plt

def marginal_mean(arr, axis):
    '''
    E(x) = int x f(x) dx
    '''
    mardens = arr.sum(axis=axis)
    mardens /= np.sum(mardens) ## rescale so that marginal density adds up to 1
    
    meanx   = np.sum((np.arange(len(mardens))*mardens))
    return meanx

def marginal_sd(arr, axis):
    """ E( (x - barx)**2 ) = int (x-barx)**2 f(x) dx
    """
    mardens = arr.sum(axis=axis)
    
    meanx   = marginal_mean(arr, axis)
    varx    = np.sqrt(np.sum((np.arange(len(mardens)) - meanx)**2*mardens))
    return varx

def get_cov(arr):
    """ E( (x - barx)*(y - bary) ) = int (x-barx)*(y - bary) f(x,y) dxdy
    """
    
    x_coor_mat = np.zeros_like(arr)
    for icol in range(x_coor_mat.shape[1]):
        x_coor_mat[:, icol] = icol
        
    y_coor_mat = np.zeros_like(arr)
    for irow in range(y_coor_mat.shape[0]):
        y_coor_mat[irow, :] = irow
        
    meanx   = marginal_mean(arr, axis=0)
    meany   = marginal_mean(arr, axis=1) 
    
    cov     = np.sum((x_coor_mat - meanx)*(y_coor_mat - meany )*arr)
    return cov

def get_PCA(sample):
    """
    """
    meanx, meany = marginal_mean(sample, axis=0), marginal_mean(s, axis=1)
    sdx, sdy = marginal_sd(sample, axis=0), marginal_sd(s, axis=1)
    cov = get_cov(sample)
    C = np.array([[sdx**2, cov],
                  [cov,    sdy**2]])

    evl, evec = np.linalg.eig(C)
    eigsort = np.argsort(evl)[::-1]
    evl, evec = evl[eigsort], evec[:,eigsort]

    np.array([meanx, meany]), evl, evec

def plot_PCA(ax, means, evl, evec, scale=1):
    """ 
    scale: upscaling factor for imshow
    """
    means *= scale
    meanx, meany = means
    evllen = np.sqrt(evl)

    ax.text(meanx,meany,"X",color="cyan")
    ax.annotate("",
                fontsize=20,
                xytext = (meanx,meany),
                xy     = ((meanx-2*evllen[0]*evec[0,0]),
                          (meany-2*evllen[0]*evec[1,0])),
                arrowprops = {"arrowstyle":"<->",
                              "color":"magenta",
                              "linewidth":2}
                )
    ax.annotate("",
                fontsize=20,
                xytext = (meanx,meany),
                xy     = ((meanx-2*evllen[1]*evec[0,1]),
                          (meany-2*evllen[1]*evec[1,1])),
                arrowprops = {"arrowstyle":"<->",
                              "color":"green",
                              "linewidth":2}
                )


if __name__ == "__main__":
    """ NOTE interesting samples

    Dataset: placing_data_pkl_cuboid_large
    good: [64, 69]
    50/50: [58]
    hard: [188]
    """
    
    dsname = "placing_data_pkl_cuboid_large"
    dataset_path = f"{os.environ['HOME']}/tud_datasets/{dsname}"

    # sample timestamp -> sample
    ds = load_dataset(dataset_path)
    samples = list(ds.items())

    sample = samples[64][1]["tactile_left"][1]
    s = preprocess_myrmex(sample)[10,:] # Nx16x16 array 

    simg = upscale_repeat(s, factor=10)
    simg = mm2img(s)

    plt.imshow(simg)

    plt.show()