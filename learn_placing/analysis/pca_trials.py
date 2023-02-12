import os
import torch
import numpy as np
import learn_placing.common.transformations as tf

from PIL import Image
from learn_placing.common.data import load_dataset
from learn_placing.common.myrmex_processing import preprocess_myrmex, mm2img, upscale_repeat

import matplotlib.pyplot as plt
from learn_placing.training.tactile_insertion_rl import TactilePlacingNet

from learn_placing.training.utils import load_train_params, rep2loss

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
    meanx, meany = marginal_mean(sample, axis=0), marginal_mean(sample, axis=1)
    sdx, sdy = marginal_sd(sample, axis=0), marginal_sd(sample, axis=1)
    cov = get_cov(sample)
    C = np.array([[sdx**2, cov],
                  [cov,    sdy**2]])

    evl, evec = np.linalg.eig(C)
    eigsort = np.argsort(evl)[::-1]
    evl, evec = evl[eigsort], evec[:,eigsort]

    # slope of eigenvectors as angle theta
    # axes in mpl are flipped, hence the PI offset
    means = [meanx, meany]
    evth = np.array([
        np.pi-get_line_angle(means, get_PC_point(means, evl, evec, 0)),
        np.pi-get_line_angle(means, get_PC_point(means, evl, evec, 1)),
    ])

    return np.array([meanx, meany]), evl, evec, evth

def get_line_y(x, theta=np.pi/4, b=0):
    return np.tan(theta) * x + b

def get_PC_point(means, evl, evec, pci):
    return means-2*np.sqrt(evl[pci])*evec[:,pci]

def get_line_angle(p1, p2):
    if p1[0]>p2[0]: p1, p2 = p2, p1
    th = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
    return th if th>0 else np.pi+th

def plot_line(ax, theta, **kw):
    # axes in mpl are flipped, hence the PI offset
    ax.axline([.5,.5], slope=np.tan(np.pi-theta), transform=ax.transAxes, **kw)

def plot_PCA(ax, means, evl, evec, scale=1):
    """ 
    scale: upscaling factor for imshow
    """
    means *= scale

    ax.text(*means,"X",color="cyan")
    ax.annotate("",
                fontsize=20,
                xytext = means,
                xy     = get_PC_point(means, evl, evec, 0),
                arrowprops = {"arrowstyle":"<->",
                              "color":"magenta",
                              "linewidth":2}
                )
    ax.annotate("",
                fontsize=20,
                xytext = means,
                xy     = get_PC_point(means, evl, evec, 1),
                arrowprops = {"arrowstyle":"<->",
                              "color":"green",
                              "linewidth":2}
                )

def try_sample(model, sample, frame_no):
    """
    x.shape
    torch.Size([8, 2, 50, 16, 16])
    -> [batch, sensors, sequence, H, W]

    gr.shape
    torch.Size([8, 4])
    -> [batch, Q]

    ft.shape
    torch.Size([8, 15, 6])
    -> [batch, sequence, FT]
    """


    if params.input_data == InData.static:
        print(params.input_data)
        xs = [[tinp_static], [Qwg], [ftinp_static]]
    elif params.input_data == InData.with_tap:
        print(params.input_data)
        xs = [[tinp], [Qwg], [ftinp]]
    prediction = model(*[torch.Tensor(np.array(x)) for x in xs])
    prediction = np.squeeze(prediction.detach().numpy())



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
    ds = list(ds.items())

    """
    dataset:
    timestamp - sample (i.e. dict of time series)
        |-> tactile_left
            |-> [timestamps]
            |-> [myrmex samples]
    """
    
    frame_no = 10
    sample_no = 64
    
    sample = ds[sample_no][1]
    mm = preprocess_myrmex(sample["tactile_right"][1])[frame_no,:] # Nx16x16 array 
    for ops in sample["opti_state"][1][0]:
        if ops["parent_frame"] == "gripper_left_grasping_frame" and ops["child_frame"] == "pot":
            lbl = tf.euler_from_quaternion(ops["rotation"])[1]

    means, evl, evec, evth = get_PCA(mm)
    print(evth, lbl)

    trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/2022.09.08_19-11-16/test_obj/test_obj_Neps20_static_tactile_gripper_ft_2022.09.08_19-12-35"
    trial_weights = f"{trial_path}/weights/final.pth"
    
    params = load_train_params(trial_path)
    model = TactilePlacingNet(**params.netp)
    criterion = rep2loss(params.loss_type)

    scale = 10
    mmimg = upscale_repeat(mm, factor=scale)
    mmimg = mm2img(mmimg)

    fig, ax = plt.subplots()
    
    ax.imshow(mmimg)
    plot_PCA(ax, means, evl, evec, scale=scale)
    plot_line(ax, np.pi-evth[0], label="PC1")
    plot_line(ax, np.pi-lbl, label="target", c="green", lw=2)

    ax.legend()

    plt.show()