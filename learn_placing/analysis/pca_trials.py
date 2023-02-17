import os
import torch
import numpy as np
import learn_placing.common.transformations as tf

from PIL import Image
from learn_placing.common.data import load_dataset
from learn_placing.common.myrmex_processing import preprocess_myrmex, mm2img, upscale_repeat

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def merge_mm_images(mm, noise_tresh=0.05):
    merged = (mm[0]+np.flip(mm[1], 1))/2
    merged = np.where(merged>noise_tresh, merged, 0)
    return merged

def double_PCA(mm):
    """
    mm: (2x16x16) myrmex array, first dimension holds (left, right)
    """

    meansl, evll, evecl, evthl = get_PCA(mm[0,:])
    meansr, evlr, evecr, evthr = get_PCA(mm[1,:])

    # fusing both PCA angle estimates: weigh angles by their eigenvalues
    # TODO should we take sqrt(ev)?
    # weights = np.array([np.sqrt(evll[0]), np.sqrt(evlr[0])])
    # weights /= np.sum(weights)
    # weighted_th = (weights[0]*evthl+weights[1]*evthr)/2
    mean_th = (evthl+evthr)/2

    return np.stack([meansl, meansr]), \
        np.stack([evll, evlr]), \
        np.stack([evecl, evecr]), \
        np.array([mean_th[0], 0])

def get_line_y(x, theta=np.pi/4, b=0):
    return np.tan(theta) * x + b

def get_PC_point(means, evl, evec, pci):
    return means-2*np.sqrt(evl[pci])*evec[:,pci]

def get_line_angle(p1, p2):
    if p1[0]>p2[0]: p1, p2 = p2, p1
    th = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
    return th if th>0 else np.pi+th

def plot_line(ax, theta, point=None, **kw):
    # axes in mpl are flipped, hence the PI offset

    # if no point (in data coordinates) is given, we just use the axis' center
    if point is None:
        point = [.5,.5]
        kw |= {"transform", ax.transAxes}

    ax.axline(point, slope=np.tan(np.pi-theta), **kw)

def plot_PCs(ax, means, evl, evec, scale=1):
    """ 
    scale: upscaling factor for imshow
    """
    means = means.copy()
    means *= scale

    ax.text(*means,"X",color="cyan")
    ax.annotate("",
                fontsize=20,
                xytext = means,
                xy     = get_PC_point(means, evl, evec, 0),
                arrowprops = {"arrowstyle":"<->",
                              "color":"magenta",
                              "linewidth":2},
                # label="PC1"
                )
    ax.annotate("",
                fontsize=20,
                xytext = means,
                xy     = get_PC_point(means, evl, evec, 1),
                arrowprops = {"arrowstyle":"<->",
                              "color":"green",
                              "linewidth":2}
                )

def extract_sample(s, frame_no=10):
    """
    given a sample sequence, extract 
    mm: [left, right] myrmex samples (2x16x16)
    gr: T_world_gripper as quaternion (1x4)
    ft: F/T sensor readings (1x6)
    lbl: target rotation as quaternion (1x6)

    TODO return label based on
    """

    mmleft  = preprocess_myrmex(sample["tactile_left"][1])[frame_no,:]  # 16x16 array
    mmright = preprocess_myrmex(sample["tactile_right"][1])[frame_no,:] # 16x16 array
    
    w2o, g2o, w2g = None, None, None
    for ops in sample["opti_state"][1][0]:
        if ops["parent_frame"] == "gripper_left_grasping_frame" and ops["child_frame"] == "pot":
            g2o = ops["rotation"]
        if ops["parent_frame"] == "base_footprint" and ops["child_frame"] == "pot":
            w2o = ops["rotation"]
        if ops["parent_frame"] == "base_footprint" and ops["child_frame"] == "gripper_left_grasping_frame":
            w2g = ops["rotation"]

    return np.array([mmleft, mmright]), w2g, None, g2o

def rotmat_to_theta(pred):
    """ given a NN prediction (3x3 rotation matrix), return the object's angle offset
    """
    return np.pi-np.arccos(np.dot([0,0,-1], pred.dot([0,0,-1])))

def label_to_theta(y):
    if type(y) == list: y=np.array(y)
    if y.shape == (4,4): y = y[:3,:3]
    elif y.shape == (4,) or y.shape == (1,4):
        y = tf.quaternion_matrix(y)[:3,:3]
    elif y.shape == (3,3): pass
    else:
        print(f"ERROR shape mismatch: {y.shape}")
    
    return rotmat_to_theta(y)

def line_similarity(th, lblth):
    """ 
    two (intersecting) lines will always have two different angles descirbing the intersection (of both are 90deg).
    to determine similarity, we take the smaller one.
    two lines are most dissimilar if their angle(s) are 90deg.
    """
    ang_diff = np.abs(th-lblth)
    return ang_diff if ang_diff < np.pi/2 else np.pi-ang_diff

def single_pred_loss(pred, label, f):
    """
    pred    (3x3) network output, rotation matrix
    label   (1x4) label, quaternion
    f       loss function
    """
    p = np.expand_dims(pred, 0)
    l = np.expand_dims(tf.quaternion_matrix(label), 0)
    return f(torch.Tensor(p), torch.Tensor(l))

if __name__ == "__main__":
    """ NOTE interesting samples

    Dataset: placing_data_pkl_cuboid_large
    good: [64, 69]
    50/50: [58]
    hard: [188]

    NN  good: 100
    PCA good: 64
    PCA bad : 188
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
    
    # load sample 
    frame_no  = 10
    sample_no = 64
    
    sample = ds[sample_no][1]
    mm, w2g, ft, lbl = extract_sample(sample)
    lblth = label_to_theta(lbl)

    # load neural net
    trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/ias_training_new_ds/Combined3D/Combined3D_Neps40_static_tactile_2022.09.13_10-41-43"
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/ias_training_new_ds/Combined3D/Combined3D_Neps40_static_tactile_gripper_2022.09.13_10-42-03"
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/2023.02.13_18-45-21/CombinedAll/CombinedAll_Neps40_static_tactile_2023.02.13_18-45-21"
    trial_weights = f"{trial_path}/weights/best.pth"
    
    params = load_train_params(trial_path)
    model = TactilePlacingNet(**params.netp)
    criterion = rep2loss(params.loss_type)
    checkp = torch.load(trial_weights)
    model.load_state_dict(checkp)
    model.eval()

    pred = model(*[torch.Tensor(np.array(x)) for x in [np.expand_dims(mm, 0), w2g, 0]])
    pred = np.squeeze(pred.detach().numpy())

    nnth  = rotmat_to_theta(pred)
    nnerr = line_similarity(nnth, lblth)

    # perform PCA
    # means, evl, evec, evth = predict_PCA(mm) # PCA on separate sensor images

    mmm = merge_mm_images(mm, noise_tresh=0.15)
    means, evl, evec, evth = get_PCA(mmm)
    evth = evth[0]
    pcaerr = line_similarity(evth, lblth)

    print()
    print(f"PCA err {pcaerr:.4f} | NN  err {nnerr:.4f}")
    print(f"PCA th  {evth:.4f} | NN  th {nnth:.4f} | LBL th {lblth:.4f}")

    scale = 10
    mmimg = upscale_repeat(mmm, factor=scale)
    # mmimg = mm2img(mmimg)

    fig, axes = plt.subplots(ncols=1, figsize=0.8*np.array([10,9]))

    plot_PCs(axes, means, evl, evec, scale=scale)
    im = axes.imshow(mmimg)

    # plot lines at means. NOTE means are estimates, lines will be slightly off!
    plot_line(axes, lblth, point=scale*means, label="target", c="green", lw=2)
    plot_line(axes, evth, point=scale*means, label=f"mean theta {pcaerr:.3f}", c="blue", lw=2)
    plot_line(axes, nnth, point=scale*means, label=f"NN {nnerr:.3f}", c="red", lw=2)

    axes.set_title("Merged sensor image - filtered PCA")

    axes.legend(loc="lower right")

    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

    fig.tight_layout()
    plt.savefig(f"{os.environ['HOME']}/pca_good.png")
    plt.show()