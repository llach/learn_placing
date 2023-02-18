import os
import numpy as np

from learn_placing.common import load_dataset, upscale_repeat, plot_line, extract_sample, label_to_theta, merge_mm_samples, get_mean_force_xy

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from learn_placing.estimators import PCABaseline, NetEstimator
from learn_placing.estimators.hough_baseline import HoughEstimator


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
    
    """ parameters
    """
    frame_no  = 10
    sample_no = 64
    noise_thresh = 0.15
    scale = 10
    
    # select and unpack sample
    sample = ds[sample_no][1]
    mm, w2g, ft, lbl = extract_sample(sample)
    lblth = label_to_theta(lbl)

    # load neural net
    trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/ias_training_new_ds/Combined3D/Combined3D_Neps40_static_tactile_2022.09.13_10-41-43"
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/ias_training_new_ds/Combined3D/Combined3D_Neps40_static_tactile_gripper_2022.09.13_10-42-03"
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/2023.02.13_18-45-21/CombinedAll/CombinedAll_Neps40_static_tactile_2023.02.13_18-45-21"
    nn = NetEstimator(trial_path)

    # create other estimators
    pca = PCABaseline(noise_thresh=noise_thresh)
    hough = HoughEstimator(noise_thresh=0.15, preproc="binary")

    (R_nn, nnth), (nnRerr, nnerr) = nn.estimate_transform(mm, lbl, Qwg=w2g)
    (_, pcath), (_, pcaerr) = pca.estimate_transform(mm, lbl)
    (_, houth), (_, houerr) = hough.estimate_transform(mm, lbl)

    print()
    print(f"PCA err {pcaerr:.4f} | NN  err {nnerr:.4f} | HOU err {houerr:.4f}")
    print(f"PCA th  {pcath:.4f} | NN  th {nnth:.4f}  | HOU th {houth:.4f} | LBL th {lblth:.4f}")
    
    mmm = merge_mm_samples(mm, noise_tresh=noise_thresh)
    mmimg = upscale_repeat(mmm, factor=scale)

    fig, axes = plt.subplots(ncols=1, figsize=0.8*np.array([10,9]))

    pca.plot_PCs(axes, mm, scale=scale)
    means = scale*get_mean_force_xy(mmm)
    im = axes.imshow(mmimg)

    # plot lines at means. NOTE means are estimates, lines will be slightly off!
    plot_line(axes, lblth, point=means, label="target", c="green", lw=2)
    plot_line(axes, nnth, point=means, label=f"NN  {nnerr:.3f}", c="red", lw=2)
    plot_line(axes, pcath, point=means, label=f"PCA {pcaerr:.3f}", c="blue", lw=2)
    plot_line(axes, houth, point=means, label=f"HOU {houerr:.3f}", c="white", lw=2)
    
    axes.set_title("Merged sensor image - filtered PCA")

    axes.legend(loc="lower right")

    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

    fig.tight_layout()
    plt.savefig(f"{os.environ['HOME']}/pca_good.png")
    plt.show()