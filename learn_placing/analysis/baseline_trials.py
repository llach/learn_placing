import os
import numpy as np

from learn_placing.common import load_dataset, extract_sample, label_to_theta

import matplotlib.pyplot as plt
from learn_placing.common.viz_tools import models_theta_plot
from learn_placing.estimators import PCABaseline, NetEstimator, HoughEstimator

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

    fig, axes = plt.subplots(ncols=1, figsize=0.8*np.array([10,9]))

    pca.plot_PCs(axes, mm, scale=scale)
    models_theta_plot(
        mm_imgs=mm,
        noise_thresh=noise_thresh,
        ax=axes,
        fig=fig,
        scale=scale,
        lines = [
            [lblth, "target", "green"],
            [nnth,  f"NN  {nnerr:.3f}", "red"],
            [pcath, f"PCA {pcaerr:.3f}", "blue"],
            [houth, f"HOU {houerr:.3f}", "white"],
        ]
    )

    axes.set_title("NN Baseline Comparison")
    fig.tight_layout()
    plt.savefig(f"{os.environ['HOME']}/pca_good.png")
    plt.show()