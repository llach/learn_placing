import os
import time
import torch
import numpy as np
from learn_placing.common.tools import to_numpy

from learn_placing.estimators import PCABaseline, NetEstimator, HoughEstimator
from learn_placing.training.utils import get_dataset, RotRepr, InRot

if __name__ == "__main__":
    noise_thresh = 0.15

    # load neural net
    trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/2023.02.24_10-41-09/UPC_v1/UPC_v1_Neps60_static_tactile_2023.02.24_10-41-09"
    nn = NetEstimator(trial_path)

    # create other estimators
    pca = PCABaseline(noise_thresh=noise_thresh)
    hough = HoughEstimator(noise_thresh=noise_thresh, preproc="binary")

    nn_times, pca_times, hou_times = [], [], []
    train_l, _, _ = get_dataset("upc_cuboid", nn.params, target_type=InRot.g2o, out_repr=RotRepr.ortho6d, seed=nn.params.dataset_seed, train_ratio=1.0)

    with torch.no_grad():
        for batch in train_l:
            for data in zip(*batch):
                mm, Qwg, ft, lbl = data

                t = time.time()
                (R_nn, nnth), (nnRerr, nnerr) = nn.estimate_transform(mm, lbl, ft=ft, Qwg=Qwg)
                nn_times.append((time.time()-t)*1e3)

                t = time.time()
                (_, pcath), (_, pcaerr) = pca.estimate_transform(*to_numpy(mm, lbl))
                pca_times.append((time.time()-t)*1e3)

                t = time.time()
                (_, houth), (_, houerr) = hough.estimate_transform(*to_numpy(mm, lbl))
                hou_times.append((time.time()-t)*1e3)

    print(f"NN  {np.mean(nn_times):.2f}ms ± {np.var(nn_times):.3f}")
    print(f"HOU {np.nanmean(pca_times):.2f}ms ± {np.nanvar(pca_times):.3f}")
    print(f"PCA {np.mean(hou_times):.2f}ms ± {np.var(hou_times):.3f}")