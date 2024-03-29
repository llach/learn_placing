import os
import torch
import numpy as np
from learn_placing.common.tools import to_numpy

from learn_placing.estimators import PCABaseline, NetEstimator, HoughEstimator
from learn_placing.training.utils import get_dataset, RotRepr, InRot

if __name__ == "__main__":
    noise_thresh = 0.15

     # load neural net
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/2023.02.13_18-45-21/CombinedAll/CombinedAll_Neps40_static_tactile_2023.02.13_18-45-21"
    trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/2023.02.24_10-41-09/UPC_v1/UPC_v1_Neps60_static_tactile_2023.02.24_10-41-09"
    nn = NetEstimator(trial_path)

    # create other estimators
    pca = PCABaseline(noise_thresh=noise_thresh)
    hough = HoughEstimator(noise_thresh=0.15, preproc="canny")

    nnerrs, pcaerrs, houerrs = [], [], []
    train_l, test_l, _ = get_dataset("upc_cuboid", nn.params, target_type=InRot.g2o, out_repr=RotRepr.ortho6d, seed=nn.params.dataset_seed)
    with torch.no_grad():
        for batch in test_l:
            for data in zip(*batch):
                mm, Qwg, ft, lbl = data
                # mm = mm[:,10,:,:] # we always take frame number 10

                (R_nn, nnth), (nnRerr, nnerr) = nn.estimate_transform(mm, lbl, ft=ft, Qwg=Qwg)
                (_, pcath), (_, pcaerr) = pca.estimate_transform(*to_numpy(mm, lbl))
                (_, houth), (_, houerr) = hough.estimate_transform(*to_numpy(mm, lbl))

                nnerrs.append(nnerr)
                pcaerrs.append(pcaerr)
                houerrs.append(houerr)

    print(f"NN  {np.mean(nnerrs):.2f}±{np.var(nnerrs):.2f}")
    print(f"HOU {np.nanmean(houerrs):.2f}±{np.nanvar(houerrs):.2f} (not counting NaNs)")
    print(f"PCA {np.mean(pcaerrs):.2f}±{np.var(pcaerrs):.2f}")