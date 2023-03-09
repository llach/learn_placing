import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from learn_placing.common.myrmex_processing import merge_mm_samples, upscale_repeat

from learn_placing.common.viz_tools import models_theta_plot, cr_plot_setup
from learn_placing.estimators import PCABaseline, NetEstimator, HoughEstimator
from learn_placing.common import line_angle_from_rotation


if __name__ == "__main__":
    # 66
    noise_thresh = 0.0
    
    basepath = f"{os.environ['HOME']}/tud_datasets/upc_cuboid/"
    for fi in os.listdir(basepath):
        if "pkl" not in fi: continue
        # if fi[:3] != "665": continue
        if fi[:3] != "352": continue

        with open(f"{basepath}{fi}", "rb") as f:
            data = pickle.load(f)

        mm = data["mm"]
        lbl = data["Qgo"]
        w2g = data["Qwg"]
        lblth = line_angle_from_rotation(lbl)

        # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/2023.02.24_10-41-09/UPC_v1/UPC_v1_Neps60_static_tactile_2023.02.24_10-41-09"
        # nn = NetEstimator(trial_path)

        # # create other estimators
        # pca = PCABaseline(noise_thresh=noise_thresh)
        # hough = HoughEstimator(noise_thresh=0.15, preproc="binary")

        # (R_nn, nnth), (nnRerr, nnerr) = nn.estimate_transform(mm, lbl, Qwg=w2g, ft=[])
        # (_, pcath), (pcaRerr, pcaerr) = pca.estimate_transform(mm, lbl)
        # (_, houth), (houRerr, houerr) = hough.estimate_transform(mm, lbl)
        
        cr_plot_setup(fsize=45)
        fig, axes = plt.subplots(ncols=1, figsize=1.8*np.array([10,9]))

        fig = plt.figure(frameon=False)
        fig.set_size_inches(5,5)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        mmm = merge_mm_samples(mm, noise_tresh=noise_thresh)
        mmimg = upscale_repeat(mmm, factor=10)

        ax.imshow(mmimg, aspect='auto', cmap="magma")
        fig.savefig(f"/tmp/{fi.replace('pkl', 'png')}")


        # pca.plot_PCs(axes, mm, scale=scale)
        models_theta_plot(
            mm_imgs=mm,
            noise_thresh=noise_thresh,
            ax=axes,
            fig=fig,
            scale=10,
            lines = [
                [lblth, "Ground-truth", "white"],
                [nnth,  f"Tactile-only ", "red"],
                [pcath, f"PCA ", "#04D9FF"],
                [houth, f"Hough ", "#41F94A"],
            ],
            lloc="upper right"
        )

        # axes.set_title("NN Baseline Comparison")
        # fig.tight_layout()
        # plt.savefig(f"{os.environ['HOME']}/{fi.replace('pkl', 'png')}")
        # plt.savefig(f"{os.environ['HOME']}/seq_heatmap.png")
        # plt.savefig(f"/tmp/{fi.replace('pkl', 'png')}")
        # plt.show()
        plt.close()