import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from learn_placing import dataset_path
from learn_placing.common import models_theta_plot, line_angle_from_rotation

save_path = f"{dataset_path}/upc_cylinder/"
for fi in os.listdir(save_path):
    if fi == "pics": continue
    sample_path = f"{save_path}{fi}"

    with open(sample_path, "rb") as f:
        data = pickle.load(f)

    mm = data["mm"]
    lblth = line_angle_from_rotation(data["Qgo"])

    scale=100
    fig, ax = plt.subplots(ncols=1, figsize=0.8*np.array([10,9]))

    lines = [
        [lblth, f"OptiTrack (lblth)", "green"],
    ]
    models_theta_plot(
        mm_imgs=mm,
        noise_thresh=0.001,
        ax=ax,
        fig=fig,
        scale=scale,
        lines=lines
    )

    ax.set_title(fi)
    fig.tight_layout()
    plt.show()