import numpy as np
import matplotlib.pyplot as plt
from learn_placing.common.viz_tools import models_theta_plot

from learn_placing.training.utils import AttrDict, DatasetName, get_dataset

train_l, test_l, seed = get_dataset(DatasetName.combined_all, AttrDict(batch_size=10))

for i, data in enumerate(train_l, 0):
    inputs, grip, ft, labels = data

    mm = inputs.numpy()[0]
    lblth = np.squeeze(labels.numpy())[0]

    fig, axes = plt.subplots(ncols=1, figsize=0.8*np.array([10,9]))

    models_theta_plot(
        mm_imgs=mm,
        noise_thresh=0.0,
        ax=axes,
        fig=fig,
        scale=100,
        lines = [
            [lblth, "target", "green"],
        ]
    )

    axes.set_title("NN Baseline Comparison")
    fig.tight_layout()
    plt.show()