import os
import numpy as np
import matplotlib.pyplot as plt

from learn_placing.common.data import load_dataset_file
from learn_placing.training.utils import DatasetName, ds2name

def bias_correct(arr):
    for i in range(arr.shape[0]):
        arr[i] = arr[i]-arr[i,0]

dsnames = [DatasetName.cuboid, DatasetName.cylinder, DatasetName.object_var, DatasetName.gripper_var]
for dd in dsnames:
    name = ds2name[dd]

    dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{name}.pkl"
    ds = load_dataset_file(dataset_file_path)

    ft = np.mean(list(ds["ft"].values()), axis=-1)
    sft = np.mean(list(ds["static_ft"].values()), axis=-1)

    myr = np.swapaxes(list(ds["inputs"].values()), 1, 2)
    smyr = np.swapaxes(list(ds["static_inputs"].values()), 1, 2)

    myr = np.mean(myr, axis=(2,3,4))
    smyr = np.mean(smyr, axis=(2,3,4))

    bias_correct(ft)
    bias_correct(sft)
    bias_correct(myr)
    bias_correct(smyr)

    mmin, mmax = np.min(myr), np.max(myr)
    fmin, fmax = np.min(ft), np.max(ft)

    fig, axs = plt.subplots(2, 2, figsize=(9.71, 8.61))

    for s in sft: axs[0,0].plot(range(len(s)),s)
    for s in ft: axs[0,1].plot(range(len(s)),s)
    for ax in axs[0,:]: ax.set_ylim(1.1*fmin, 1.1*fmax)

    for m in smyr: axs[1,0].plot(range(len(m)),m)
    for m in myr: axs[1,1].plot(range(len(m)),m)
    for ax in axs[1,:]: ax.set_ylim(1.1*mmin, 1.1*mmax)

    axs[0,0].set_title("FT static")
    axs[0,1].set_title("FT dynamic")

    axs[1,0].set_title("Tactile static")
    axs[1,1].set_title("Tactile dynamic")

    fig.suptitle(f"Input Data after Pre-Processing (Dataset: {dd})")
    fig.tight_layout()
    plt.show()