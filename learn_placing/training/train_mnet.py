import os
import json
import torch
import pickle

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from datetime import datetime
from typing import List
from learn_placing.training.utils import line_similarity_th
from utils import get_dataset, DatasetName, test_net, AttrDict
from mnet import MyrmexNet

from learn_placing import now, training_path

def plot_learning_curve(train_loss, test_loss, a, ax, min_test=None, min_test_i=0, small_title=False):
    xs = np.arange(len(test_loss)).astype(int)+1

    tl_mean = np.squeeze(np.mean(test_loss, axis=1))
    tl_upper = np.squeeze(np.percentile(test_loss, 0.95, axis=1))
    tl_lower = np.squeeze(np.percentile(test_loss, 0.05, axis=1))

    ax.plot(xs, train_loss, label=f"training loss | {train_loss[min_test_i]:.5f}")
    ax.plot(xs, tl_mean, label=f"test loss | {tl_mean[min_test_i]:.5f}")

    ax.fill_between(xs, tl_mean+tl_upper, tl_mean-tl_lower, color="#A9A9A9", alpha=0.3, label="test loss 95%ile")

    ax.set_ylim([0.0,np.pi/2])
    ax.set_ylabel("loss [radian]")

    ax.scatter(min_test_i, min_test, c="green", marker="X", linewidths=0.7, label="best avg. test loss")
    ax.set_xlabel("#batches")

    if not small_title:
        ax.set_title(f"dsname={a.dsname}\ntactile={a.with_tactile} gripper={a.with_gripper} ft={a.with_ft}")
    else:
        ax.set_title(f"tactile={a.with_tactile} gripper={a.with_gripper} ft={a.with_ft}")

    ax.legend()

def train(
    dataset: DatasetName,
    input_modalities: List[bool],
    trial_path: str,
    augment: list = None,
    aug_n: int = 1,
    Neps: int = 10,
    other_ax = None
):
    """ PARAMETERS
    """
    with_tactile, with_gripper, with_ft = input_modalities

    a = AttrDict(
        N_episodes = Neps,
        dsname = dataset,

        with_tactile = with_tactile,
        with_gripper = with_gripper,
        with_ft = with_ft,

        augment = augment,
        aug_n = aug_n,
        batch_size = 48,
        validate = False,
        store_training = True,
        start_time = now(),
        save_freq = 0.1
    )
    a.__setattr__("netp", AttrDict(
        with_tactile = a.with_tactile,
        with_gripper = a.with_gripper,
        with_ft = a.with_ft,
        kernel_sizes = [(3,3), (3,3)],
        cnn_out_channels = [16, 32],
        conv_stride = (2,2),
        conv_padding = (0,0),
        conv_output = 128,
    ))
    a.__setattr__("adamp", AttrDict(
        lr=1e-3,
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=0, 
        amsgrad=False
    ))

    trial_name = f"{a.dsname}_Neps{a.N_episodes}"
    if a.with_tactile: trial_name += "_tactile"
    if a.with_gripper: trial_name += "_gripper"
    if a.with_ft: trial_name += "_ft"
    trial_name += f"_{a.start_time}"

    trial_path = f"{trial_path}/{trial_name}/"

    train_l, test_l, seed = get_dataset(a.dsname, a)
    a.__setattr__("dataset_seed", seed)

    model = MyrmexNet(**a.netp)
    optimizer = optim.Adam(model.parameters(), **a.adamp)
    criterion = line_similarity_th

    train_losses = []
    test_losses = []
    val_losses = []

    N_test_avg = 10
    min_test_loss = np.inf
    min_test_loss_i = 0

    os.makedirs(trial_path, exist_ok=True)
    os.makedirs(f"{trial_path}/weights", exist_ok=True)
    with open(f"{trial_path}parameters.json", "w") as f:
        json.dump(a, f, indent=2)

    print(f"starting training for {trial_name}")

    # code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    nbatch = 0
    save_batches = int(a.N_episodes*len(train_l)*a.save_freq)
    for epoch in range(a.N_episodes):  # loop over the dataset multiple times
        for i, data in enumerate(train_l, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, grip, ft, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, grip, ft)
            loss = torch.mean(criterion(outputs, labels))
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss = loss.item()
            train_losses.append(train_loss)

            test_out, test_lbl, test_loss, _ = test_net(model, criterion, test_l)
            test_losses.append(test_loss)

            if a.validate:
                print("validation not supported atm")

            test_avg = np.mean(test_losses[-N_test_avg:])
            if len(test_losses) >= N_test_avg and test_avg < min_test_loss:
                print(f"new best model with {test_avg:.5f}")
                torch.save(model.state_dict(), f"{trial_path}/weights/best.pth")
                min_test_loss_i = nbatch
                min_test_loss = test_avg

            # store model weights
            if nbatch % save_batches == save_batches-1:
                torch.save(model.state_dict(), f"{trial_path}/weights/batch_{nbatch}.pth")
            nbatch += 1

            print(f"[{epoch + 1}, {i + 1:5d}] loss: {train_loss:.5f} | test loss: {np.mean(test_loss):.5f}", end="")
            print()
    torch.save(model.state_dict(), f"{trial_path}/weights/final.pth")

    with open(f"{trial_path}/losses.pkl", "wb") as f:
        pickle.dump({
            "train": train_losses,
            "test": test_losses,
            "validation": val_losses,
            "min_test": [min_test_loss, min_test_loss_i]
        }, f)

    fig, ax = plt.subplots(figsize=(8.71, 6.61))
    plot_learning_curve(train_losses, test_losses, a, ax, min_test=min_test_loss, min_test_i=min_test_loss_i)
    if other_ax is not None: plot_learning_curve(train_losses, test_losses, a, other_ax, min_test=min_test_loss, min_test_i=min_test_loss_i, small_title=True)

    fig.tight_layout()
    fig.savefig(f"{trial_path}/learning_curve.png")

    plt.clf()
    plt.close()

    print('training done!')
    print(trial_name)
    return trial_name

if __name__ == "__main__":
    t_path = f"{training_path}/../mnet"
    base_path = f"{t_path}/{now()}"

    Neps=100
    datasets = [DatasetName.combined_all, DatasetName.combined_3d]
    # datasets = [DatasetName.combined_large]

    # full training
    input_modalities = [
        [True , False, False],
        # [False, True , False],
        # [False, False, True],
        # [True , True , False],
        # [True , False, True],
        # [False, True , True],
        [True , True , True],
    ]
    augment = [[False, False]]

    trial_times = []
    for dataset in datasets:
        dspath = f"{base_path}/{dataset}"
        os.makedirs(dspath, exist_ok=True)

        nrows=len(input_modalities)
        ncols=len(augment)
        fig, axs = plt.subplots(nrows,ncols,figsize=(4.3*ncols, 3.3*nrows))

        trials = {}
        train_start = datetime.now()
        for i, au in enumerate(augment):
            for j, input_mod in enumerate(input_modalities):
                oax = axs[j,i] if ncols>1 else axs[j]

                trial_start = datetime.now()
                trialname = train(
                    dataset=dataset,
                    input_modalities=input_mod,
                    trial_path=dspath,
                    augment=au,
                    aug_n=0,
                    Neps=Neps,
                    other_ax=oax
                )
                trial_times.append(datetime.now() - trial_start)
                print(f"trial took {trial_times[-1]}")

        fig.suptitle(f"Dataset '{dataset}'")
        fig.tight_layout()
        fig.savefig(f"{dspath}/trainings_{dataset.lower()}_{Neps}.png")
        
    print(f"trial times")
    for trt in trial_times: print(trt)

    train_end = datetime.now()-train_start
    print(f"training took {train_end}")