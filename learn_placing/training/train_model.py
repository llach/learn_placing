import os
import json
import torch
import pickle

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from datetime import datetime
from typing import List
from utils import LossType, get_dataset, InData, RotRepr, InRot, DatasetName, test_net, AttrDict
from tactile_insertion_rl import TactilePlacingNet, ConvProc

from learn_placing import now, training_path
from learn_placing.training.train_utils import rep2loss

def plot_learning_curve(train_loss, test_loss, a, ax, min_test=None, min_test_i=0, small_title=False):
    xs = np.arange(len(test_loss)).astype(int)+1

    tl_mean = np.mean(test_loss, axis=1)
    tl_upper = np.percentile(test_loss, 0.95, axis=1)
    tl_lower = np.percentile(test_loss, 0.05, axis=1)

    ax.plot(xs, train_loss, label=f"training loss | {train_loss[min_test_i]:.5f}")
    ax.plot(xs, tl_mean, label=f"test loss | {tl_mean[min_test_i]:.5f}")

    ax.fill_between(xs, tl_mean+tl_upper, tl_mean-tl_lower, color="#A9A9A9", alpha=0.3, label="test loss 95%ile")

    if a.loss_type == LossType.pointarccos or a.loss_type == LossType.geodesic:
        ax.set_ylim([0.0,np.pi])
        ax.set_ylabel("loss [radian]")
    elif a.loss_type == LossType.msesum or a.loss_type == LossType.quaternion or a.loss_type == LossType.pointcos:
        ax.set_ylim([-0.05,1.0])
        ax.set_ylabel("loss")

    ax.scatter(min_test_i, min_test, c="green", marker="X", linewidths=0.7, label="best avg. test loss")
    ax.set_xlabel("#batches")

    if not small_title:
        ax.set_title(f"dsname={a.dsname} input={a.input_data} tactile={a.with_tactile} gripper={a.with_gripper} ft={a.with_ft}")
    else:
        ax.set_title(f"input={a.input_data};\ntactile={a.with_tactile} gripper={a.with_gripper} ft={a.with_ft};")

    ax.legend()

def train(
    dataset: DatasetName,
    input_type: InData,
    input_modalities: List[bool],
    target_type: InRot,
    trial_path: str,
    ## TODO add parameters as input parameters
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
        input_data = input_type,
        with_tactile = with_tactile,
        with_gripper = with_gripper,
        with_ft = with_ft,

        augment = augment,
        aug_n = aug_n,
        batch_size = 48,
        loss_type = LossType.pointarccos,
        out_repr = RotRepr.ortho6d,
        target_type = target_type,
        validate = False,
        store_training = True,
        gripper_repr = RotRepr.quat,
        start_time = now(),
        save_freq = 0.1
    )
    a.__setattr__("netp", AttrDict(
        preproc_type = ConvProc.ONEFRAMESINGLETRL, ### NOTE set 3DConv here
        ## TODO add bool here for post-network trafo multiplication
        output_type = a.out_repr,
        with_tactile = a.with_tactile,
        with_gripper = a.with_gripper,
        with_ft = a.with_ft,
        kernel_sizes = [(3,3), (3,3)],
        cnn_out_channels = [16, 32],
        conv_stride = (2,2),
        conv_padding = (0,0),
        conv_output = 128,
        rnn_neurons = 128,
        rnn_layers = 1,
        ft_rnn_neurons = 16,
        ft_rnn_layers = 1,
        fc_neurons = [64, 32],
    ))

    # a.__setattr__("netp", AttrDict(
    #     preproc_type = ConvProc.TDCONV, ### NOTE set 3DConv here
    #     ## TODO add bool here for post-network trafo multiplication
    #     output_type = a.out_repr,
    #     with_tactile = a.with_tactile,
    #     with_gripper = a.with_gripper,
    #     with_ft = a.with_ft,
    #     input_dim=[40, 16, 16],
    #     kernel_sizes = [(10,3,3), (4,3,3)],
    #     cnn_out_channels = [16, 32],
    #     conv_stride = (4,2,2),
    #     conv_padding = (0,0,0),
    #     conv_output = 1, # NOTE: the final output is this * cnn_out_channels
    #     rnn_neurons = 128,
    #     rnn_layers = 1,
    #     ft_rnn_neurons = 16,
    #     ft_rnn_layers = 1,
    #     fc_neurons = [64, 32],
    # ))
    a.__setattr__("adamp", AttrDict(
        lr=1e-3,
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=0, 
        amsgrad=False
    ))

    trial_name = f"{a.dsname}_Neps{a.N_episodes}_{a.input_data}"
    if a.with_tactile: trial_name += "_tactile"
    if a.with_gripper: trial_name += "_gripper"
    if a.with_ft: trial_name += "_ft"
    trial_name += f"_{a.start_time}"

    trial_path = f"{trial_path}/{trial_name}/"

    train_l, test_l, seed = get_dataset(a.dsname, a)
    a.__setattr__("dataset_seed", seed)

    model = TactilePlacingNet(**a.netp)
    optimizer = optim.Adam(model.parameters(), **a.adamp)

    criterion = rep2loss(a.loss_type)

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
    t_path = f"{training_path}/../batch_trainings"
    base_path = f"{t_path}/{now()}"

    Neps=40
    datasets = [DatasetName.combined_all, DatasetName.combined_3d]
    # datasets = [DatasetName.combined_large]
    target_type = InRot.g2o
    aug_n = 1

    # full training
    input_types = [InData.static]
    input_modalities = [
        [True , False, False],
        # [False, True , False],
        # [False, False, True],
        [True , True , False],
        # [True , False, True],
        # [False, True , True],
        [True , True , True],
    ]
    # augment = [
    #     [True, True],
    #     [True, False],
    #     [False, True]
    # ]

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
                    input_type=input_types[0],
                    input_modalities=input_mod,
                    target_type=target_type,
                    trial_path=dspath,
                    ## TODO add parameters for 3dconv preproc & trafo multiplication
                    augment=au,
                    aug_n = aug_n,
                    Neps=Neps,
                    other_ax=oax
                )
                trial_times.append(datetime.now() - trial_start)
                print(f"trial took {trial_times[-1]}")

        fig.suptitle(f"Dataset '{dataset}' - [{target_type}]")
        fig.tight_layout()
        fig.savefig(f"{dspath}/trainings_{dataset.lower()}_{Neps}_{target_type}.png")
        
    print(f"trial times")
    for trt in trial_times: print(trt)

    train_end = datetime.now()-train_start
    print(f"training took {train_end}")