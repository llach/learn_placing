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
from learn_placing.training.utils import rep2loss

def plot_learning_curve(train_loss, test_loss, a, ax, small_title=False):
    lastN = int(len(train_loss)*0.05)

    xs = np.arange(len(test_loss)).astype(int)+1
    ax.plot(xs, train_loss, label=f"training loss | {np.mean(train_loss[-lastN:]):.5f}")
    ax.plot(xs, test_loss, label=f"test loss | {np.mean(test_loss[-lastN:]):.5f}")

    if a.loss_type == LossType.pointarccos or a.loss_type == LossType.geodesic:
        ax.set_ylim([0.0,np.pi])
        ax.set_ylabel("loss [radian]")
    elif a.loss_type == LossType.msesum or a.loss_type == LossType.quaternion or a.loss_type == LossType.pointcos:
        ax.set_ylim([-0.05,1.0])
        ax.set_ylabel("loss")

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

        batch_size = 32,
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
        preproc_type = ConvProc.SINGLETRL, ### NOTE set 3DConv here
        ## TODO add bool here for post-network trafo multiplication
        output_type = a.out_repr,
        with_tactile = a.with_tactile,
        with_gripper = a.with_gripper,
        with_ft = a.with_ft,
        kernel_sizes = [(3,3), (3,3)],
        cnn_out_channels = [32, 64],
        conv_stride = (2,2),
        conv_padding = (0,0),
        conv_output = 128,
        rnn_neurons = 128,
        rnn_layers = 1,
        ft_rnn_neurons = 16,
        ft_rnn_layers = 1,
        fc_neurons = [64, 32],
    ))
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

    train_l, test_l, seed = get_dataset(a.dsname, a, batch_size=a.batch_size)
    a.__setattr__("dataset_seed", seed)

    model = TactilePlacingNet(**a.netp)
    optimizer = optim.Adam(model.parameters(), **a.adamp)

    criterion = rep2loss(a.loss_type)

    train_losses = []
    test_losses = []
    val_losses = []

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
            test_loss = np.mean(test_loss)
            test_losses.append(test_loss)

            if a.validate:
                print("validation not supported atm")

            # store model weights
            if nbatch % save_batches == save_batches-1:
                torch.save(model.state_dict(), f"{trial_path}/weights/batch_{nbatch}.pth")
            nbatch += 1

            print(f"[{epoch + 1}, {i + 1:5d}] loss: {train_loss:.5f} | test loss: {test_loss:.5f}", end="")
            print()
    torch.save(model.state_dict(), f"{trial_path}/weights/final.pth")

    with open(f"{trial_path}/losses.pkl", "wb") as f:
        pickle.dump({
            "train": train_losses,
            "test": test_losses,
            "validation": val_losses
        }, f)

    fig, ax = plt.subplots(figsize=(8.71, 6.61))
    plot_learning_curve(train_losses, test_losses, a, ax)
    if other_ax is not None: plot_learning_curve(train_losses, test_losses, a, other_ax, small_title=True)

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
    datasets = [DatasetName.combined_large, DatasetName.cylinder_large, DatasetName.cuboid_large]
    target_type = InRot.g2o

    # full training
    input_types = [InData.static, InData.with_tap]
    input_modalities = [
        [True , False, False],
        [False, True , False],  
        [False, False, True],
        [True , True , False],
        [True , False, True],
        [False, True , True],
        [True , True , True],
    ]

    # quick testing config
    # input_types = [InData.static]
    # input_modalities = [
    #     [True , False, False],
    #     [False, True , False],  
    #     [False, False, True],
    #     [True , True , True],
    # ]

    trial_times = []
    for dataset in datasets:
        dspath = f"{base_path}/{dataset}"
        os.makedirs(dspath, exist_ok=True)

        nrows=len(input_modalities)
        ncols=len(input_types) 
        fig, axs = plt.subplots(nrows,ncols,figsize=(4.3*ncols, 3.3*nrows))

        trials = {}
        train_start = datetime.now()
        for i, input_type in enumerate(input_types):
            for j, input_mod in enumerate(input_modalities):
                oax = axs[j,i] if ncols>1 else axs[j]

                trial_start = datetime.now()
                trialname = train(
                    dataset=dataset,
                    input_type=input_type,
                    input_modalities=input_mod,
                    target_type=target_type,
                    trial_path=dspath,
                    ## TODO add parameters for 3dconv preproc & trafo multiplication
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