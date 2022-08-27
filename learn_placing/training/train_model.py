import os
import json
import torch
import pickle

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from datetime import datetime

from utils import get_dataset_loaders, compute_geodesic_distance_from_two_matrices, qloss, RotRepr, InRot, DatasetName, test_net, AttrDict
from learn_placing import datefmt, training_path
from tactile_insertion_rl import TactileInsertionRLNet

""" PARAMETERS
"""
a = AttrDict(
    dsname = DatasetName.cuboid,
    with_gripper_tf = True,
    gripper_repr = RotRepr.quat,
    N_episodes = 50,
    out_repr = RotRepr.ortho6d,
    target_type = InRot.w2o,
    validate = False,
    store_training = True,
    start_time = datetime.now().strftime(datefmt),
    save_freq = 0.1
)
a.__setattr__("netp", AttrDict(
    output_type = a.out_repr,
    with_gripper = a.with_gripper_tf,
    kernel_sizes = [(3,3), (3,3)],
    cnn_out_channels = [32, 64],
    conv_stride = (2,2),
    conv_padding = (0,0),
    conv_output = 64,
    rnn_neurons = 64,
    rnn_layers = 2,
    fc_neurons = [32, 16],
))
a.__setattr__("adamp", AttrDict(
    lr=1e-3, 
    betas=(0.9, 0.999), 
    eps=1e-8, 
    weight_decay=0, 
    amsgrad=False
))

trial_path = f"{training_path}/{a.dsname}_Neps{a.N_episodes}_{a.out_repr}_{a.target_type}_gripper-{a.with_gripper_tf}_{a.start_time.replace(':','-')}/"
os.makedirs(trial_path, exist_ok=True)
os.makedirs(f"{trial_path}/weights", exist_ok=True)

if a.dsname == DatasetName.cuboid:
    (train_l, train_ind), (test_l, test_ind) = get_dataset_loaders("second", target_type=a.target_type, out_repr=a.out_repr, train_ratio=0.8)
    (val_l, val_ind), _ = get_dataset_loaders("third", target_type=a.target_type, out_repr=a.out_repr, train_ratio=0.5)
elif a.dsname == DatasetName.cylinder:
    (train_l, train_ind), (test_l, test_ind) = get_dataset_loaders("third", target_type=a.target_type, out_repr=a.out_repr, train_ratio=0.8)
    (val_l, val_ind), _ = get_dataset_loaders("second", target_type=a.target_type, out_repr=a.out_repr, train_ratio=0.5)

a.__setattr__("train_indices", train_ind)
a.__setattr__("test_indices", test_ind)

with open(f"{trial_path}parameters.json", "w") as f:
    json.dump(a, f, indent=2)

model = TactileInsertionRLNet(**a.netp)
optimizer = optim.Adam(model.parameters(), **a.adamp)

if a.out_repr == RotRepr.quat:
    # criterion = lambda a, b: torch.sqrt(qloss(a,b)) 
    criterion = qloss
elif a.out_repr == RotRepr.ortho6d:
    criterion = compute_geodesic_distance_from_two_matrices
elif a.out_repr == RotRepr.sincos:
    criterion = nn.MSELoss()

train_losses = []
test_losses = []
val_losses = []

# code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
nbatch = 0
save_batches = int(a.N_episodes*len(train_l)*a.save_freq)
for epoch in range(a.N_episodes):  # loop over the dataset multiple times
    for i, data in enumerate(train_l, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, grip, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs, grip)
        loss = torch.mean(criterion(outputs, labels))
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss = loss.item()
        train_losses.append(train_loss)

        test_out, test_lbl, test_loss = test_net(model, criterion, test_l)
        test_loss = np.mean(test_loss)
        test_losses.append(test_loss)

        if a.validate:
            _, _, val_loss = test_net(model, criterion, val_l)

            val_loss = np.mean(val_loss)
            val_losses.append(val_loss)

        # store model weights
        if nbatch % save_batches == save_batches-1:
            torch.save(model.state_dict, f"{trial_path}/weights/batch_{nbatch}.pth")
        nbatch += 1

        print(f"[{epoch + 1}, {i + 1:5d}] loss: {train_loss:.5f} | test loss: {test_loss:.5f}", end="")
        if a.validate:
            print(f" | val loss: {val_loss:.3f}", end="")
        print()

# from learn_placing.common.label_processing import rotate_v, normalize
# from learn_placing.common.vecplot import AxesPlot

# plotting to verify network works
# foutputs, flabels, floss = test_net(net, criterion, test_l)
# for out, lbl, lo in zip(foutputs, flabels, floss):
#     axp = AxesPlot()

#     if out_repr == RotRepr.quat:
#         ov = rotate_v([0,0,-1], normalize(out))
#         lv = rotate_v([0,0,-1], lbl)
#     elif out_repr == RotRepr.ortho6d:
#         ov = out@[0,0,-1]
#         lv = lbl@[0,0,-1]

#     axp.plot_v(ov, label=f"out {lo:.5f}", color="black")
#     axp.plot_v(lv, label="lbl", color="grey")
#     axp.show()

with open(f"{trial_path}/losses.pkl", "wb") as f:
    pickle.dump({
        "train": train_losses,
        "test": test_losses,
        "validation": val_losses
    }, f)

plt.figure(figsize=(8.71, 6.61))

lastN = int(len(train_losses)*0.05)

xs = np.arange(len(test_losses)).astype(int)+1
plt.plot(xs, train_losses, label=f"training loss - {np.mean(train_losses[-lastN:])}")
plt.plot(xs, test_losses, label=f"test loss - {np.mean(test_losses[-lastN:])}")
if a.validate: plt.plot(xs, val_losses, label=f"validation loss - {np.mean(val_losses[-lastN:])}")

if a.out_repr == RotRepr.ortho6d:
    plt.ylim([0.0,np.pi])
    plt.ylabel("geodesic error in [0,PI]")
elif a.out_repr == RotRepr.quat:
    plt.ylim([0.0,1.0])
    plt.ylabel("quaternion loss")

plt.xlabel("Batches")

plt.title(f"dsname={a.dsname}; out_repr={a.out_repr}; target={a.target_type}; gripper_tf={a.with_gripper_tf}")

plt.legend()
plt.tight_layout()
plt.savefig(f"{trial_path}/learning_curve.png")
plt.clf()

print('training done!')