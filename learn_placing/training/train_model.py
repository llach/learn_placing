import os
import json
import torch
import pickle

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from datetime import datetime
from learn_placing.training.utils import rep2loss

from utils import LossType, get_dataset, InData, RotRepr, InRot, DatasetName, test_net, AttrDict
from learn_placing import datefmt, training_path
from tactile_insertion_rl import TactileInsertionRLNet, ConvProc

""" PARAMETERS
"""
a = AttrDict(
    dsname = DatasetName.object_var,
    input_data = InData.with_tap,
    with_gripper_tf = False,
    N_episodes = 10,

    loss_type = LossType.pointarccos,
    out_repr = RotRepr.ortho6d,
    target_type = InRot.w2o,
    validate = False,
    store_training = True,
    gripper_repr = RotRepr.quat,
    start_time = datetime.now().strftime(datefmt),
    save_freq = 0.1
)
a.__setattr__("netp", AttrDict(
    preproc_type = ConvProc.SINGLETRL,
    output_type = a.out_repr,
    with_gripper = a.with_gripper_tf,
    only_gripper = False,
    kernel_sizes = [(3,3), (3,3)],
    cnn_out_channels = [32, 64],
    conv_stride = (2,2),
    conv_padding = (0,0),
    conv_output = 128,
    rnn_neurons = 128,
    rnn_layers = 2,
    fc_neurons = [64, 32],
))
a.__setattr__("adamp", AttrDict(
    lr=1e-3, 
    betas=(0.9, 0.999), 
    eps=1e-8, 
    weight_decay=0, 
    amsgrad=False
))

trial_name = f"{a.dsname}_Neps{a.N_episodes}_{a.out_repr}_{a.target_type}_gripper-{a.with_gripper_tf}_{a.start_time.replace(':','-')}"
trial_path = f"{training_path}/{trial_name}/"

(train_l, train_ind), (test_l, test_ind) = get_dataset(a.dsname, a)

a.__setattr__("train_indices", train_ind)
a.__setattr__("test_indices", test_ind)

model = TactileInsertionRLNet(**a.netp)
optimizer = optim.Adam(model.parameters(), **a.adamp)

criterion = rep2loss(a.loss_type)

train_losses = []
test_losses = []
val_losses = []

os.makedirs(trial_path, exist_ok=True)
os.makedirs(f"{trial_path}/weights", exist_ok=True)
with open(f"{trial_path}parameters.json", "w") as f:
    json.dump(a, f, indent=2)

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

plt.figure(figsize=(8.71, 6.61))

lastN = int(len(train_losses)*0.05)

xs = np.arange(len(test_losses)).astype(int)+1
plt.plot(xs, train_losses, label=f"training loss | {np.mean(train_losses[-lastN:]):.5f}")
plt.plot(xs, test_losses, label=f"test loss | {np.mean(test_losses[-lastN:]):.5f}")
if a.validate: plt.plot(xs, val_losses, label=f"validation loss - {np.mean(val_losses[-lastN:])}")

if a.loss_type == LossType.pointarccos or a.loss_type == LossType.geodesic:
    plt.ylim([0.0,np.pi])
elif a.loss_type == LossType.msesum or a.loss_type == LossType.quaternion or a.loss_type == LossType.pointcos:
    plt.ylim([-0.05,1.0])
plt.ylabel("loss")

plt.xlabel("Batches")

plt.title(f"dsname={a.dsname}; out_repr={a.out_repr}; target={a.target_type}; gripper_tf={a.with_gripper_tf}; \ninput={a.input_data}; loss={a.loss_type};")

plt.legend()
plt.tight_layout()
plt.savefig(f"{trial_path}/learning_curve.png")
plt.clf()

print('training done!')
print(trial_name)