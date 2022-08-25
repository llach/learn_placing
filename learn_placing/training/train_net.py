import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from learn_placing.common.label_processing import rotate_v, normalize
from learn_placing.common.vecplot import AxesPlot
from learn_placing.training.utils import qloss, RotRepr, InRot, DatasetName

from utils import get_dataset_loaders, compute_geodesic_distance_from_two_matrices
from tactile_insertion_rl import TactileInsertionRLNet

""" PARAMETERS
"""
dsname = DatasetName.cuboid
with_gripper_tf = True
N_episodes = 3
out_repr = RotRepr.ortho6d
gripper_repr = RotRepr.quat
target_type = InRot.w2o

train_cub, test_cub = get_dataset_loaders("second", target_type=target_type, out_repr=out_repr, train_ratio=0.8)
train_cyl, test_cyl = get_dataset_loaders("third", target_type=target_type, out_repr=out_repr, train_ratio=0.5)

if dsname == DatasetName.cuboid:
    train_l = train_cub
    test_l = test_cub
elif dsname == DatasetName.cylinder:
    train_l = train_cyl
    test_l = test_cyl

net = TactileInsertionRLNet(
    output_type = out_repr,
    with_gripper = with_gripper_tf,
    kernel_sizes = [(3,3), (3,3)],
    cnn_out_channels = [32, 64],
    conv_stride = (2,2),
    conv_padding = (0,0),
    conv_output = 64,
    rnn_neurons = 64,
    rnn_layers = 2,
    fc_neurons = [32, 16],
)
optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

if out_repr == RotRepr.quat:
    # criterion = lambda a, b: torch.sqrt(qloss(a,b)) 
    criterion = qloss
elif out_repr == RotRepr.ortho6d:
    criterion = compute_geodesic_distance_from_two_matrices
elif out_repr == RotRepr.sincos:
    criterion = nn.MSELoss()

train_losses = []
test_losses = []
cyl_losses = []

# code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
for epoch in range(N_episodes):  # loop over the dataset multiple times
    for i, data in enumerate(train_l, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, grip, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs, grip)
        loss = torch.mean(criterion(outputs, labels))
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss = loss.item()
        train_losses.append(train_loss)
        with torch.no_grad():
            test_loss = 0

            for tdata in test_l:
                tinputs, tgrip, tlabels = tdata
                toutputs = net(tinputs, tgrip)

                tloss = torch.mean(criterion(toutputs, tlabels))
                test_loss += tloss.item()
            test_loss /= len(test_l)
            test_losses.append(test_loss)

        # OOS cylinder dataset
        # with torch.no_grad():
        #     cyl_loss = 0

        #     for cdata in train_cyl:
        #         cinputs, cgrip, clabels = cdata
        #         coutputs = net(cinputs)

        #         closs = criterion(coutputs, clabels)
        #         cyl_loss += closs.item()
        #     cyl_loss /= len(train_cyl)
        #     cyl_losses.append(cyl_loss)

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss:.5f} | test loss: {test_loss:.5f}')# | cyl loss: {cyl_loss:.3f}')

# for tdata in test_l:
#     with torch.no_grad():
#         tinputs, tgrip, tlabels = tdata
#         toutputs = net(tinputs)

#         loss = criterion(toutputs, tlabels)

#         for out, lbl, lo in zip(toutputs.numpy(), tlabels.numpy(), loss.numpy()):
#             axp = AxesPlot()

#             if out_repr == OutRepr.quat:
#                 ov = rotate_v([0,0,-1], normalize(out))
#                 lv = rotate_v([0,0,-1], lbl)
#             elif out_repr == OutRepr.ortho6d:
#                 ov = out@[0,0,-1]
#                 lv = lbl@[0,0,-1]

#             axp.plot_v(ov, label=f"out {np.squeeze(lo):.5f}", color="black")
#             axp.plot_v(lv, label="lbl", color="grey")
#             axp.show()

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8.71, 6.61))

xs = np.arange(len(test_losses)).astype(int)+1
plt.plot(xs, train_losses, label="training loss")
plt.plot(xs, test_losses, label="test loss")
# plt.plot(xs, cyl_losses, label="cylinder test loss")

if out_repr == RotRepr.ortho6d:
    plt.ylim([0.0,np.pi])
    plt.ylabel("geodesic error in [0,PI]")
elif out_repr == RotRepr.quat:
    plt.ylim([0.0,1.0])
    plt.ylabel("quaternion loss")

plt.xlabel("Batches")

plt.title(f"dsname={dsname}; out_repr={out_repr}; target={target_type}; gripper_tf={with_gripper_tf}")

plt.legend()
plt.tight_layout()
plt.savefig(f"{os.environ['HOME']}/tud_datasets/Neps{N_episodes}_{out_repr}_{target_type}.png")
plt.show()

print('training done!')