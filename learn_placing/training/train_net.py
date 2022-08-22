import torch
import torch.nn as nn
import torch.optim as optim

from utils import get_dataset_loaders
from tactile_insertion_rl import TactileInsertionRLNet

""" PARAMETERS
"""
N_episodes = 1

train_l, test_l = get_dataset_loaders("second", train_ratio=0.8)
train_cyl, test_cyl = get_dataset_loaders("third", train_ratio=0.5)

net = TactileInsertionRLNet(output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

train_losses = []
test_losses = []
cyl_losses = []

# code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
for epoch in range(N_episodes):  # loop over the dataset multiple times
    for i, data in enumerate(train_l, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss = loss.item()
        train_losses.append(train_loss)
        with torch.no_grad():
            test_loss = 0

            for tdata in test_l:
                tinputs, tlabels = tdata
                toutputs = net(tinputs)

                tloss = criterion(toutputs, tlabels)
                test_loss += tloss.item()
            test_loss /= len(test_l)
            test_losses.append(test_loss)

        # OOS cylinder dataset
        with torch.no_grad():
            cyl_loss = 0

            for cdata in train_cyl:
                cinputs, clabels = cdata
                coutputs = net(cinputs)

                closs = criterion(coutputs, clabels)
                cyl_loss += closs.item()
            cyl_loss /= len(train_cyl)
            cyl_losses.append(cyl_loss)

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss:.3f} | test loss: {test_loss:.3f} | cyl loss: {cyl_loss:.3f}')

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8.71, 6.61))

xs = np.arange(len(test_losses)).astype(int)+1
plt.plot(xs, train_losses, label="training loss")
plt.plot(xs, test_losses, label="test loss")
plt.plot(xs, cyl_losses, label="cylinder test loss")

plt.xlabel("Batches")
plt.ylabel("MSE Loss (angle)")
plt.title("Training Results (dot product) on Cubiod Dataset (Tactile Insertion Net)")

plt.legend()
plt.tight_layout()
plt.show()

print('training done!')
pass