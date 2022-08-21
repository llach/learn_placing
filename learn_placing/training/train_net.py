import torch
import torch.nn as nn
import torch.optim as optim

from utils import get_dataset_loaders
from tactile_insertion_rl import TactileInsertionRLNet

""" PARAMETERS
"""
N_episodes = 5

train_l, test_l = get_dataset_loaders("second")

net = TactileInsertionRLNet(output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

# code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
for epoch in range(N_episodes):  # loop over the dataset multiple times

    running_loss = 0.0
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
        running_loss += loss.item()
        with torch.no_grad():
            test_loss = 0

            for tdata in test_l:
                tinputs, tlabels = tdata
                toutputs = net(tinputs)

                loss = criterion(toutputs, tlabels)
                test_loss += loss.item()

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f} | test loss: {test_loss / len(test_l):.3f}')
        running_loss = 0.0

print('training done!')
pass