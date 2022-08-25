import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

X = torch.Tensor(np.repeat([1], 100).reshape((100,1)))
Y = torch.Tensor(np.repeat([2], 100).reshape((100,1)))
GR = torch.Tensor(np.repeat([3], 100).reshape((100,1)))

tds = TensorDataset(X, GR, Y)

train, test = torch.utils.data.random_split(
    tds,
    [80,20],
    generator=torch.Generator().manual_seed(42)
)
for sample in train:
    print(sample)
pass