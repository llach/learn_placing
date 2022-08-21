import os
import torch

from torch.utils.data import TensorDataset, DataLoader
from learn_placing.common.data import load_dataset_file

def get_dataset_loaders(name, train_ratio=0.8, batch_size=8, shuffle=True):
    dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{name}.pkl"
    ds = load_dataset_file(dataset_file_path)

    print("WARNING labels are still angles, should be polar coordinates!")
    
    X = [v for _, v in ds["inputs"].items()]
    Y = [d["angle"] for d in list(ds["labels"].values())]

    N_train = int(len(X)*train_ratio)
    N_test = len(X)-N_train

    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    tds = TensorDataset(X, Y)

    train, test = torch.utils.data.random_split(
        tds, 
        [N_train, N_test], 
        generator=torch.Generator().manual_seed(42)
    )

    train_l = DataLoader(train, shuffle=shuffle, batch_size=batch_size)
    test_l = DataLoader(test, shuffle=False, batch_size=batch_size)

    return train_l, test_l