import numpy as np
from learn_placing.common import load_dataset, preprocess_myrmex

def build_train_data(ds, Tn=40):
    # this label generation is not in its final state. missing:
    #   * average / filter angle estimates over sample instead of taking first one
    #   * don't use angle, use polar coordinate angles
    #   * how to choose Tn? (look at averages)
    print("!!! WARNING !!! labels are not correct yet")
    Y = [v["object_state"][1][0]["angle"] for k, v in ds.items()]

    tleft = np.array([v["tactile_left"][1][-Tn:] for k, v in ds.items()])
    tright = np.array([v["tactile_right"][1][-Tn:] for k, v in ds.items()])
    
    # swap sensor and batch dimension 
    mm = np.swapaxes([tright, tleft], 0, 1)
    X = preprocess_myrmex(mm)
    return X, Y

dataset_path = "/home/llach/tud_datasets/2022.08.09_first/placing_data_pkl"
ds = load_dataset(dataset_path)
X, Y = build_train_data(ds)
pass