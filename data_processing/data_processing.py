import os
import numpy as np
from yaml import load

"""
TODO what is max, what is min value
"""

def remove_outer(data, B=0):
    """
    expects an array of (N, 16, 16), with N = number of samples
    removes the outermost rows / columns of the 16x16 matrices
    """
    if B==0: return data
    assert B<=5, f"B<=5"
    return data[:,B:-B,B:-B]

def reshape_mm_vector(data):
    """ tactile data is published as a 256 vector, but we need it as 16x16 matrix
    """
    data = np.array(data)
    N = data.shape[0]
    return np.reshape(data, (N,16,16))

def normalize_mm(data):
    """ converts values from range [0,4095] to [0,1], 1 == maximum force (hence 1-data)
    """
    return 1-(data/4095)

def preprocess_myrmex(data):
    # (optional) since the outermost taxels are prone 
    # to false positives, we can cut them away 
    # data = remove_outer(data, B=1)
    return normalize_mm(reshape_mm_vector(data))

def sync_mm_sample(m1, m2):
    """ TODO here we sync samples: right now it's some dummy function
    """
    return np.array([
        m1[:40],
        m2[:40]
    ])

def load_dataset(folder):
    ds = {
        "tactile": [],
        "obj_vector": [],
        "angle": [],
        "ft": []
    }
    for fi in os.listdir(folder):
        if "pkl" not in fi: continue
        
        pkl = f"{folder}/{fi}"
        with open(pkl, "rb") as f:
            sample = pickle.load(f)
        
        ds["tactile"].append(sync_mm_sample(
            preprocess_myrmex(sample["myrmex_left"]),
            preprocess_myrmex(sample["myrmex_right"])
        ))
    # dimensions: (batch, sensors, sequence, H, W)
    ds["tactile"] = np.array(ds["tactile"])
    return ds

if __name__ == "__main__":
    import torch
    import pickle
    
    from torch.utils.tensorboard import SummaryWriter

    from tactile_insertion_rl import TactileInsertionRLNet

    base_path = f"{__file__.replace(__file__.split('/')[-1], '')}"
    ds = load_dataset(f"{base_path}/test_samples")

    net = TactileInsertionRLNet()
    res = net(ds["tactile"])
    print(res.shape)

    # summary(net, input_size=(2, 40, 16, 16))
    
    # sw = SummaryWriter()
    # sw.add_graph(net, torch.randn((30, 2, 40, 16, 16)))
    # sw.close()
    
    pass