import os
import pickle
import numpy as np
from datetime import datetime

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
    if len(data.shape)==1: data = np.reshape(data, [1]+list(data.shape))
    N = data.shape[0]
    return np.reshape(data, list(data.shape[:-1])+[16,16])

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

def mm2img(data, cidx=2):
    imgs = np.zeros(list(data.shape) + [3])
    imgs[:,:,:,cidx] = data
    imgs *= 255
    return imgs.astype(np.uint8)

def upscale_repeat(frames, factor=10):
    return frames.repeat(factor, axis=1).repeat(factor, axis=2)

def load_dataset(folder):
    ds = {}

    for fi in os.listdir(folder):
        if ".pkl" not in fi: continue
        fname = fi.replace(".pkl", "")
        if len(fname)<10:continue
        
        stamp = datetime.strptime(fname, "%Y-%m-%d.%H_%M_%S")
        
        pkl = f"{folder}/{fi}"
        with open(pkl, "rb") as f:
            sample = pickle.load(f)
        ds.update({stamp: sample})
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