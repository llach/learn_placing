import torch
import numpy as np
import torch.nn.functional as F

from numpy.random import randint, choice
from learn_placing.common.utils import marginal_mean

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
    if type(data) == list: data = np.array(data)
    if len(data.shape)==2: data = np.expand_dims(data, axis=0)
    
    imgs = np.zeros(list(data.shape) + [3])
    imgs[:,:,:,cidx] = data
    imgs *= 255
    return np.squeeze(imgs.astype(np.uint8))

def upscale_repeat(frames, factor=10):
    if len(frames.shape)==2: frames = np.expand_dims(frames, axis=0)
    return np.squeeze(frames.repeat(factor, axis=1).repeat(factor, axis=2))

def get_pad_and_slice(shift):
         if shift <= 0:
            return [-shift, 0], slice(0, shift)
         else:
            return [0, shift], slice(shift, 1000)

def shift_columns(frames, rpad, rsli, cpad, csli):
    """ 

    frames.shape = [16,16] = [H,W] (single frame)
    OR 
    frames.shape = [N,16,16] = [batch,H,W] (sequence)

    pad: how many columns we need to pad
    shift: how many columns we'll shift

    padding list: [left, right, top, bottom, front, back]
    -> indices 0,1 are columns, indices 2,3 are rows
    """
    if isinstance(frames, np.ndarray): frames = torch.Tensor(frames)
    return F.pad(frames, pad=cpad+rpad+[0,0])[:,:,rsli,csli].numpy()

def random_shift_seq(seq, augment):
    rows, cols = augment
    if rows: 
        rpad, rsli = get_pad_and_slice(choice([randint(-4,0), randint(1,5)]))
    else:
        rpad, rsli = [0,0], slice(-100000, 100000)
    if cols: 
        cpad, csli = get_pad_and_slice(choice([randint(-4,0), randint(1,5)]))
    else:
        cpad, csli = [0,0], slice(-100000, 100000)
    return shift_columns(seq, rpad=rpad, rsli=rsli, cpad=cpad, csli=csli)

def merge_mm_samples(mm, noise_tresh=0.0):
    """ merge two myrmex samples by flipping the right sensor image, adding force intensities and normalizing. optionally filter noise.
    """
    merged = (mm[0]+np.flip(mm[1], 1))/2
    if noise_tresh > 0.0: merged = np.where(merged>noise_tresh, merged, 0)
    return merged

def get_mean_force_xy(mm):
    return np.array([
        marginal_mean(mm, axis=0),
        marginal_mean(mm, axis=1) 
    ])
    