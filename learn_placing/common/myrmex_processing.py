import numpy as np

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
