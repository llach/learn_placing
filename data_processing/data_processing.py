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
    N = data.shape[0]
    return np.reshape(data, (N,16,16))


if __name__ == "__main__":
    import pickle

    # read data
    with open(f"{__file__.replace(__file__.split('/')[-1], '')}/2022-07-27-16-44-18.pkl", "rb") as f:
        mm_data = pickle.load(f)
    
    # convert tactile vector to tactile matrix
    left = reshape_mm_vector(mm_data["/tactile_left"])
    right = reshape_mm_vector(mm_data["/tactile_right"])

    # one matrix for all samples
    data = np.concatenate([left, right], axis=0)

    # (optional) since the outermost taxels are prone 
    # to false positives, we can cut them away 
    # data = remove_outer(data, B=1)
    pass