import numpy as np

from learn_placing.common.transformations import *

def rotate_vector_about_q(v, q):
    v = list(unit_vector(v))
    v.append(0.0) # vector as pure quaternion, i.e. normalized and 4D
    return quaternion_multiply(
        quaternion_multiply(q, v),
        quaternion_conjugate(q)
    )[:3]

def ensure_homog(r):
    R = ensure_rotmat(r)
    T = np.eye(4)
    T[:3,:3] = R

    return T

def ensure_rotmat(y):
    if type(y) != np.array(y): y=np.array(y)
    y = np.squeeze(y)
    if y.shape == (4,4): y = y[:3,:3]
    elif y.shape == (4,) or y.shape == (1,4): y = quaternion_matrix(y)[:3,:3]
    elif y.shape == (3,3): pass
    else:
        print(f"ERROR shape mismatch: {y.shape}")
        return None
    return y