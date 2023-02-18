import torch
import numpy as np

from learn_placing.common import tf_tools as tft

def marginal_mean(arr, axis):
    '''
    E(x) = int x f(x) dx
    '''
    mardens = arr.sum(axis=axis)
    mardens /= np.sum(mardens) ## rescale so that marginal density adds up to 1
    
    meanx   = np.sum((np.arange(len(mardens))*mardens))
    return meanx

def marginal_sd(arr, axis):
    """ E( (x - barx)**2 ) = int (x-barx)**2 f(x) dx
    """
    mardens = arr.sum(axis=axis)
    
    meanx   = marginal_mean(arr, axis)
    varx    = np.sqrt(np.sum((np.arange(len(mardens)) - meanx)**2*mardens))
    return varx

def get_cov(arr):
    """ E( (x - barx)*(y - bary) ) = int (x-barx)*(y - bary) f(x,y) dxdy
    """
    
    x_coor_mat = np.zeros_like(arr)
    for icol in range(x_coor_mat.shape[1]):
        x_coor_mat[:, icol] = icol
        
    y_coor_mat = np.zeros_like(arr)
    for irow in range(y_coor_mat.shape[0]):
        y_coor_mat[irow, :] = irow
        
    meanx   = marginal_mean(arr, axis=0)
    meany   = marginal_mean(arr, axis=1) 
    
    cov     = np.sum((x_coor_mat - meanx)*(y_coor_mat - meany )*arr)
    return cov

def rotmat_to_theta(pred):
    """ given a NN prediction (3x3 rotation matrix), return the object's angle offset
    """
    return np.pi-np.arccos(np.dot([0,0,-1], pred.dot([0,0,-1])))

def label_to_theta(y):
    if type(y) == list: y=np.array(y)
    if y.shape == (4,4): y = y[:3,:3]
    elif y.shape == (4,) or y.shape == (1,4):
        y = tft.quaternion_matrix(y)[:3,:3]
    elif y.shape == (3,3): pass
    else:
        print(f"ERROR shape mismatch: {y.shape}")
    
    return rotmat_to_theta(y)

def line_similarity(th, lblth):
    """ 
    two (intersecting) lines will always have two different angles descirbing the intersection (of both are 90deg).
    to determine similarity, we take the smaller one.
    two lines are most dissimilar if their angle(s) are 90deg.
    """
    ang_diff = np.abs(th-lblth)
    return ang_diff if ang_diff < np.pi/2 else np.pi-ang_diff

def single_pred_loss(pred, label, f):
    """
    pred    (3x3) network output, rotation matrix
    label   (1x4) label, quaternion
    f       loss function
    """
    p = np.expand_dims(pred, 0)
    l = np.expand_dims(tft.quaternion_matrix(label), 0)
    return f(torch.Tensor(p), torch.Tensor(l))

def to_tensors(*args):
    return [torch.Tensor(a) for a in args]

def to_numpy(*args):
    return [a.numpy() for a in args]