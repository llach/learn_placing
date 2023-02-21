import torch
import numpy as np

from learn_placing.common import tf_tools as tft

def q2l(q): return [q.x, q.y, q.z, q.w]
def v2l(vec): return [vec.x, vec.y, vec.z]

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

def line_angle_from_rotation(Rgo):
    """
    calculate the angle to draw on a merged sensor image from the quaternion gripper to object

    first, we determine the angle between the object's z-axis and the gripper's x-axis
    then we rotate by PI/2 to (in the sensor images, the z-axis goes off to the left side)
    """
    Rgo = tft.ensure_rotmat(Rgo)
    zO = Rgo.dot([0,0,1])
    th = np.arctan2(zO[2], zO[0])
    return ensure_positive_angle(th)

def rotation_from_line_angle(th):
    """ line angle to gripper frame-based rotation of object. the offset of PI stems from us flipping and rotating the image before calculating the baseline angles.
    """
    if np.isnan(th): return tft.Ry(-np.pi)
    return tft.Ry(-th+np.pi/2)

def ensure_positive_angle(th): return th if th>=0 else np.pi+th

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