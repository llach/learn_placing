import os
import torch
import numpy as np
import torch.nn as nn

from enum import Enum
from torch.utils.data import TensorDataset, DataLoader
from learn_placing.common.data import load_dataset_file

class OutRepr(str, Enum):
    ortho6d="ortho6d"
    quat="quat"
    sincos="sincos"

class InRot(str, Enum):
    w2o = "world2object"
    w2g = "world2gripper"
    g2o = "gripper2object"
    gripper_angle = "gripper_angle"

def get_dataset_loaders(name, target_type=InRot.w2o, out_repr=OutRepr.quat, train_ratio=0.8, batch_size=8, shuffle=True):
    dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{name}.pkl"
    ds = load_dataset_file(dataset_file_path)
    
    X = [v for _, v in ds["inputs"].items()]
    Y = [d[target_type] for d in list(ds["labels"].values())]
    if out_repr==OutRepr.sincos: Y = np.stack([np.sin(Y), np.cos(Y)], axis=1)

    N_train = int(len(X)*train_ratio)
    N_test = len(X)-N_train

    X = torch.Tensor(np.array(X))
    Y = torch.Tensor(np.array(Y))
    if out_repr==OutRepr.ortho6d: Y = compute_rotation_matrix_from_quaternion(Y)
    
    tds = TensorDataset(X, Y)

    train, test = torch.utils.data.random_split(
        tds, 
        [N_train, N_test], 
        generator=torch.Generator().manual_seed(42)
    )

    train_l = DataLoader(train, shuffle=shuffle, batch_size=batch_size)
    test_l = DataLoader(test, shuffle=False, batch_size=batch_size)

    return train_l, test_l

def wrap_torch_fn(fn, *args, **kwargs):
    """ receives torch function and args as np array,
        calls function with arrays as tensors and returns numpy array
    """
    args = [torch.Tensor(a) for a in args]
    return fn(*args, **kwargs).numpy()

# https://math.stackexchange.com/questions/90081/quaternion-distance
# https://link.springer.com/chapter/10.1007/978-3-030-50936-1_106
def qloss(out, lbl):
    assert out.shape == lbl.shape, f"{out.shape} == {lbl.shape}"
    batch = out.shape[0]
    qdim = 4

    out = nn.functional.normalize(out)
    lbl = nn.functional.normalize(lbl)

    return 1-torch.square(torch.bmm(out.view([batch,1,qdim]),lbl.view(batch,qdim,1)))

def qloss_sqrt(out, lbl):
    return torch.sqrt(qloss(out, lbl))

"""
matrices batch*3*3
both matrix are orthogonal rotation matrices
out theta between 0 to 180 degree batch
-> theta in [0.0,PI]

from: https://github.com/thohemp/6DRepNet/blob/master/loss.py#L12
original: https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py#L282
"""
def compute_geodesic_distance_from_two_matrices(m1, m2, eps=1e-7):
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
        
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2        
    theta = torch.acos(torch.clamp(cos, -1+eps, 1-eps))
        
    return theta

# batch*n
# https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py#L20
def normalize_vector( v, return_mag=False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag,torch.FloatTensor([1e-8]))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v

# u, v batch*n
# https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py#L32
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out

#poses batch*6
#poses
# https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py#L47
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

#quaternion batch*4
# https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py#L107
def compute_rotation_matrix_from_quaternion( quaternion):
    batch=quaternion.shape[0]
    
    quat = normalize_vector(quaternion).contiguous()
    
    qw = quat[...,0].contiguous().view(batch, 1)
    qx = quat[...,1].contiguous().view(batch, 1)
    qy = quat[...,2].contiguous().view(batch, 1)
    qz = quat[...,3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation  
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    return matrix