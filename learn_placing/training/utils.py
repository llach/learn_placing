import os
import json

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from learn_placing import dataset_path
from learn_placing.common.data import load_dataset_file
from learn_placing.common.myrmex_processing import random_shift_seq
from learn_placing.common.transformations import quaternion_matrix

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DatasetName(str, Enum):
    cyl23 = "Cyliner"
    cub23 = "Cuboid"
    combined23v1 ="23v1"


ds2name = {
    DatasetName.cyl23: "23_cyl",
    DatasetName.cub23: "23_cub",
    DatasetName.combined23v1: "combined23v1",
}


class RotRepr(str, Enum):
    ortho6d="ortho6d"
    quat="quat"
    sincos="sincos"
    angle="angle"

class InRot(str, Enum):
    w2o = "world2object"
    w2g = "world2gripper"
    g2o = "gripper2object"
    gripper_angle = "gripper_angle"
    gripper_angle_x = "gripper_angle_x"
    w2cleanX = "world2object_cleanX"
    w2cleanZ = "world2object_cleanZ"
    local_dotp = "local_dotproduct"

class InData(str, Enum):
    static = "static"

indata2key = {
    InData.static: "static_inputs"
}

class LossType(str, Enum):
    geodesic = "geodesic"
    quaternion = "quatloss"
    msesum = "msesum"
    pointcos = "pointcos"
    pointarccos = "pointarccos"
    line_sim = "line_similarity"

def load_train_params(trial_path):
    with open(f"{trial_path}/parameters.json", "r") as f:
        params = json.loads(f.read())
    params = AttrDict(**params)
    params.netp = AttrDict(**params.netp)
    params.adamp = AttrDict(**params.adamp)

    try:
        params.val_indices
    except:
        params.__setattr__("val_indices", [])
    return params

def get_dataset(dsname, a, target_type, out_repr, seed=None, train_ratio=0.8):
    if seed is None: seed = np.random.randint(np.iinfo(np.int64).max)

    if "augment" not in a:
        a.update({"augment": None, "aug_n": 0})

    if dsname in [DatasetName.combined23v1]:
        if dsname == DatasetName.combined23v1:
            dss = [
                ds2name[DatasetName.cyl23], 
                ds2name[DatasetName.cub23],
            ]

        trainds, testds = load_concatds(dss, seed=seed, target_type=target_type, out_repr=out_repr, train_ratio=train_ratio, augment=a.augment, aug_n=a.aug_n)
    else:
        dname = dsname if dsname not in ds2name else ds2name[dsname]
        trainds, testds = load_tensords(dname, seed=seed, target_type=target_type, out_repr=out_repr, train_ratio=train_ratio, augment=a.augment, aug_n=a.aug_n)

    train_l = DataLoader(trainds, shuffle=True, batch_size=a.batch_size)
    test_l = DataLoader(testds, shuffle=False, batch_size=a.batch_size) if testds is not None else None

    return train_l, test_l, seed

def split_tds(tds, seed, train_ratio):
    N_train = int(len(tds)*train_ratio)
    N_test = len(tds)-N_train
    if train_ratio != 0.0: 
        return torch.utils.data.random_split(
            tds, 
            [N_train, N_test], 
            generator=torch.Generator().manual_seed(seed)
        )
    return tds, None

def load_concatds(dsnames, seed, target_type, out_repr, train_ratio=0.8, augment=None, aug_n=None):
    tdss = []
    for ds in dsnames:
        tdss.append(load_tensords(ds, seed, target_type, out_repr, train_ratio=0.0, augment=augment, aug_n=aug_n)[0])
    return split_tds(ConcatDataset(tdss), seed=seed, train_ratio=train_ratio)

def load_tensords(name, seed, target_type, out_repr, train_ratio=0.8, augment=None, aug_n=None):
    dataset_file_path = f"{dataset_path}/{name}.pkl"
    ds = load_dataset_file(dataset_file_path)

    ds_sorted = {}
    for mod, dat in ds.items():
        ds_sorted.update({mod: dict([(sk, dat[sk]) for sk in sorted(dat)])})
    ds = ds_sorted
    
    X =  [v for v in ds[indata2key[InData.static]].values()]
    Y =  [d[target_type] for d in list(ds["labels"].values())]
    GR = [d[InRot.w2g] for d in list(ds["labels"].values())]
    FT = [[f] for f in ds["static_ft"].values()]

    if out_repr==RotRepr.sincos:  Y = np.stack([np.sin(Y), np.cos(Y)], axis=1)
    if out_repr==RotRepr.ortho6d: Y = [quaternion_matrix(y)[:3,:3] for y in Y]

    X =  torch.Tensor(np.array(X))
    Y =  torch.Tensor(np.array(Y))
    GR = torch.Tensor(np.array(GR))
    FT = torch.Tensor(np.array(FT))
    
    if augment is not None and aug_n > 0 and np.any(augment):
        print(f"augmenting dataset: {aug_n} times, rows and cloumns: {augment}")
        XSshape = np.array(X.shape)
        XSshape[0] *= aug_n
        XS = np.zeros(XSshape)

        for a in range(aug_n):
            for i, x in enumerate(X):
                sseq = random_shift_seq(x, augment)
                XS[a*X.shape[0]+i] = sseq
        X = torch.cat([X, torch.Tensor(XS)], axis=0)
        Y = Y.repeat(aug_n+1,1,1)
        GR = GR.repeat(aug_n+1,1)
        FT = FT.repeat(aug_n+1,1,1)

    tds = TensorDataset(X, GR, FT, Y)
    return split_tds(tds, seed=seed, train_ratio=train_ratio)


def test_net(model, crit, dataset):
    losses = []
    outputs = []
    labels = []
    grip_rots = []

    model.eval()
    with torch.no_grad():
        for data in dataset:
            inputs, grip, ft, lbls = data
            outs = model(inputs, grip, ft)
            loss_t = crit(outs, lbls)

            losses.append(loss_t.numpy())
            outputs.append(outs.numpy())
            labels.append(lbls.numpy())
            grip_rots.append(grip.numpy())
    model.train()
    return np.concatenate(outputs, axis=0), np.concatenate(labels, axis=0), np.concatenate(losses, axis=0), np.concatenate(grip_rots, axis=0)

def get_loss_fn(loss_type):
    if loss_type == LossType.quaternion:
        # criterion = lambda a, b: torch.sqrt(qloss(a,b)) 
        return qloss
    elif loss_type == LossType.geodesic:
        return compute_geodesic_distance_from_two_matrices
    elif loss_type == LossType.msesum:
        return lambda x, y: torch.sum(F.mse_loss(x, y, reduction='none'), axis=1)
    elif loss_type == LossType.pointarccos:
        return lambda x, y: point_loss(x, y)
    elif loss_type == LossType.pointcos:
        return lambda x, y: 1-torch.cos(point_loss(x, y))
    elif loss_type == LossType.line_sim:
        return line_similarity_th

def bdot(v1, v2):
    batch = v1.shape[0]
    vdim = v1.shape[1]
    return torch.bmm(v1.view([batch,1,vdim]),v2.view([batch,vdim,1])).squeeze()

def wrap_torch_fn(fn, *args, **kwargs):
    """ receives torch function and args as np array,
        calls function with arrays as tensors and returns numpy array
    """
    args = [torch.Tensor(np.array(a)) for a in args]
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

def line_similarity_th(th, lblth): 
    ang_diff = torch.abs(th-lblth)
    ang_diff = torch.where(torch.le(ang_diff, torch.pi/2), ang_diff, torch.pi-ang_diff)
    return ang_diff

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

def point_loss(m1, m2, eps=1e-7):
    batch = m1.shape[0]

    vs = torch.Tensor([0,0,-1]).repeat(batch,1).unsqueeze(2)

    v1 = torch.bmm(m1[:,:3,:3], vs)
    v2 = torch.bmm(m2[:,:3,:3], vs)

    cos = bdot(v1, v2)
    theta = torch.acos(torch.clamp(cos, -1+eps, 1-eps))
    return theta
    # return 1-cos

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