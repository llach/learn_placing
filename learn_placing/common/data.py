import os
import pickle
import numpy as np

from datetime import datetime
from learn_placing.common.myrmex_processing import preprocess_myrmex

def extract_sample(s, frame_no=10):
    """
    given a sample sequence, extract 
    mm: [left, right] myrmex samples (2x16x16)
    gr: T_world_gripper as quaternion (1x4)
    ft: F/T sensor readings (1x6)
    lbl: target rotation as quaternion (1x6)

    TODO return label based on
    """

    mmleft  = preprocess_myrmex(s["tactile_left"][1])[frame_no,:]  # 16x16 array
    mmright = preprocess_myrmex(s["tactile_right"][1])[frame_no,:] # 16x16 array
    
    w2o, g2o, w2g = None, None, None
    for ops in s["opti_state"][1][0]:
        if ops["parent_frame"] == "gripper_left_grasping_frame" and ops["child_frame"] == "pot":
            g2o = ops["rotation"]
        if ops["parent_frame"] == "base_footprint" and ops["child_frame"] == "pot":
            w2o = ops["rotation"]
        if ops["parent_frame"] == "base_footprint" and ops["child_frame"] == "gripper_left_grasping_frame":
            w2g = ops["rotation"]

    return np.array([mmleft, mmright]), w2g, None, g2o

def load_dataset(folder):
    ds = {}

    for fi in os.listdir(folder):
        if ".pkl" not in fi: continue
        fname = fi.replace(".pkl", "")
        if len(fname)<10:continue
        
        try:
            stamp = datetime.strptime(fname, "%Y-%m-%d.%H_%M_%S")
        except:
            stamp = datetime.strptime(fname, "%Y-%m-%d.%H:%M:%S")
        
        pkl = f"{folder}/{fi}"
        with open(pkl, "rb") as f:
            sample = pickle.load(f)
        ds.update({stamp: sample})
    return ds

def load_dataset_file(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
