import os
import pickle

import numpy as np
from datetime import timedelta
from learn_placing.common.label_processing import normalize, rotate_v

from learn_placing.common.vecplot import AxesPlot
from learn_placing.common.transformations import quaternion_conjugate, quaternion_from_matrix, quaternion_matrix, quaternion_multiply, quaternion_inverse
from learn_placing.common import load_dataset, cam_stats, qO2qdiff, v_from_qdiff, qavg, preprocess_myrmex, extract_gripper_T
from learn_placing.training.utils import InRot


""" PARAMETERS
"""
ZETA = timedelta(milliseconds=100)
MAX_DEV = 0.005
MIN_N = 10 # per camera
M = 50  # myrmex lookback

data_root = f"{os.environ['HOME']}/tud_datasets"
dataset_path = f"{data_root}/placing_data_pkl_second"
dataset_file = f"{data_root}/second.pkl"
# dataset_path = f"{data_root}/placing_data_pkl_third"
# dataset_file = f"{data_root}/third.pkl"

# sample timestamp -> sample
ds = load_dataset(dataset_path)

# step 1: filter object state samples at or after contact
os = {}
for k, v in ds.items():
    # [ [timetsamp], [object state]]
    o = v["object_state"]

    # get perceived contact time, subtract a bit to adjust for delay
    contact_time = v["bag_times"][0][1]
    last_time = contact_time-ZETA
    
    # only keep samples before last allowed time
    o_filtered = [[], []]
    for t, state in zip(*o):
        if t < last_time: 
            o_filtered[0].append(t)
            o_filtered[1].append(state)
    os.update({k: o_filtered})

min_lens=100

bad_timestamps = []
labels = {}

for i, (t, sample) in enumerate(os.items()):
    # sample is sequence: [ [timetsamp], [object state]]
    states = sample[1]
    
    # stats: {cam: [N, mu, std]}
    _, stats, dists = cam_stats(states)

    # we ignore cams if they have a high deviation in samples or too few measurements
    ignored_cams = [k for k, v in stats.items() if v[2]>MAX_DEV or v[0] < MIN_N]

    # if all cams are ignored, we ignore the sample
    if len(ignored_cams)==len(stats):
        print(f"sample {i} is bad! --> ignore")
        bad_timestamps.append(t)
        continue

    # remove bad cameras
    for ic in ignored_cams: stats.pop(ic)

    # get the one with lowest measurement deviation    
    chosen_cam = ""
    chosen_dev = MAX_DEV
    chosen_N = 0
    for cam, sta in stats.items():
        if sta[2]<chosen_dev: 
            chosen_cam = cam
            chosen_dev = sta[2]
            chosen_N = sta[0]

    # extract cam measurements
    all_quaternions = []
    for st in states:
        for q, cam in zip(st["qOs"], st["cameras"]):
            if cam == chosen_cam: all_quaternions.append(q)

    finalq = qO2qdiff(qavg(all_quaternions))
    finalv = v_from_qdiff(finalq)

    # extract gripper transforms
    tfs = ds[t]["tf"][1]
    T, world2obj, grip2obj = extract_gripper_T(tfs)

    # world -> gripper
    qWG = quaternion_from_matrix(T)
    qWG = normalize(qWG)

    # gripper -> object
    qGO = quaternion_multiply(quaternion_inverse(qWG), finalq)
    qGO = normalize(qGO)

    Rgo = quaternion_matrix(qGO)
    Zgo = Rgo@[0,0,1,1]
    ZgoNorm = normalize([Zgo[2], -Zgo[0]])
    gripper_angle = np.arctan2(ZgoNorm[1], ZgoNorm[0])

    # axp = AxesPlot()

    # published gripper to object transform VS the one we calculated based on tf msgs
    # axp.plot_v(rotate_v([0,0,-1], grip2obj[0]), color="grey", label="published tf")
    # axp.plot_v(rotate_v([0,0,-1], qGO), color="black", label="calculated tf")
    # axp.title("gripper -> object TF")

    # calculated object tf after filtering noisy camera measurements vs the one recorded during sample collection.
    # might be off a little bit since the publised tf can be noisy, but not too much off across samples
    # axp.plot_v(rotate_v([0,0,-1], world2obj[0]), color="grey", label="published tf")
    # axp.plot_v(rotate_v([0,0,-1], finalq), color="black", label="calculated tf")
    # axp.title("world -> object TF")

    # make sure the FK + gripper -> object transform matches the finalq we calculate based on measurements
    # axp.plot_v(rotate_v([0,0,-1], quaternion_multiply(qWG, qGO)), color="grey", label="complete tf")
    # axp.plot_v(rotate_v([0,0,-1], finalq), color="black", label="finalq")
    # axp.title("world -> object Quat.Mult.")

    # axp.show()

    angle = np.dot(finalv, [0,0,-1])

    labels.update({
        t: {
            InRot.w2o: finalq,
            InRot.w2g: qWG,
            InRot.g2o: qGO,
            InRot.gripper_angle: gripper_angle,
            "vec": finalv,
            "angle": angle,
        }
    })

inputs = {}
for i, (t, sample) in enumerate(ds.items()):
    if t not in labels:
        print(f"skipping myrmex sample {i}")
        continue
    
    le = preprocess_myrmex(sample["tactile_left"][1])
    ri = preprocess_myrmex(sample["tactile_right"][1])

     # cut to same length (we determined that in `myrmez_lookback.py`)
    ri = ri[-M:]
    le = le[-M:]

    inp = np.stack([le, ri])

    inputs.update({t: inp})

with open(dataset_file, "wb") as f:
    pickle.dump({"labels": labels, "inputs": inputs}, f)