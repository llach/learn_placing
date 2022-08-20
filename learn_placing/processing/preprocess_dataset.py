import os 

import numpy as np
from vecplot import AxesPlot
from datetime import timedelta
from preprocessing import cam_stats, qO2qdiff, v_from_qdiff, qavg
from data_processing import load_dataset


""" PARAMETERS
"""
ZETA = timedelta(milliseconds=100)
MAX_DEV = 0.005
MIN_N = 10 # per camera

dataset_path = f"{os.environ['HOME']}/tud_datasets/placing_data_pkl_second"

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
    os |= {k: o_filtered}

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

    # TODO get rotation as polar coordinates starting from [0,0,-1]
    angle = np.dot(finalv, [0,0,-1])

    labels |= {
        t: {
            "quat": finalq,
            "vec": finalv,
            "angle": angle
        }
    }
