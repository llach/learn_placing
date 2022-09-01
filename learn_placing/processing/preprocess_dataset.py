import os
import pickle

import numpy as np
from datetime import timedelta
from learn_placing.common.label_processing import normalize, rotate_v

from learn_placing.common.vecplot import AxesPlot
from learn_placing.common.transformations import quaternion_conjugate, quaternion_from_matrix, quaternion_matrix, quaternion_multiply, quaternion_inverse, inverse_matrix, Ry
from learn_placing.common import load_dataset, cam_stats, qO2qdiff, v_from_qdiff, qavg, preprocess_myrmex, extract_gripper_T
from learn_placing.training.utils import InRot


""" PARAMETERS
"""
ZETA = timedelta(milliseconds=100)
MAX_DEV = 0.005
MIN_N = 10 # per camera
M = 50  # myrmex lookback TODO depends on dataset

dsnames = ["five"]
data_root = f"{os.environ['HOME']}/tud_datasets"
for dsname in dsnames: 
    print(f"processing dataset {dsname} ...")

    dataset_path = f"{data_root}/placing_data_pkl_{dsname}"
    dataset_file = f"{data_root}/{dsname}.pkl"

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
        Rwg = quaternion_matrix(qWG)

        # gripper -> object
        qGO = quaternion_multiply(quaternion_inverse(qWG), finalq)
        qGO = normalize(qGO)
        Rgo = quaternion_matrix(qGO)

        Zgo = Rgo@[0,0,1,1]
        ZgoNorm = normalize([Zgo[2], -Zgo[0]])
        gripper_angle = np.arctan2(ZgoNorm[1], ZgoNorm[0])

        Xgo = Rgo@[1,0,0,1]
        XgoNorm = normalize([Xgo[0], Xgo[2]])
        gripper_angle_x = np.arctan2(XgoNorm[1], XgoNorm[0])

        RcleanX = Ry(-gripper_angle_x)
        RcleanZ = Ry(-gripper_angle)

        RwCleanX = Rwg@RcleanX
        RwCleanZ = Rwg@RcleanZ

        Rgw = inverse_matrix(Rwg)

        Zgw = Rgw[:3,:3]@[0,0,1]
        Zgc = RcleanX@[0,0,1,1]

        local_dotp = np.arccos(np.dot(Zgw, Zgc[:3]))

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
                InRot.w2cleanX: quaternion_from_matrix(RwCleanX),
                InRot.w2cleanZ: quaternion_from_matrix(RwCleanZ),
                InRot.w2g: qWG,
                InRot.g2o: qGO,
                InRot.gripper_angle: gripper_angle,
                InRot.gripper_angle_x: gripper_angle_x,
                InRot.local_dotp: local_dotp,
                "vec": finalv,
                "angle": angle,
            }
        })

    inputs = {}
    static_inputs = {}
    for i, (t, sample) in enumerate(ds.items()):
        if t not in labels:
            print(f"skipping myrmex sample {i}")
            continue
        
        ler = preprocess_myrmex(sample["tactile_left"][1])
        rir = preprocess_myrmex(sample["tactile_right"][1])

        # cut to same length (we determined that in `myrmex_lookback.py`)
        ri = rir[-M:]
        le = ler[-M:]

        ri_static = rir[-2*M:-M]
        le_static = ler[-2*M:-M]

        inp = np.stack([le, ri])
        inp_static = np.stack([le_static, ri_static])

        inputs.update({t: inp})
        static_inputs.update({t: inp_static})

    with open(dataset_file, "wb") as f:
        pickle.dump({
            "labels": labels, 
            "inputs": inputs,
            "static_inputs": static_inputs
        }, f)
print("all done!")