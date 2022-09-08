import os
import pickle

import numpy as np
from datetime import timedelta
from learn_placing.common.label_processing import normalize, rotate_v

from learn_placing.common.vecplot import AxesPlot
from learn_placing.common.transformations import quaternion_conjugate, quaternion_from_matrix, quaternion_matrix, quaternion_multiply, quaternion_inverse, inverse_matrix, Ry
from learn_placing.common import load_dataset, cam_stats, qO2qdiff, v_from_qdiff, qavg, preprocess_myrmex, extract_gripper_T
from learn_placing.training.utils import DatasetName, InRot, ds2name


# we switched to longer record times around the detected touch, so different datasets have different timestamps
dsLookback = {
    DatasetName.cuboid: [[-50,None], [-100,-50]],
    DatasetName.cylinder: [[-50,None], [-100,-50]],
    DatasetName.object_var: [[-80,-30], [-130,-80]],
    DatasetName.object_var2: [[-80,-30], [-130,-80]],
    DatasetName.gripper_var: [[-80,-30], [-130,-80]],
    DatasetName.gripper_var2: [[-80,-30], [-130,-80]],
    DatasetName.combined_var2: [[-80,-30], [-130,-80]],
    DatasetName.opti_gripper_test: [[-80,-20], [-120,-80]],
    DatasetName.test: [[-80,-40], [-120,-80]],
}

ftLookback = {
    DatasetName.cuboid: [[-15,None], [-30,-15]],
    DatasetName.cylinder: [[-15,None], [-30,-15]],
    DatasetName.object_var: [[-20,-5], [-35,-20]],
    DatasetName.object_var2: [[-20,-5], [-35,-20]],
    DatasetName.gripper_var: [[-20,-5], [-35,-20]],
    DatasetName.gripper_var2: [[-20,-5], [-35,-20]],
    DatasetName.combined_var2: [[-20,-5], [-35,-20]],
    DatasetName.opti_gripper_test: [[-20,-5], [-35,-20]],
    DatasetName.test: [[-20,-5], [-35,-20]],
}

def myrmex_transform(left, right, dd):
    ler = preprocess_myrmex(left)
    rir = preprocess_myrmex(right)

    # cut to same length (we determined that in `myrmex_lookback.py`)
    fro, to = dsLookback[dd][0]
    ri = rir[fro:to]
    le = ler[fro:to]

    sfro, sto = dsLookback[dd][1]
    ri_static = rir[sfro:sto]
    le_static = ler[sfro:sto]

    inp = np.stack([le, ri])
    print(le_static.shape, ri_static.shape)
    inp_static = np.stack([le_static, ri_static])

    return inp, inp_static

def ft_transform(ft, dd):
    data_ft = np.reshape(ft[-35:], (35,6))
            
    fro, to = ftLookback[dd][0]
    ftt = data_ft[fro:to]

    sfro, sto = ftLookback[dd][1]
    ft_static = data_ft[sfro:sto]

    return ftt, ft_static

if __name__ == "__main__":
    """ PARAMETERS
    """
    basef = "base_footprint"
    gripf = "gripper_left_grasping_frame"
    objf = "pot"

    ZETA = timedelta(milliseconds=100)
    MAX_DEV = 0.005
    MIN_N = 10 # per camera

    dsnames = [DatasetName.opti_gripper_test]
    data_root = f"{os.environ['HOME']}/tud_datasets"
    for dd in dsnames: 
        dsname = ds2name[dd]
        print(f"processing dataset {dsname} ...")

        dataset_path = f"{data_root}/placing_data_pkl_opti_test"
        dataset_file = f"{data_root}/opti_test.pkl"

        # sample timestamp -> sample
        ds = load_dataset(dataset_path)

        # step 1: filter object state samples at or after contact
        ### NOTE this will only be excuted for camera-setup-based dataset
        os = {}
        labels = {}
        if "opti_state" in list(ds.values())[0]:
            """
            OPTITRACK PREPROCESSING
            """
            for k, v in ds.items():
                opti = v["opti_state"][1]

                # opti is the sequence of opti state messages during one sample collection
                Qwg, Qwo, Qgo = None, None, None

                # we just take the first frame here, trusting optitrac to be stable enough
                # to not need averaging over samples
                for op in opti[0]:
                    if op["parent_frame"] == basef and op["child_frame"] == gripf:
                        Qwg = op["rotation"]
                    elif op["parent_frame"] == basef and op["child_frame"] == objf:
                        Qwo = op["rotation"]
                    elif op["parent_frame"] == gripf and op["child_frame"] == objf:
                        Qgo = op["rotation"]
                    else: 
                        print(f"unknown transform: {op['parent_frame']} to {op['child_frame']}")
                if np.any([q is None for q in[Qwg, Qwo, Qgo]]): print("missing transform ...")
                labels.update({
                    k: {
                        InRot.w2g: Qwg,
                        InRot.w2o: Qwo,
                        InRot.g2o: Qwg,
                    }
                })
        elif "object_state" in list(ds.values())[0]:
            """
            OLD CAMERA SETUP PREPROCESSING
            """
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

            # step 2: store cleaned object samples
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
        
        # step 3: store myrmex input samples, skipping bad ones
        inputs = {}
        static_inputs = {}
        for i, (t, sample) in enumerate(ds.items()):
            if t not in labels:
                print(f"skipping myrmex sample {i}")
                continue

            inp, inp_static = myrmex_transform(sample["tactile_left"][1], sample["tactile_right"][1], dd)

            inputs.update({t: inp})
            static_inputs.update({t: inp_static})

        # step 4: preprocess FT data
        ft = {}
        static_ft = {}
        for i, (t, sample) in enumerate(ds.items()):
            if t not in labels:
                print(f"skipping FT sample {i}")
                continue

            ftt, ft_static = ft_transform(sample["ft"][1], dd)
            
            ft.update({t: ftt})
            static_ft.update({t: ft_static})

        with open(dataset_file, "wb") as f:
            pickle.dump({
                "labels": labels, 
                "inputs": inputs,
                "static_inputs": static_inputs,
                "ft": ft,
                "static_ft": static_ft
            }, f)
        print()
    print("all done!")