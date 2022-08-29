import os
import tf
import rospy

import numpy as np

from learn_placing.common import load_dataset,load_dataset_file, extract_gripper_T, normalize
from learn_placing.common.transformations import euler_from_matrix, euler_from_quaternion, inverse_matrix, quaternion_from_matrix, quaternion_inverse, quaternion_multiply, make_T, quaternion_matrix, Rz, Rx, Ry

def broadcast(trans, rot, target, source):
    global br
    br.sendTransform(
        trans,
        rot, 
        rospy.Time.now(),
        target,
        source
    )

def statics():
    broadcast(
        [0.000, 0.000, 0.099],
        [0.000, 0.000, 0.000, 1.000],
        "base_link",
        "base_footprint"
    )
    broadcast(
        [-0.062, 0.000, 0.193],
        [0.000, 0.000, 0.000, 1.000],
        "torso_fixed_link",
        "base_link"
    )
    broadcast(
        [-0.000, 0.000, 0.077],
        [-0.707, 0.707, -0.000, -0.000],
        "gripper_link",
        "arm_7_link"
    )
    broadcast(
        [0.000, 0.000, -0.120],
        [-0.500, 0.500, 0.500, 0.500],
        "gripper_grasping_frame",
        "gripper_link"
    )

data_root = f"{os.environ['HOME']}/tud_datasets"
dataset_path = f"{data_root}/placing_data_pkl_second"
dataset_file = f"{data_root}/second.pkl"
# dataset_path = f"{data_root}/placing_data_pkl_third"
# dataset_file = f"{data_root}/third.pkl"

ds = load_dataset(dataset_path)
os = load_dataset_file(dataset_file)

for i, (stamp, label) in enumerate(os["labels"].items()):
    if i != 2: continue
    tfs = ds[stamp]["tf"][1]

    T, w2o, g2o = extract_gripper_T(tfs)
    Rwo = quaternion_matrix(w2o[0])
    Rgo = quaternion_matrix(g2o[0])
    Rwg = T

    xO = Rgo@[1,0,0,1]
    yO = Rgo@[0,1,0,1]
    zO = Rgo@[0,0,1,1]

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

    # eulers = euler_from_matrix(Rgo, "syxz")

    # xNorm = normalize([xO[0], xO[2]])
    # zNorm = normalize([zO[2], -zO[0]])

    # print(gripper_angle)
    # print(np.arctan2(xNorm[1], xNorm[0]), np.arctan2(zNorm[1], zNorm[0]), np.arctan2(-xO[1], xO[0]))
    

    # Xog = inverse_matrix(Rgo)@[1,0,0,1]
    # X_off_O = np.arctan2(Xog[1], Xog[0])
    # Rexp1 = Rgo@Rz(X_off_O)

    # Zexp1g = inverse_matrix(Rexp1)@[0,0,1,1]
    # Z_off_exp1 = np.arctan2(-Zexp1g[1], Zexp1g[2])
    # Rexp2 = Rexp1@Rx(Z_off_exp1)


    # ytheta = np.arctan2(-yO[0], yO[1])
    # Rgc1 = Rgo@Rz(-ytheta)
    # print(ytheta)

    # yC1 = Rgc1@[0,1,0,1]
    # yphi = np.arctan2(yC1[2], yC1[1])
    # Rgc2 = Rgc1@Rx(-yphi)
    # print(yphi)

    # xC = Rgc2@[1,0,0,1]
    # yC = Rgc2@[0,1,0,1]
    # zC = Rgc2@[0,0,1,1]

    # print(np.arctan2(xC[2], xC[0]), np.arctan2(-zC[0], zC[2]))

    # xExp = Rexp2@[1,0,0,1]
    # yExp = Rexp2@[0,1,0,1]
    # zExp = Rexp2@[0,0,1,1]

    # print(np.arctan2(xExp[2], xExp[0]), np.arctan2(-zExp[0], zExp[2]))


    Rgw = inverse_matrix(T)

    rospy.init_node("tf_rebroadcast")
    r = rospy.Rate(5)
    br = tf.TransformBroadcaster()
    while not rospy.is_shutdown():
        for t in tfs:
            statics()

            broadcast(
                [0,0,0],
                quaternion_from_matrix(Rgw),
                "gripper_world",
                "gripper_grasping_frame"
            )

            # broadcast(
            #     [0,0,0],
            #     quaternion_from_matrix(Rexp2),
            #     "gripper_experimental",
            #     "gripper_grasping_frame"
            # )

            # broadcast(
            #     [0,0,0],
            #     quaternion_from_matrix(Rgc2),
            #     "grasped_correct",
            #     "gripper_grasping_frame"
            # )

            broadcast(
                [0,0,0],
                quaternion_from_matrix(RcleanX),
                "grasped_clean_X",
                "gripper_grasping_frame"
            )

            broadcast(
                [0,0,0],
                quaternion_from_matrix(RcleanZ),
                "grasped_clean_Z",
                "gripper_grasping_frame"
            )

            broadcast(
                [0,0,0],
                quaternion_from_matrix(RwCleanX),
                "world_clean_X",
                "base_footprint"
            )

            broadcast(
                [0,0,0],
                quaternion_from_matrix(RwCleanZ),
                "world_clean_Z",
                "base_footprint"
            )
            for tr in t:
                broadcast(
                    tr["translation"],
                    tr["rotation"],
                    tr["child_frame"],
                    tr["parent_frame"]
                )
        r.sleep()