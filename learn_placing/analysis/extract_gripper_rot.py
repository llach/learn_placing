import os
import tf
import rospy

import numpy as np

from learn_placing.common import load_dataset,load_dataset_file, extract_gripper_T, normalize
from learn_placing.common.transformations import quaternion_from_matrix, quaternion_inverse, quaternion_multiply, make_T, quaternion_matrix, Rz, Rx

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

    xNorm = normalize([xO[0], xO[2]])
    zNorm = normalize([zO[2], -zO[0]])

    Zgo = Rgo@[0,0,1,1]
    ZgoNorm = normalize([Zgo[2], -Zgo[0]])
    gripper_angle = np.arctan2(ZgoNorm[1], ZgoNorm[0])
    print(gripper_angle)
    print(np.arctan2(xNorm[1], xNorm[0]), np.arctan2(zNorm[1], zNorm[0]))

    ytheta = np.arctan2(-yO[0], yO[1])
    Rgc1 = Rgo@Rz(-ytheta)
    print(ytheta)

    yC1 = Rgc1@[0,1,0,1]
    yphi = np.arctan2(yC1[2], yC1[1])
    Rgc2 = Rgc1@Rx(-yphi)
    print(yphi)

    xC = Rgc2@[1,0,0,1]
    yC = Rgc2@[0,1,0,1]
    zC = Rgc2@[0,0,1,1]

    print(np.arctan2(xC[2], xC[0]), np.arctan2(-zC[0], zC[2]))

    rospy.init_node("tf_rebroadcast")
    r = rospy.Rate(5)
    br = tf.TransformBroadcaster()
    while not rospy.is_shutdown():
        for t in tfs:
            statics()

            broadcast(
                [0,0,0],
                quaternion_from_matrix(Rgc2),
                "grasped_correct",
                "gripper_grasping_frame"
            )

            broadcast(
                [0,0,0],
                g2o[0],
                "grasped_new",
                "gripper_grasping_frame"
            )
            for tr in t:
                broadcast(
                    tr["translation"],
                    tr["rotation"],
                    tr["child_frame"],
                    tr["parent_frame"]
                )
        r.sleep()