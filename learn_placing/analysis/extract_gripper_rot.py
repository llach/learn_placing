import os
import tf
import rospy

import numpy as np

from learn_placing.common import load_dataset,load_dataset_file, extract_gripper_T
from learn_placing.common.transformations import quaternion_from_matrix, quaternion_inverse, quaternion_multiply, make_T

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

rospy.init_node("tf_rebroadcast")
r = rospy.Rate(5)
br = tf.TransformBroadcaster()
for i, (stamp, label) in enumerate(os["labels"].items()):
    if i != 2: continue
    tfs = ds[stamp]["tf"][1]

    footprint2object = []
    gripper2object = []
    T, world2obj, grip2obj = extract_gripper_T(tfs)

    for t in tfs:
        for tr in t:
            if tr["child_frame"] == "object" and tr["parent_frame"] == "base_footprint":
                footprint2object.append(tr["rotation"])
            elif tr["child_frame"] == "grasped_object" and tr["parent_frame"] == "gripper_grasping_frame":
                gripper2object.append(tr["rotation"])

    print(T[:3,3])
    print(quaternion_from_matrix(T))

    qG = quaternion_multiply(quaternion_inverse(quaternion_from_matrix(T)), footprint2object[0])

    while not rospy.is_shutdown():
        for t in tfs:
            statics()

            broadcast(
                [0,0,0],
                qG,
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