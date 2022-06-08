#!/usr/bin/python

import time
import rospy
import argparse

from sensor_msgs.msg import JointState
from wrist_motion.reorient import Reorient

parser = argparse.ArgumentParser()
parser.add_argument("--sim", default=False, action="store_true")

args, _ = parser.parse_known_args()

rospy.init_node("c")

if args.sim:
    print("sim setup")
    joint_pub = rospy.Publisher('/move_group/fake_controller_joint_states', JointState, queue_size=10)

    INITIAL_STATE = [0.0, 0.46, -0.48, -0.86, 1.74, -1.00, 0.29, 0.00]
    ACTIVE_JOINTS = ['torso_lift_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint']

    assert len(INITIAL_STATE)==len(ACTIVE_JOINTS)
    init_js = {jn: [jp] for jn, jp in zip(ACTIVE_JOINTS, INITIAL_STATE)}

    print("waiting for js subscriber")
    while joint_pub.get_num_connections()<1:
        time.sleep(0.5)
    joint_pub.publish(JointState(name=ACTIVE_JOINTS, position=INITIAL_STATE))

ro = Reorient()
try:
    while True:
        tr = ro.plan_random()
        inp = input("next?\n")
        if inp == "q":
            break
        elif inp == "e":
            ro.execute(tr)
except KeyboardInterrupt:
    print("bye")
