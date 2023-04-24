import os
import sys
import time
import rospy
import pickle

import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from tactile_msgs.msg import TactileState
from geometry_msgs.msg import WrenchStamped

from tf import TransformListener
from datetime import datetime
from learn_placing import dataset_path, datefmt
from learn_placing.common import v2l, preprocess_myrmex


class SequenceCollector:
    world_frame = "base_footprint"
    grasping_frame = "gripper_grasping_frame"
    left_finger_name = "gripper_left_finger_joint"
    right_finger_name = "gripper_right_finger_joint"
    mm_left, mm_right, ft, gripper_pose, js = {}, {}, {}, {}, {}
    mli, mri = 0, 0

    def __init__(self, save_path, name, mmskip=20):
        self.mmskip = mmskip
        self.name = name
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.ftsub = rospy.Subscriber("/wrist_ft", WrenchStamped, callback=self.ft_cb)
        self.tlsub = rospy.Subscriber("/tactile_left",  TactileState, callback=self.tl_cb)
        self.trsub = rospy.Subscriber("/tactile_right", TactileState, callback=self.tr_cb)
        self.jssub = rospy.Subscriber("/joint_states", JointState, callback=self.js_cb)

        self.li = TransformListener()
        self.li.waitForTransform(self.world_frame, self.grasping_frame, rospy.Time(0), rospy.Duration(5))
        time.sleep(0.5)
        self.start = datetime.now()
        
    def tl_cb(self, m): 
        self.mli += 1
        if self.mli % self.mmskip != 0: return
        
        self.mm_left.update({time.time(): preprocess_myrmex(m.sensors[0].values)})

    def tr_cb(self, m): 
        self.mri += 1
        if self.mri % self.mmskip != 0: return
    
        self.mm_right.update({time.time(): preprocess_myrmex(m.sensors[0].values)})

    def ft_cb(self, m): self.ft.update({time.time(): np.concatenate([v2l(m.wrench.force), v2l(m.wrench.torque)])})

    def js_cb(self, m): 
        try:
            self.gripper_pose.update({time.time(): self.li.lookupTransform(self.world_frame, self.grasping_frame,    rospy.Time())})
        except: print("couldn't get gripper pose")

        self.js.update({time.time(): {self.right_finger_name: m.position[m.name.index(self.right_finger_name)], self.left_finger_name: m.position[m.name.index(self.left_finger_name)]}})

    def finalize(self):
        for sub in [self.ftsub, self.tlsub, self.trsub, self.jssub]: sub.unregister()

        print(f"saving {self.name} - recording time {datetime.now()-self.start}")
        print(f"    Myrmex Left  {len(self.mm_left)}")
        print(f"    Myrmex Right {len(self.mm_right)}")
        print(f"    Joint States {len(self.js)}")
        print(f"    F/T Data     {len(self.ft)}")
        print(f"    Gripper Pos  {len(self.gripper_pose)}")

        with open(f"{self.save_path}/seq_{self.name}_{datetime.now().strftime(datefmt)}.pkl", "wb") as f:
            pickle.dump({
                "mm_left": self.mm_left,
                "mm_right": self.mm_right,
                "joint_states": self.js,
                "ft": self.ft,
                "gripper_pose": self.gripper_pose,
            }, f)
        print("done.")

if __name__ == "__main__":
    assert len(sys.argv)==2, "sample name missing"

    rospy.init_node("collect_sequence_data")
    save_path = f"{dataset_path}/sequences/"

    print(f"saving data in {save_path}")
    sc = SequenceCollector(save_path, sys.argv[1])

    while not rospy.is_shutdown(): rospy.Rate(10).sleep()
    sc.finalize()