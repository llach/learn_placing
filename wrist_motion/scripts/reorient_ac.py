#!/usr/bin/python
import rospy
import actionlib

from tf.transformations import quaternion_from_matrix

from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Quaternion, Transform, TransformStamped
from wrist_motion import Reorient
from wrist_motion.reorient import GRIPPER_FRAME
from wrist_motion.msg import PlanWristAction, PlanWristResult

def plan(goal): 
    global ro
    global _as
    print("generating new arm trajectory ...")
    failed = True
    while failed:
        tr, failed = ro.plan_random()

    oT = ro.get_target_joint_tf()
    pwr = PlanWristResult()
    pwr.trajectory = tr
    pwr.final_arm_state = ro.c.joint_msg
    pwr.object_transform = TransformStamped(
        header=Header(frame_id=GRIPPER_FRAME),
        child_frame_id="target",
        transform=Transform(
            translation=Vector3(*oT[0:3, 3]),
            rotation=Quaternion(*quaternion_from_matrix(oT))
        )
    )
    _as.set_succeeded(pwr)

rospy.init_node("wrist_motion")
ro = Reorient()

_as = actionlib.SimpleActionServer("/wrist_plan", PlanWristAction, execute_cb=plan, auto_start=False)
_as.start()

rospy.spin()