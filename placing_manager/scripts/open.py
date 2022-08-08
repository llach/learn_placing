#!/usr/bin/python
import rospy
import actionlib

from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

rname, lname = 'gripper_right_finger_joint', 'gripper_left_finger_joint'
JT_MAX = 0.045
JOINT_NAMES = [rname, lname]

if __name__ == "__main__":
    rospy.init_node("open_mm")

    mmKill = rospy.ServiceProxy("/myrmex_gripper_controller/kill", Empty)
    mmAC = actionlib.SimpleActionClient(f"/myrmex_gripper_controller/follow_joint_trajectory", FollowJointTrajectoryAction)

    print("waiting for myrmex controller server ...")
    mmAC.wait_for_server()

    print("waiting for myrmex kill service ...")
    mmKill.wait_for_service()

    mmKill()

    jt = JointTrajectory()
    jt.joint_names = JOINT_NAMES
    jt.points.append(JointTrajectoryPoint(positions=[JT_MAX, JT_MAX], time_from_start=rospy.Time(1.0)))

    mmAC.send_goal(FollowJointTrajectoryGoal(trajectory=jt))
    mmAC.wait_for_result()

