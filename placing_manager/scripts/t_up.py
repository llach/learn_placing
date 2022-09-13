#!/usr/bin/python
import rospy
import actionlib

from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal


if __name__ == "__main__":
    rospy.init_node("yup")

   
    up_goal = FollowJointTrajectoryGoal()
    up_goal.trajectory.joint_names = ["torso_lift_joint"]
    up_goal.trajectory.points.append(JointTrajectoryPoint(positions=[0.35], time_from_start=rospy.Duration(1.0)))

    from cm_interface import safe_switch, is_running
    if not is_running("torso_stop_controller"):
        print("enabling torso controller ...")
        safe_switch("torso_controller", "torso_stop_controller")

    torsoAC = actionlib.SimpleActionClient("/torso_stop_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
    torsoAC.wait_for_server()

    print("moving torso ... ")
    torsoAC.send_goal_and_wait(up_goal)
