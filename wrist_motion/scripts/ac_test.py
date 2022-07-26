#!/usr/bin/python
import rospy
import actionlib

from moveit_msgs.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal
from wrist_motion.msg import PlanWristAction, PlanWristGoal



rospy.init_node("wrist_motion_test")

wac = actionlib.SimpleActionClient("/wrist_plan", PlanWristAction)
eac = actionlib.SimpleActionClient("/execute_trajectory", ExecuteTrajectoryAction)

print("waiting for wac")
wac.wait_for_server()

print("waiting for eac")
eac.wait_for_server()


pwg = PlanWristGoal()
wac.send_goal_and_wait(pwg)
pwr = wac.get_result()

etg = ExecuteTrajectoryGoal()
etg.trajectory = pwr.trajectory
eac.send_goal_and_wait(etg)

print(eac.get_result())