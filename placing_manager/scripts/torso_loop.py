import rospy
import actionlib

from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

from cm_interface import safe_switch

rospy.init_node("torso_loop")

ftCalib = rospy.ServiceProxy("/table_contact/calibrate", Empty)
torsoAC = actionlib.SimpleActionClient("/torso_stop_controller/follow_joint_trajectory", FollowJointTrajectoryAction)

print("enabling torso controller ...")
safe_switch("torso_controller", "torso_stop_controller")

print("waiting for ft calibration service ...")
ftCalib.wait_for_service()

print("waiting for torso action ... ")
torsoAC.wait_for_server()

print("setup done!")

start_pos = 0.15

up_goal = FollowJointTrajectoryGoal()
up_goal.trajectory.joint_names = ["torso_lift_joint"]
up_goal.trajectory.points.append(JointTrajectoryPoint(positions=[0.15], time_from_start=rospy.Duration(1.0)))

do_goal = FollowJointTrajectoryGoal()
do_goal.trajectory.joint_names = ["torso_lift_joint"]
do_goal.trajectory.points.append(JointTrajectoryPoint(positions=[0.10], time_from_start=rospy.Duration(2.0)))

torsoAC.send_goal_and_wait(up_goal)

while not rospy.is_shutdown():
    ftCalib()

    print("moving down")
    torsoAC.send_goal_and_wait(do_goal)

    print("moving up")
    torsoAC.send_goal_and_wait(up_goal)


