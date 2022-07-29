import rospy
import actionlib

from std_srvs.srv import Empty, Trigger
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

from cm_interface import safe_switch

rospy.init_node("grasp_loop")

mmKill = rospy.ServiceProxy("/myrmex_gripper_controller/kill", Empty)
mmCalib = rospy.ServiceProxy("/myrmex_gripper_controller/calibrate", Trigger)
mmAC = actionlib.SimpleActionClient(f"/myrmex_gripper_controller/follow_joint_trajectory", FollowJointTrajectoryAction)

print("enabling myrmex controller ...")
safe_switch("gripper_controller", "myrmex_gripper_controller")

print("waiting for myrmex controller server ...")
mmAC.wait_for_server()

print("waiting for myrmex calibration service ...")
mmCalib.wait_for_service()

print("waiting for kill service ...")
mmKill.wait_for_service()

open_goal = FollowJointTrajectoryGoal()
open_goal.trajectory.joint_names = [ 'gripper_right_finger_joint', 'gripper_left_finger_joint']
open_goal.trajectory.points.append(JointTrajectoryPoint(positions=2*[0.043], time_from_start=rospy.Duration(1.0)))

close_goal = FollowJointTrajectoryGoal()
close_goal.trajectory.joint_names = [ 'gripper_right_finger_joint', 'gripper_left_finger_joint']
close_goal.trajectory.points.append(JointTrajectoryPoint(positions=2*[0.0], time_from_start=rospy.Duration(2.0)))

print("opening gripper")
mmAC.send_goal_and_wait(open_goal)

print("calibrating myrmex ...")
mmKill()
res = mmCalib()
if not res.success:
    print("MYRMEX CALIBRATION FAILED!")
    exit(-1)

while not rospy.is_shutdown():
    print("opening gripper")
    mmAC.send_goal_and_wait(open_goal)

    print("closing gripper")
    mmAC.send_goal_and_wait(close_goal)

    input("next?")