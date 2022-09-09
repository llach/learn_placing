#!/usr/bin/python
from pickletools import optimize
import time
import rospy
import rosnode
import actionlib
import numpy as np

from tf2_msgs.msg import TFMessage
from std_srvs.srv import Empty, Trigger
from sensor_msgs.msg import JointState
from pal_common_msgs.msg import EmptyAction, EmptyActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

from cm_interface import safe_switch

def safe_exit():
    global mmAC, gravityAC
    for ac in [mmAC, gravityAC]:
        ac.cancel_all_goals()
    print("bye")
    exit(0)

def input_or_quit(text):
    i = input(text)
    if i.lower() == "q": 
        safe_exit()
    elif i.lower() == "s":
        return False
    return True


ridx, lidx = 8, 7
rpos, lpos = None, None


rname, lname = 'gripper_left_right_finger_joint', 'gripper_left_left_finger_joint'

N_SECS = 1.2
N_POINTS = 5
JT_MIN = 0.0
JT_MAX = 0.045
JOINT_NAMES = [rname, lname]


def send_trajectory(to):
    global mmAC

    jt = JointTrajectory()
    jt.joint_names = JOINT_NAMES
    jt.points.append(JointTrajectoryPoint(positions=[to, to], time_from_start=rospy.Time(1.5)))

    mmAC.send_goal(FollowJointTrajectoryGoal(trajectory=jt))

def open_gripper(): send_trajectory(JT_MAX)
def close_gripper(): send_trajectory(JT_MIN)

optipub = False
if __name__ == "__main__":
    rospy.init_node("setup_experiment")


    ################################################
    ############## ROS COMMUNICATION ###############
    ################################################

    def cb(_):
        global optipub
        optipub=True

    osSub = rospy.Subscriber('/opti_state', TFMessage, callback=cb)

    mmKill = rospy.ServiceProxy("/myrmex_gripper_controller/kill", Empty)
    mmCalib = rospy.ServiceProxy("/myrmex_gripper_controller/calibrate", Trigger)
    # objectStateCalib = rospy.ServiceProxy("/object_state_calibration", Empty)
    ftCalib = rospy.ServiceProxy("/table_contact/calibrate", Empty)
    reinitSrv = rospy.ServiceProxy("/reinit_wrist_planner", Empty)

    gravityAC = actionlib.SimpleActionClient("/gravity_compensation", EmptyAction)
    mmAC = actionlib.SimpleActionClient(f"/myrmex_gripper_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
    torsoAC = actionlib.SimpleActionClient("/torso_stop_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
    headAC = actionlib.SimpleActionClient("/head_controller/follow_joint_trajectory", FollowJointTrajectoryAction)

    up_goal = FollowJointTrajectoryGoal()
    up_goal.trajectory.joint_names = ["torso_lift_joint"]
    up_goal.trajectory.points.append(JointTrajectoryPoint(positions=[0.35], time_from_start=rospy.Duration(3.0)))

    head_goal = FollowJointTrajectoryGoal()
    head_goal.trajectory.joint_names = ["head_1_joint", "head_2_joint"]
    head_goal.trajectory.points.append(JointTrajectoryPoint(positions=[0.0, -0.65], time_from_start=rospy.Duration(1.0)))

    print("enabling torso controller ...")
    safe_switch("torso_controller", "torso_stop_controller")

    print("enabling myrmex controller ...")
    safe_switch("gripper_left_controller", "myrmex_gripper_controller")

    print("waiting for torso action ... ")
    torsoAC.wait_for_server()

    print("waiting for head action ... ")
    headAC.wait_for_server()

    print("moving torso ... ")
    torsoAC.send_goal_and_wait(up_goal)

    print("killing head manager ...")
    rosnode.kill_nodes(["/pal_head_manager", "/aruco_single", "/gripper_right_grasping", "/xtion_node_doctor"])

    print("moving head ...")
    headAC.send_goal_and_wait(head_goal)

    print("waiting for myrmex controller server ...")
    mmAC.wait_for_server()

    print("waiting for gravity compensation server ...")
    gravityAC.wait_for_server()

    # print("waiting for object state estimation calibration service ...")
    # objectStateCalib.wait_for_service()

    print("waiting for myrmex kill service ...")
    mmKill.wait_for_service()

    print("waiting for myrmex calibration service ...")
    mmCalib.wait_for_service()

    print("waiting for ft calibration service ...")
    ftCalib.wait_for_service()

    # print("waiting for wrist planner reinit service ...")
    # reinitSrv.wait_for_service()

    print("################ setup done")

    print("checking optitrack ...")
    start = rospy.Time.now()
    while not optipub:
        rospy.Rate(20).sleep()
        if rospy.Time.now()-start > rospy.Duration(5.0):
            print("ERROR optitrack doesn't seem to be published ...")
            exit(-1)

    print("optitrack is being published!")
    
    ################################################
    ################ TAG CALIBRATION ###############
    ################################################

    # if input_or_quit("object calib?"): objectStateCalib()

    ################################################
    ################ REPOSITION ARM ################
    ################################################


    if input_or_quit("gravity off?"):
        arm_goal = EmptyActionGoal()
        gravityAC.send_goal(arm_goal)

        input_or_quit("done?")
        gravityAC.cancel_all_goals()

        # print("reiniting reorient")
        # reinitSrv()

    ################################################
    ########### SETUP MYRMEX CONTROLLER ############
    ################################################

    if input_or_quit("myrmex setup?"):

        print("opening gripper")
        open_gripper()
        mmAC.wait_for_result()

        print("calibrating myrmex ...")
        res = mmCalib()
        if not res.success:
            print("MYRMEX CALIBRATION FAILED!")
            exit(-1)

        input_or_quit("close?")
        close_gripper()
        # print("waiting 3sec to settle...")
        # time.sleep(3)
        # mmKill()

    ################################################
    ################ FT CALIBRATION ################
    ################################################
    if input_or_quit("ft calibration?"): ftCalib()

    print("all done, bye!")