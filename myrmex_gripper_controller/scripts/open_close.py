#!/usr/bin/env python
import time
import rospy
import actionlib
import numpy as np

from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from pal_common_msgs.msg import EmptyAction, EmptyActionGoal
# from experiment_manager.common.cm_interface import safe_switch
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

ridx, lidx = 8, 7
tidx = 19
tpos = None
rpos, lpos = None, None
rname, lname = 'gripper_right_finger_joint', 'gripper_left_finger_joint'

N_SECS = 1.2
N_POINTS = 5
JT_MIN = 0.0
JT_MAX = 0.045
JOINT_NAMES = [rname, lname]

def activate_torso():
    global torsoAC

    from cm_interface import safe_switch, is_running
    if not is_running("torso_stop_controller"):
        print("enabling torso controller ...")
        safe_switch("torso_controller", "torso_stop_controller")
        torsoAC = actionlib.SimpleActionClient("/torso_stop_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        print("waiting for torso action ... ")
        torsoAC.wait_for_server()
    else:
        print("torso stop controller running!")


def input_or_quit(text):
    i = input(text)
    if i.lower() == "q": 
        return False
    elif i.lower() == "s":
        return False
    return True


def js_cb(msg):
    global ridx, lidx, rpos, lpos, tidx, tpos
    rpos, lpos = msg.position[ridx], msg.position[lidx]
    tpos = msg.position[tidx]

def send_trajectory(to):
    global mmAC

    jt = JointTrajectory()
    jt.joint_names = JOINT_NAMES
    jt.points.append(JointTrajectoryPoint(positions=[to, to], time_from_start=rospy.Time(1.5)))

    res = cpub.send_goal(FollowJointTrajectoryGoal(trajectory=jt))
    print(f"result {res}")

def open_gripper(): send_trajectory(JT_MAX)
def close_gripper(): send_trajectory(JT_MIN)

def set_torso(pos=None, secs=7, wait=True):
    global torsoAC, tpos

    activate_torso()
    
    if pos == None: pos = tpos 

    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ["torso_lift_joint"]
    goal.trajectory.points.append(JointTrajectoryPoint(positions=[pos], time_from_start=rospy.Duration(secs)))
    torsoAC.send_goal(goal)
    if wait: torsoAC.wait_for_result()

rospy.init_node("oc")
rospy.Subscriber("/joint_states", JointState, js_cb)
kill_service = rospy.ServiceProxy("/myrmex_gripper_controller/kill", Empty)
calib_service = rospy.ServiceProxy("/myrmex_gripper_controller/calibrate", Empty)

cpub = actionlib.SimpleActionClient(f"/myrmex_gripper_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
cpub.wait_for_server()

ftCalib = rospy.ServiceProxy("/table_contact/calibrate", Empty)
torsoAC = actionlib.SimpleActionClient("/torso_stop_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
torsoAC.wait_for_server()

gravityAC = actionlib.SimpleActionClient("/gravity_compensation", EmptyAction)
gravityAC.wait_for_server()

try:
    while True:
        inp = input("o = open; c = close; k = kill myrmex; cal = calibrate; u = torso up; d = torso down\n")
        if inp == "q":
            break
        elif inp == "o":
            open_gripper()
        elif inp == "c":
            close_gripper()
        elif inp == "k":
            kill_service()
        elif inp == "cal":
            ftCalib()
        elif inp== "u":
            print("torso up")
            ftCalib()

            set_torso(0.35, secs=3)
            print("done")

        elif inp== "d":
            print("torso down")
            ftCalib()
            set_torso(0.0)
            set_torso(wait=False)

        elif inp=="g":
            arm_goal = EmptyActionGoal()
            gravityAC.send_goal(arm_goal)

            input_or_quit("done?")
            gravityAC.cancel_all_goals()

except KeyboardInterrupt:
    pass
print("bye")