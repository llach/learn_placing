#!/usr/bin/python
import rospy
import actionlib
import numpy as np

from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from experiment_manager.common.cm_interface import safe_switch
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

ridx, lidx = 8, 7
rpos, lpos = None, None
rname, lname = 'gripper_right_finger_joint', 'gripper_left_finger_joint'

N_SECS = 1.2
N_POINTS = 5
JT_MIN = 0.0
JT_MAX = 0.045
JOINT_NAMES = [rname, lname]

def js_cb(msg):
    global ridx, lidx, rpos, lpos
    rpos, lpos = msg.position[ridx], msg.position[lidx]

def send_trajectory(to):
    global cpub, N_POINTS, N_SECS

    if rpos == None or lpos == None:
        print("we don't have poses yet")
        return
    
    lpoints = np.linspace(lpos, to, N_POINTS)
    rpoints = np.linspace(rpos, to, N_POINTS)
    times = np.linspace(0, N_SECS, N_POINTS)
    times[0] += 0.01

    jt = JointTrajectory()
    jt.joint_names = JOINT_NAMES

    for rp, lp, t in zip(rpoints, lpoints, times):
        jt.points.append(JointTrajectoryPoint(positions=[rp, lp], time_from_start=rospy.Time(t)))

    res = cpub.send_goal_and_wait(FollowJointTrajectoryGoal(trajectory=jt))
    print(f"result {res}")

def open_gripper(): send_trajectory(JT_MAX)
def close_gripper(): send_trajectory(JT_MIN)

rospy.init_node("oc")
rospy.Subscriber("/joint_states", JointState, js_cb)
kill_service = rospy.ServiceProxy("/myrmex_gripper_controller/kill", Empty)

USE_MM = True
_pre = ""
if USE_MM:
    _pre = "myrmex_"
    safe_switch("gripper_controller", "myrmex_gripper_controller")

cpub = actionlib.SimpleActionClient(f"/{_pre}gripper_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
cpub.wait_for_server()

try:
    while True:
        inp = input("what next?\n")
        if inp == "q":
            break
        elif inp == "o":
            open_gripper()
        elif inp == "c":
            close_gripper()
except KeyboardInterrupt:
    pass
print("bye")