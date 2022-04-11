import sys
import rospy
import actionlib
import numpy as np

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal 

from cm_interface import load_controller, stop_controller, start_controller

TORSO_JOINT_NAME = 'torso_lift_joint'
TORSO_IDX = 11

torso_pos = None

def generate_trajectory(first, last, total_time=5, num_points=5):

    jt = JointTrajectory()
    jt.header.frame_id = 'base_footprint'
    jt.joint_names = ['torso_lift_joint']

    pts = []
    for t, j in zip(np.linspace(0, total_time, num_points), np.linspace(first, last, num_points)):
        jp = JointTrajectoryPoint()
        jp.positions = [j]

        if t == 0.0: t += 0.1
        tm = rospy.Time(t)
        jp.time_from_start.secs = tm.secs
        jp.time_from_start.nsecs = tm.nsecs

        pts.append(jp)

    jt.points = pts
    return jt

def joint_states_cb(m):
    global torso_pos
    global TORSO_IDX

    torso_pos = np.round(m.position[TORSO_IDX], 4)

if len(sys.argv)<2:
    print("goal missing. using 0.15")
    q_goal = 0.15
else:
    q_goal = float(sys.argv[1])
print(f"moving torso to {q_goal}")

rospy.init_node("torso_movement")

js_sub = rospy.Subscriber("/joint_states", JointState, joint_states_cb, queue_size=1)
c = actionlib.SimpleActionClient("/torso_stop_controller/follow_joint_trajectory", FollowJointTrajectoryAction)

# this fails if torso_stop_controller is already running, but that's ok
load_controller("torso_stop_controller")
stop_controller("torso_controller")
start_controller("torso_stop_controller")

print("waiting for action server ...")
if c.wait_for_server(timeout=rospy.Duration(3)):
    print("found it!")
else:
    print("ac server not found ...")

while torso_pos == None:
    pass

print(f"moving torso from {torso_pos} to {q_goal}")

traj = generate_trajectory(first=torso_pos, last=q_goal)
g = FollowJointTrajectoryGoal()
g.trajectory = traj

res = c.send_goal_and_wait(g)
print(res)