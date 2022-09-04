import rospy
import actionlib
import numpy as np

from tf import TransformListener
from wrist_motion.marker import frame
from wrist_motion.reorient import Reorient
from visualization_msgs.msg import MarkerArray
from moveit_msgs.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal
from learn_placing.common.transformations import inverse_matrix, quaternion_inverse, quaternion_matrix, quaternion_multiply, quaternion_from_matrix, rotation_from_matrix, rotation_matrix

def Tf2T(pos, rot):
    T = quaternion_matrix(rot)
    T[:3,3] = pos
    return T

def RtoT(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

class PlacingPlanner:

    def __init__(self):
        self.grasping_frame = "gripper_left_grasping_frame"
        self.world_frame = "base_footprint"

        self.ro = Reorient()
        self.marker_pub = rospy.Publisher('/placing_markers', MarkerArray, queue_size=10)
        self.execute_ac = actionlib.SimpleActionClient("/execute_trajectory", ExecuteTrajectoryAction)

        self.li = TransformListener()
        for _ in range(6):
            try:
                self.li.waitForTransform(self.grasping_frame, self.world_frame, rospy.Time(0), rospy.Duration(3))
                break
            except Exception as e:
                print(e)
        print("planner init done")

    def input_or_quit(self, text):
        i = input(text)
        if i.lower() == "q" or i.lower() == "s":
            return False
        return True

    def plan_placing(self, Two):
        Tfwg = self.li.lookupTransform(self.world_frame, self.grasping_frame, rospy.Time(0))
        Twg = Tf2T(*Tfwg)
        Tgw = inverse_matrix(Twg)

        Tgo = quaternion_matrix(quaternion_multiply(quaternion_inverse(Tfwg[1]), quaternion_from_matrix(Two)))

        Tow = inverse_matrix(Two)

        u = [0,0,1]
        w = Tow[:3,:3]@[0,0,1]

        axis = np.cross(u,w)
        angle = np.arccos(np.dot(u,w))

        Toocorr = rotation_matrix(angle, axis)
        Tgocorr = Tgo@Toocorr

        start_frame = frame(Twg@Tgo, ns="start_frame")
        target_frame = frame(Twg@Tgocorr, ns="target_frame")

        ma = MarkerArray(markers=[
            *start_frame,
            *target_frame
        ])
        self.marker_pub.publish(ma)

        tr, failed = self.ro.plan_random(
            Tgo,
            Twg@Tgocorr,
            publish_traj=True, 
            check_validity=True,
            table_height=0.4, 
        )

        print("planning done")

        # input("execute?")

        # self.execute_ac.send_goal_and_wait(ExecuteTrajectoryGoal(trajectory=tr))
        
