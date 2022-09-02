import rospy
import actionlib
import numpy as np

from tf import TransformListener
from wrist_motion.marker import frame
from wrist_motion.reorient import Reorient
from visualization_msgs.msg import MarkerArray
from moveit_msgs.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal
from learn_placing.common.transformations import inverse_matrix, quaternion_matrix

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

    def plan_placing(self, Tdiff):
        Tfg = self.li.lookupTransform(self.world_frame, self.grasping_frame, rospy.Time(0))
        Tg = Tf2T(*Tfg)

        Tdiffinv = inverse_matrix(Tdiff)

        Rgoal = Tg[:3,:3]@Tdiffinv[:3,:3]
        Tgoal = RtoT(Rgoal, Tfg[0])

        target_frame = frame(Tgoal, ns="target_frame")

        ma = MarkerArray(markers=[
            *target_frame
        ])
        self.marker_pub.publish(ma)

        tr, failed = self.ro.plan_random(
            publish_traj=True, 
            check_validity=True,
            table_height=0.4, 
            To=Tgoal
        )

        # input("execute?")

        # self.execute_ac.send_goal_and_wait(ExecuteTrajectoryGoal(trajectory=tr))
        
