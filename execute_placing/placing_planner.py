import time
import rospy
import actionlib
import numpy as np

from tf import TransformListener
from wrist_motion.marker import frame
from wrist_motion.reorient import Reorient
from visualization_msgs.msg import MarkerArray
from moveit_msgs.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal
from learn_placing.common.transformations import inverse_matrix, quaternion_inverse, quaternion_matrix, quaternion_multiply, quaternion_from_matrix, rotation_matrix

from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState

from cm_interface import safe_switch, is_running

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
    JT_MAX = 0.045
    TO_MAX = 0.35

    def __init__(self):
        self.table_height = 0.72
        self.torso_vel = 0.35 # sec/cm
        self.grasping_frame = "gripper_left_grasping_frame"
        self.world_frame = "base_footprint"
        
        self.t_current = None

        self.ro = Reorient()
        self.troso_state_sub = rospy.Subscriber("/torso_stop_controller/state", JointTrajectoryControllerState, self.tcb, queue_size=1)
        self.marker_pub = rospy.Publisher("/placing_markers", MarkerArray, queue_size=10)
        self.execute_ac = actionlib.SimpleActionClient("/execute_trajectory", ExecuteTrajectoryAction)

        self.ftCalib = rospy.ServiceProxy("/table_contact/calibrate", Empty)
        
        self.mmAC = actionlib.SimpleActionClient(f"/myrmex_gripper_controller/follow_joint_trajectory", FollowJointTrajectoryAction)

        self.torsoAC = actionlib.SimpleActionClient("/torso_stop_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        print("waiting for torso action ... ")
        self.torsoAC.wait_for_server()
        self.activate_torso()

        self.mmKill = rospy.ServiceProxy("/myrmex_gripper_controller/kill", Empty)

        print("enabling torso controller ...")
        safe_switch("torso_controller", "torso_stop_controller")

        print("enabling myrmex controller ...")
        safe_switch("gripper_left_controller", "myrmex_gripper_controller")

        print("waiting for myrmex controller server ...")
        self.mmAC.wait_for_server()

        print("waiting for myrmex kill service ...")
        self.mmKill.wait_for_service()

        print("waiting for ft calibration service ...")
        self.ftCalib.wait_for_service()

        self.li = TransformListener()
        for _ in range(6):
            try:
                self.li.waitForTransform(self.grasping_frame, self.world_frame, rospy.Time(0), rospy.Duration(3))
                break
            except Exception as e:
                print(e)
        print("planner init done")

    def tcb(self, msg): self.t_current = msg.actual.positions[0]

    def activate_torso(self):
        
        if not is_running("torso_stop_controller"):
            print("enabling torso controller ...")
            safe_switch("torso_controller", "torso_stop_controller")
            self.torsoAC = actionlib.SimpleActionClient("/torso_stop_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
            print("waiting for torso action ... ")
            self.torsoAC.wait_for_server()
        else:
            print("torso stop controller running!")

    def open_gripper(self): self.gripper_traj(self.JT_MAX)
    def gripper_traj(self, to):
        jt = JointTrajectory()
        jt.joint_names = ['gripper_left_right_finger_joint', 'gripper_left_left_finger_joint']
        jt.points.append(JointTrajectoryPoint(positions=[to, to], time_from_start=rospy.Time(1.5)))

        self.mmAC.send_goal(FollowJointTrajectoryGoal(trajectory=jt))
        self.mmAC.wait_for_result()

    def torso_up(self): self.torso_traj(self.TO_MAX, 3)
    def torso_to(self, goal, secs): self.torso_traj(goal, secs)
    def torso_traj(self, goal, secs):
        up_goal = FollowJointTrajectoryGoal()
        up_goal.trajectory.joint_names = ["torso_lift_joint"]
        up_goal.trajectory.points.append(JointTrajectoryPoint(positions=[goal], time_from_start=rospy.Duration(secs)))
        
        self.torsoAC.send_goal(up_goal)
        self.torsoAC.wait_for_result()

    def input_or_quit(self, text):
        i = input(text)
        if i.lower() == "q" or i.lower() == "s":
            return False
        return True

    def align(self, Two):
        Tfwg = self.li.lookupTransform(self.world_frame, self.grasping_frame, rospy.Time(0))
        Twg = Tf2T(*Tfwg)
        Tgw = inverse_matrix(Twg)

        Tgo = quaternion_matrix(
                quaternion_multiply(
                    quaternion_inverse(Tfwg[1]), 
                    quaternion_from_matrix(Two)
                )
            )

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

        if self.input_or_quit("execute?"):
            print("EXECUTE")
            self.execute_ac.send_goal_and_wait(ExecuteTrajectoryGoal(trajectory=tr))
        
    def place(self):
        self.activate_torso()

        if self.t_current == None: 
            print("no torso state yet ...")
            return

        try:
            self.ftCalib()
        except:
            print("ERROR calibrating FT sensor, quitting")
            return

        print("killing mm goal")
        self.mmKill()

        (tG, _) = self.li.lookupTransform(self.world_frame, self.grasping_frame, rospy.Time(0))
        torso_diff = np.min([self.TO_MAX, tG[2]-self.table_height])
        torso_goal = self.TO_MAX-torso_diff
        torso_secs = self.torso_vel*(torso_diff*100)
        
        print("moving torso down")
        self.torso_to(torso_goal, torso_secs)
        self.torso_to(self.t_current, 0.5)

        # wait a bit to settle
        # time.sleep(1)
        print("opening gripper")
        self.open_gripper()

        print("moving torso up")
        self.torso_up()
        
        print("placing done! :)")