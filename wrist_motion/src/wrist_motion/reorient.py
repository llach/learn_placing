from tkinter import N
import rospy
import time
import numpy as np

from tf import TransformListener
from tf import transformations as tf
from tf.transformations import quaternion_matrix
from wrist_motion.tiago_controller import TIAGoController
from wrist_motion.marker import frame, box, orientationArrow

from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Vector3
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from moveit_msgs.msg import DisplayTrajectory, RobotState, RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import MarkerArray

x_ = [1,0,0]
y_ = [0,1,0]
z_ = [0,0,1]

Rx = lambda a: tf.rotation_matrix(a, x_)
Ry = lambda a: tf.rotation_matrix(a, y_)
Rz = lambda a: tf.rotation_matrix(a, z_)

Qx = lambda a: tf.quaternion_about_axis(a, x_)
Qy = lambda a: tf.quaternion_about_axis(a, y_)
Qz = lambda a: tf.quaternion_about_axis(a, z_)


def sample_random_orientation_southern_hemisphere():
    # uniform sphere sampling modified to only sample in lower hemisphere (i.e. phi in [PI/2, 3PI/2])
    theta, phi = 2*np.pi*np.random.uniform(0, 1), np.arccos(-np.random.uniform(0, 1))

    Qzt = Qz(theta)
    Qyp = Qy(phi)

    return tf.quaternion_matrix(tf.quaternion_multiply(Qzt, Qyp))

OBJECT_FRAME = "grasped_object"
GRIPPER_FRAME = "gripper_left_grasping_frame"

class Reorient:
    JOINTS = ['torso_lift_joint', 'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint']

    def __init__(self) -> None:
        self.Tinit = None
        self.start_state = None
        self.should_get_js = True

        self.c: TIAGoController = TIAGoController(initial_state=len(self.JOINTS)*[0.0])
        # self.mg = moveit_commander.MoveGroupCommander(self.PLANNING_GROUP)
        
        # tolerances are the box's side length (we pass tolernaces/2 to the planner)
        # the black wagon at PAL measures 40x40x57
        self.tol = np.array(3*[0.1])
        self.eef_axis = np.array([0,0,1])

        self.Toff = tf.rotation_matrix(-0.5*np.pi, [0,1,0]) # this can be done in a general fashion. find orthogonal axis and than the dot product of X (arrow base orientation) and desired axis
        # self.Toss = np.eye(4)

        self.valid_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        self.reinit_srv = rospy.Service('/reinit_wrist_planner', Empty, self.reinit)
        self.js_sub = rospy.Subscriber('/joint_states', JointState, self.jointstate_cb)
        self.marker_pub = rospy.Publisher('/reorient_markers', MarkerArray, queue_size=10)
        self.traj_pub = rospy.Publisher('/reorient_trajectory', DisplayTrajectory, queue_size=10)

        print("setup done")

    def reinit(self, req):
        print("REINIT wrist planner")
        self.Tinit = None
        return EmptyResponse() 

    def update_obj_transform(self, Tgo):
        self.c.robot.joints["target"].set_T(Tgo)
    
    def get_target_joint_tf(self):
        return self.c.robot.joints["target"].T

    def jointstate_cb(self, msg):
        if not self.should_get_js: return # only process joint state if we need it

        self.start_state = len(self.JOINTS)*[0.0]
        for n, p in zip(msg.name, msg.position):
            if n in self.JOINTS:
                idx = self.JOINTS.index(n)
                self.start_state[idx] = p
        if self.Tinit is None: self.Tinit = self.c.fk_for_joint_position(self.start_state)[0]
        self.should_get_js = False

    def raw2rs(self, names, pos):
        return self.js2rs(JointState(name=names, position=pos))

    def js2rs(self, js: JointState):
        return RobotState(joint_state=js)

    def check_validity(self, rs: RobotState):
        req = GetStateValidityRequest(robot_state=rs)
        res = self.valid_srv.call(req)
        return res.valid

    def orientation_only_T(self, T):
        _T = np.identity(4)
        _T[0:3, 0:3] = T[0:3, 0:3]
        return _T

    def position_only_T(self, T):
        _T = np.identity(4)
        _T[0:3, 3] = T[0:3, 3]
        return _T

    def pub_markers(self, To, pub_goal=True):
        start_T, _ = self.c.robot.fk(self.c.target_link, dict(zip(self.JOINTS, self.start_state)))

        start_frame = frame(start_T, ns="start_frame")
        goal_frame = frame(self.c.T, ns="goal_frame", alpha=0.8)

        tolerance_box = box(self.position_only_T(self.Tinit), Vector3(*self.tol))
        tolerance_box.header.frame_id = "base_footprint"
        tolerance_box.id = 6

        # orientation for arrows is along the +X axis, so Toff rotates the X axis to align with the Z axis (eef axis)
        start_arrow = orientationArrow(self.orientation_only_T(start_T), color=ColorRGBA(0, 0, 1, 1))
        start_arrow.id = 7

        # black arrow for goal
        goal_arrow = orientationArrow(self.orientation_only_T(To), color=ColorRGBA(0, 0, 0, 0.8))
        goal_arrow.id = 8

        # white arrow for final arow after planning
        end_arrow = orientationArrow(self.orientation_only_T(self.c.T), color=ColorRGBA(1, 1, 1, 0.6))
        end_arrow.id = 9

        ma = MarkerArray(markers=[
            *start_frame,
            # tolerance_box,
            # start_arrow,
            # end_arrow
        ])

        if pub_goal:
            ma.markers.append(goal_arrow)
            for gf in goal_frame: ma.markers.append(gf)

        self.marker_pub.publish(ma)

    def plan_random(self, Tgo, Tgocorr, publish_traj=False, check_validity=True, table_height=0.0):
        print("getting current state")

        self.should_get_js = True
        while self.should_get_js: time.sleep(0.1) # jointstate subscriber thread will set flag to false when done
        
        # sets the transform from gripper to object as a fixed joint in chain
        self.update_obj_transform(Tgo)

        print("generating trajectory")
        # self.To = sample_random_orientation_southern_hemisphere()
        # if To is None: To = self.To

        goal_state, failed = self.c.reorientation_trajectory(
            Tgocorr, 
            self.c.T, 
            self.start_state, 
            tol=.5*self.tol, 
            eef_axis=self.eef_axis, # 
            max_steps=100
        )

        if failed: 
            self.pub_markers(Tgocorr, False)
            return None, failed

        Tarm, _ = self.c.fk_for_link("arm_left_7_link")
        Thand, _ = self.c.fk_for_link(GRIPPER_FRAME)

        # if Tarm[2,3]<Thand[2,3]: 
        #     print("z constraint failed")
        #     self.pub_markers(To, True)
        #     return None, True

        # torso is frist joint, on the real robot we can't use it (otherwise moveit complains about planning groups / controllers)
        traj_points = np.linspace(self.start_state[1:], goal_state[1:], 10)
        traj_times = np.linspace(0, 2, 10)

        gripper_zs = np.array([self.c.fk_for_joint_position([0.0]+list(tp))[0][2,3] for tp in traj_points])

        # if not np.all(gripper_zs>table_height):
        #     self.pub_markers(Tgocorr, True)
        #     print(f"some points were below table height {table_height} \n\t{gripper_zs}  \n\t{gripper_zs>table_height}")
        #     return None, True

        rss = []
        traj = JointTrajectory()
        traj.joint_names = self.c.joint_msg.name[1:]

        for p, t in zip(traj_points, traj_times):
            rss.append(self.raw2rs(self.JOINTS[1:], p))
            traj.points.append(JointTrajectoryPoint(positions=p, time_from_start=rospy.Duration(t)))

        init_rs = self.raw2rs(self.JOINTS, self.start_state)

        rt = RobotTrajectory(joint_trajectory=traj)
        disp = DisplayTrajectory(trajectory_start=init_rs)
        disp.trajectory.append(rt)

        if check_validity:
            validities = [self.check_validity(rs) for rs in rss]
            if all(validities):
                print("valid trajectory")
            else:
                print("trajectory not valid!")
                self.pub_markers(False)
                return rt, True

        if publish_traj:
            self.traj_pub.publish(disp)

        self.pub_markers(Tgocorr, True)
        return rt, failed
    
    # def execute(self, tr, wait=True):
    #     print("executing trajectory ...")
    #     self.mg.execute(tr, wait=wait)
    #     print("execution finished.")
