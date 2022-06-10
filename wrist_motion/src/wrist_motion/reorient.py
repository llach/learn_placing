import rospy
import time
import numpy as np
import moveit_commander

from tf import transformations as tf
from wrist_motion.tiago_controller import TIAGoController
from wrist_motion.marker import frame, box, orientationArrow

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


class Reorient:
    JOINTS = ['torso_lift_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint']
    PLANNING_GROUP = 'arm_torso'

    def __init__(self) -> None:
        self.Tinit = None
        self.start_state = None
        self.should_get_js = True

        self.c: TIAGoController = TIAGoController(initial_state=len(self.JOINTS)*[0.0])
        self.mg = moveit_commander.MoveGroupCommander(self.PLANNING_GROUP)
        
        self.tol = np.array([0.2, 0.2, 0.1])
        self.eef_axis = np.array([0,0,1])

        self.Toff = tf.rotation_matrix(-0.5*np.pi, [0,1,0]) # this can be done in a general fashion. find orthogonal axis and than the dot product of X (arrow base orientation) and desired axis

        self.valid_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        self.js_sub = rospy.Subscriber('/joint_states', JointState, self.jointstate_sb)
        self.marker_pub = rospy.Publisher('/reorient_markers', MarkerArray, queue_size=10)
        self.traj_pub = rospy.Publisher('/reorient_trajectory', DisplayTrajectory, queue_size=10)

    def jointstate_sb(self, msg):
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

    def pub_markers(self):
        start_T, _ = self.c.robot.fk(self.c.target_link, dict(zip(self.JOINTS, self.start_state)))

        start_frame = frame(start_T, ns="start_frame")
        goal_frame = frame(self.c.T, ns="goal_frame", alpha=0.8)

        tolerance_box = box(self.position_only_T(start_T), Vector3(*self.tol))
        tolerance_box.header.frame_id = "base_footprint"
        tolerance_box.id = 6

        # orientation for arrows is along the +X axis, so Toff rotates the X axis to align with the Z axis (eef axis)
        start_arrow = orientationArrow(self.orientation_only_T(start_T@self.Toff), color=ColorRGBA(0, 0, 1, 1))
        start_arrow.id = 7

        goal_arrow = orientationArrow(self.orientation_only_T(self.To), color=ColorRGBA(0, 0, 0, 0.8))
        goal_arrow.id = 8

        end_arrow = orientationArrow(self.orientation_only_T(self.c.T@self.Toff), color=ColorRGBA(1, 1, 1, 0.6))
        end_arrow.id = 9

        ma = MarkerArray(markers=[
            *start_frame, 
            *goal_frame,
            tolerance_box,
            start_arrow,
            goal_arrow,
            end_arrow
        ])

        self.marker_pub.publish(ma)

    def plan_random(self, publish_traj=True):
        print("getting current state")

        self.should_get_js = True
        while self.should_get_js: time.sleep(0.1) # jointstate subscriber thread will set flag to false when done

        print("generating trajectory")
        self.To = sample_random_orientation_southern_hemisphere()
        goal_state = self.c.reorientation_trajectory(self.To, self.Tinit, self.start_state, tol=.5*self.tol, eef_axis=self.eef_axis)

        traj_points = np.linspace(self.start_state, goal_state, 10)
        traj_times = np.linspace(0, 3, 10)

        rss = []
        traj = JointTrajectory()
        traj.joint_names = self.c.joint_msg.name

        for p, t in zip(traj_points, traj_times):
            rss.append(self.raw2rs(self.JOINTS, p))
            traj.points.append(JointTrajectoryPoint(positions=p, time_from_start=rospy.Duration(t)))

        init_rs = self.raw2rs(self.JOINTS, self.start_state)

        rt = RobotTrajectory(joint_trajectory=traj)
        disp = DisplayTrajectory(trajectory_start=init_rs)
        disp.trajectory.append(rt)

        validities = [self.check_validity(rs) for rs in rss]
        if all(validities):
            print("valid trajectory")
        else:
            print("trajectory not valid!")
            return False

        if publish_traj:
            while self.traj_pub.get_num_connections()<1:
                time.sleep(0.1)
            self.traj_pub.publish(disp)

        self.pub_markers()
        return rt
    
    def execute(self, tr, wait=True):
        print("executing trajectory ...")
        self.mg.execute(tr, wait=wait)
        print("execution finished.")