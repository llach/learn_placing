import numpy
import qpsolvers
import learn_placing.common.transformations as tf

from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Transform, Quaternion, Vector3

from wrist_motion.robot_model import RobotModel, Joint

TIAGO_IGNORED_JOINTS = [
    "base_laser_joint",
    "base_sonar_03_joint",
    "base_sonar_02_joint",
    "base_sonar_01_joint",
    "base_mic_front_left_joint",
    "base_mic_front_right_joint",
    "base_mic_back_left_joint",
    "base_mic_back_right_joint",
    "cover_joint",
    "base_antenna_left_joint",
    "base_antenna_right_joint",
    "base_imu_joint",
    "suspension_right_joint",
    "wheel_right_joint",
    "suspension_left_joint",
    "wheel_left_joint",
    "caster_front_right_1_joint",
    "caster_front_right_2_joint",
    "caster_front_left_1_joint",
    "caster_front_left_2_joint",
    "caster_back_right_1_joint",
    "caster_back_right_2_joint",
    "caster_back_left_1_joint",
    "caster_back_left_2_joint",
    "head_1_joint",
    "head_2_joint",
    "gripper_right_finger_joint",
    "gripper_left_finger_joint",
    "xtion_joint",
    "xtion_optical_joint",
    "xtion_orbbec_aux_joint",
    "xtion_depth_joint",
    "xtion_depth_optical_joint",
    "xtion_rgb_joint",
    "xtion_rgb_optical_joint",
    "rgbd_laser_joint",
    "torso_fixed_column_link",
    "arm_right_1_joint", 
    "arm_right_2_joint", 
    "arm_right_3_joint", 
    "arm_right_4_joint", 
    "arm_right_5_joint", 
    "arm_right_6_joint", 
    "arm_right_7_joint",
    "arm_right_tool_joint",
    "wrist_right_ft_joint",
    "wrist_right_tool_joint",
    "hand_right_tool_joint",
    "hand_right_palm_joint",
    "hand_right_thumb_joint",
    "hand_right_index_joint",
    "hand_right_mrl_joint",
    "hand_right_thumb_abd_joint",
    "hand_right_thumb_virtual_1_joint",
    "hand_right_thumb_flex_1_joint",
    "hand_right_thumb_virtual_2_joint",
    "hand_right_thumb_flex_2_joint",
    "hand_right_index_abd_joint",
    "hand_right_index_virtual_1_joint",
    "hand_right_index_flex_1_joint",
    "hand_right_index_virtual_2_joint",
    "hand_right_index_flex_2_joint",
    "hand_right_index_virtual_3_joint",
    "hand_right_index_flex_3_joint",
    "hand_right_little_flex_1_joint",
    "hand_right_little_virtual_2_joint",
    "hand_right_little_flex_2_joint",
    "hand_right_little_virtual_3_joint",
    "hand_right_little_flex_3_joint",
    "hand_right_grasping_fixed_joint",
    "hand_right_safety_box_joint",
    "hand_right_middle_flex_3_joint",
    "hand_right_ring_abd_joint",
    "hand_right_ring_virtual_1_joint",
    "hand_right_ring_flex_1_joint",
    "hand_right_ring_virtual_2_joint",
    "hand_right_ring_flex_2_joint",
    "hand_right_ring_virtual_3_joint",
    "hand_right_ring_flex_3_joint",
    "hand_right_little_abd_joint",
    "hand_right_little_virtual_1_joint",
    "hand_right_middle_abd_joint",
    "hand_right_middle_virtual_1_joint",
    "hand_right_middle_flex_1_joint",
    "hand_right_middle_virtual_2_joint",
    "hand_right_middle_flex_2_joint",
    "hand_right_middle_virtual_3_joint",
    "hand_right_middle_flex_3_joint",
    "gripper_left_right_finger_joint",
    "gripper_left_left_finger_joint",
]

TIAGO_DISABLED_JOINTS = [
    "torso_lift_joint"
]

GRASP_FRAME_POSE = TransformStamped(
    header=Header(frame_id='gripper_left_grasping_frame'),
    child_frame_id='target',
    transform=Transform(rotation=Quaternion(*[0,0,0,0]),translation=Vector3(0, 0, 0))
)

def skew(w):
    return numpy.array([[0, -w[2], w[1]],
                        [w[2], 0, -w[0]],
                        [-w[1], w[0], 0]])


class TIAGoController(object):
    damping = 0.1
    threshold = 0.1

    def __init__(self, initial_state, pose=GRASP_FRAME_POSE, ignore_joints=TIAGO_IGNORED_JOINTS, disabled_joints=TIAGO_DISABLED_JOINTS):

        self.robot = RobotModel(ignore_joints=ignore_joints)
        self.robot._add(Joint(pose))  # add a fixed end-effector transform
        self.joint_msg = JointState()
        self.target_link = pose.child_frame_id

        self.joint_msg.name = [j.name for j in self.robot.active_joints]
        self.joint_msg.position = initial_state
        self.T, self.J = self.robot.fk(self.target_link, dict(zip(self.joint_msg.name, self.joint_msg.position)))
        self.N = self.J.shape[1]  # number of (active) joints
        self.preferred_joints = self.joint_msg.position.copy()
        self.joint_weights = numpy.ones(self.N)
        self.cartesian_weights = numpy.ones(6)
        self.mins = numpy.array([j.min for j in self.robot.active_joints])
        self.maxs = numpy.array([j.max for j in self.robot.active_joints])
        self.prismatic = numpy.array([j.jtype == j.prismatic for j in self.robot.active_joints])

        for dj in disabled_joints:
            self.robot.disable_joint(dj)

        self.targets=dict()

    def setTarget(self, name, goal):
        self.targets[name] = goal

    def actuate(self, q_delta):
        self.joint_msg.position += q_delta.ravel()
        # clip (prismatic) joints
        self.joint_msg.position[self.prismatic] = numpy.clip(self.joint_msg.position[self.prismatic],
                                                             self.mins[self.prismatic], self.maxs[self.prismatic])
        self.T, self.J = self.robot.fk(self.target_link, dict(zip(self.joint_msg.name, self.joint_msg.position)))

    def set_state(self, state):
        assert len(state)==len(self.joint_msg.position), "number of joints / state mismatch"

        self.joint_msg.position = state
        self.T, self.J = self.robot.fk(self.target_link, dict(zip(self.joint_msg.name, self.joint_msg.position)))

    def fk_for_joint_position(self, pos):
        assert len(pos)==len(self.joint_msg.position), "number of joints / state mismatch"
        return self.robot.fk(self.target_link, dict(zip(self.joint_msg.name, pos)))

    def fk_for_link(self, target_link):
        return self.robot.fk(target_link, dict(zip(self.joint_msg.name, self.joint_msg.position)))

    def solve(self, tasks):
        """Hierarchically solve tasks of the form J dq = e"""
        def invert_clip(s):
            return 1./s if s > self.threshold else 0.

        def invert_damp(s):
            return s/(s**2 + self.damping**2)

        def invert_smooth_clip(s):
            return s/(self.threshold**2) if s < self.threshold else 1./s

        N = numpy.identity(self.N)  # nullspace projector of previous tasks
        JA = numpy.zeros((0, self.N))  # accumulated Jacobians
        qdot = numpy.zeros(self.N)

        if isinstance(tasks, tuple):
            tasks = [tasks]

        for J, e in tasks:
            U, S, Vt = numpy.linalg.svd(J.dot(N) * self.joint_weights[None, :])
            # compute V'.T = V.T * Mq.T
            Vt *= self.joint_weights[None, :]

            rank = min(U.shape[0], Vt.shape[1])
            for i in range(rank):
                S[i] = invert_smooth_clip(S[i])

            qdot += numpy.dot(Vt.T[:, 0:rank], S * U.T.dot(numpy.array(e) - J.dot(qdot))).reshape(qdot.shape)

            # compute new nullspace projector
            JA = numpy.vstack([JA, J])
            U, S, Vt = numpy.linalg.svd(JA)
            accepted_singular_values = (S > 1e-3).sum()
            VN = Vt[accepted_singular_values:].T
            N = VN.dot(VN.T)
        self.nullspace = VN  # remember nullspace basis
        return qdot

    def solve_qp(self, tasks):
        """Solve tasks (J, ub, lb) of the form lb ≤ J dq ≤ ub
           using quadratic optimization: https://pypi.org/project/qpsolvers"""
        maxM = numpy.amax([task[0].shape[0] for task in tasks]) # max task dimension
        sumM = numpy.sum([task[0].shape[0] for task in tasks]) # sum of all task dimensions
        usedM = 0
        # allocate arrays once
        G, h = numpy.zeros((2*sumM, self.N + maxM)), numpy.zeros(2*sumM)
        P = numpy.identity(self.N+maxM)
        P[self.N:, self.N:] *= 1.0  # use different scaling for slack variables?
        q = numpy.zeros(self.N + maxM)

        # joint velocity bounds + slack bounds
        upper = numpy.hstack([numpy.minimum(0.1, self.maxs - self.joint_msg.position), numpy.zeros(maxM)])
        lower = numpy.hstack([numpy.maximum(-0.1, self.mins - self.joint_msg.position), numpy.full(maxM, -numpy.infty)])

        # fallback solution
        dq = numpy.zeros(self.N)

        def add_constraint(A, bound):
            G[usedM:usedM+M, :N] = A
            G[usedM:usedM+M, N:N+M] = numpy.identity(M)  # allow (negative) slack variables
            h[usedM:usedM+M] = bound
            return usedM + M

        failed=False
        for idx, task in enumerate(tasks):
            try:  # inequality tasks are pairs of (J, ub, lb=None)
                J, ub, lb = task
            except ValueError:  # equality tasks are pairs of (J, err)
                J, ub = task
                lb = ub  # turn into inequality task: err ≤ J dq ≤ err
            J = numpy.atleast_2d(J)
            M, N = J.shape

            # augment G, h with current task's constraints
            oldM = usedM
            usedM = add_constraint(J, ub)
            if lb is not None:
                usedM = add_constraint(-J, -lb)

            result = qpsolvers.solve_qp(P=P[:N+M, :N+M], q=q[:N+M],
                                        G=G[:usedM, :N+M], h=h[:usedM], A=None, b=None,
                                        lb=lower[:N+M], ub=upper[:N+M])
            if result is None:
                print("{}: failed  ".format(idx))
                usedM = oldM  # ignore subtask and continue with subsequent tasks
                failed=True
            else: # adapt added constraints for next iteration
                dq, slacks = result[:N], result[N:]
                # print("{}:".format(idx), slacks, " ", end='')
                G[oldM:usedM,N:N+M] = 0
                h[oldM:oldM+M] += slacks
                if oldM+M < usedM:
                    h[oldM+M:usedM] -= slacks
        # print()
        self.nullspace = numpy.zeros((self.N, 0))
        return dq, failed

    @staticmethod
    def vstack(items):
        return numpy.vstack(items) if items else None

    @staticmethod
    def hstack(items):
        return numpy.hstack(items) if items else None

    def stack(self, tasks):
        """Combine all tasks by stacking them into a single Jacobian"""
        Js, errs = zip(*tasks)
        return self.vstack(Js), numpy.hstack(errs)

    def position_task(self, T_tgt, T_cur, scale=1.0):
        """Move eef towards a specific target point in base frame"""
        return self.J[:3], scale*(T_tgt[0:3, 3]-T_cur[0:3, 3])

    def orientation_task(self, T_tgt, T_cur, scale=1.0):
        """Move eef into a specific target orientation in base frame"""
        delta = numpy.identity(4)
        delta[0:3, 0:3] = T_cur[0:3, 0:3].T.dot(T_tgt[0:3, 0:3])
        angle, axis, _ = tf.rotation_from_matrix(delta)
        # transform rotational velocity from end-effector into base frame orientation (only R!)
        return self.J[3:], scale*(T_cur[0:3, 0:3].dot(angle * axis))

    def cone_task(self, axis, reference, threshold):
        """Align axis in eef frame to lie in cone spanned by reference axis and opening angle acos(threshold)"""
        axis = self.T[0:3, 0:3].dot(axis)  # transform axis from eef frame to base frame
        return reference.T.dot(skew(axis)).dot(self.J[3:]), (reference.T.dot(axis) - threshold), None

    def pose_task(self, T_tgt, T_cur, scale=(1.0, 1.0)):
        """Perform position and orientation task with same priority"""
        return self.stack([self.position_task(T_tgt, T_cur, scale=scale[0]),
                           self.orientation_task(T_tgt, T_cur, scale=scale[1])])

    def reorientation_trajectory(
        self, 
        To,
        Tinit,
        initial_state,
        tol,  # tolerance box
        eef_axis,
        max_steps=25, 
        eps=0.00005):

        self.set_state(initial_state)
        state = self.joint_msg.position[:] # copy

        prev_qdelta = None
        for _ in range(max_steps):
            tasks = []
            # Jp, ep = self.position_task(Tinit, self.T)
            # ub, lb = (ep+tol), (ep-tol)
            # tasks.append((Jp, ub, lb))

            oTask = self.pose_task(To, self.T)
            tasks.append(oTask)

            # _, Jelbow = self.fk_for_link("arm_7_link")
            # Jup = Jelbow[2]-self.J[2]
            # tasks.append((Jup, 10.0, 0.02))

            q_delta, failed = self.solve_qp(tasks)
            if failed: return None, True

            if prev_qdelta is not None:
                qd = numpy.array(q_delta[1:])
                qdd = numpy.abs(qd-prev_qdelta)

                if numpy.all(qdd<eps): 
                    print("minium error")
                    break
            prev_qdelta = numpy.array(q_delta[1:])

            state += q_delta
            self.actuate(q_delta)
        else:
            print("maximum steps")

        return state, failed