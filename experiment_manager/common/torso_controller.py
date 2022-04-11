import rospy
import actionlib
import numpy as np

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal 

class TorsoStopController:
    """
    calls `torso_stop_controller` to move the torso down until contact / up
    subscribes to `/joint_states/` to receive the current troso position.
    """

    TORSO_IDX = 11
    TORSO_JOINT_NAME = 'torso_lift_joint'

    def __init__(self):
        self.torso_pos = None
        self.initialized = False

        self.js_sub = rospy.Subscriber("/joint_states", JointState, self.joint_states_cb, queue_size=1)
        self.torso_client = actionlib.SimpleActionClient("/torso_stop_controller/follow_joint_trajectory", FollowJointTrajectoryAction)

    def _require_setup(self):
        assert self.initialized, "TorsoStopController not initialized, did you call setup()?"

    def setup(self, timeout: float = 5.0) -> bool:
        rospy.loginfo("waiting for joint state")

        start = rospy.Time.now()
        while self.torso_pos == None:
            if rospy.Time.now() - start > timeout:
                rospy.logerr("didn't get torso position")
                return self.initialized

        rospy.loginfo("waiting for controller's action server ...")
        if self.torso_client.wait_for_server(timeout=rospy.Duration(timeout)):
            rospy.loginfo("found it!")
            self.initialized = True
        else:
            rospy.logerr("ac server not found ...")
        return self.initialized

    def generate_trajectory(self, first: float, last: float, duration: float = 5.0, num_points:int = 5) -> JointTrajectory:
        jt = JointTrajectory()
        jt.header.frame_id = 'base_footprint'
        jt.joint_names = [self.TORSO_JOINT_NAME]

        pts = []
        for t, j in zip(np.linspace(0, duration, num_points), np.linspace(first, last, num_points)):
            jp = JointTrajectoryPoint()
            jp.positions = [j]

            if t == 0.0: t += 0.1
            tm = rospy.Time(t)
            jp.time_from_start.secs = tm.secs
            jp.time_from_start.nsecs = tm.nsecs

            pts.append(jp)

        jt.points = pts
        return jt

    def joint_states_cb(self, m: JointState):
        self.torso_pos = np.round(m.position[self.TORSO_IDX], 4)

    def move_rel(self, q_offset: float, duration: float = 5.0) -> int:
        return self.move_to(self.torso_pos+q_offset, duration=duration)

    def move_to(self, q_goal: float, duration: float = 5.0) -> int:
        self._require_setup() # make sure we're setup

        rospy.loginfo(f"moving torso from {self.torso_pos:.4f} to {q_goal:.4f}")
        return self._send_sync(self.generate_trajectory(
            first=self.torso_pos, 
            last=q_goal, 
            duration=duration)
        )

    def _send_sync(self, traj: JointTrajectory) -> bool:
        g = FollowJointTrajectoryGoal()
        g.trajectory = traj

        return self.torso_client.send_goal_and_wait(g)

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", "-g", default=None, type=float)
    ap.add_argument("--offset", "-o", default=None, type=float)

    args = ap.parse_args()
    assert args.goal is not None or args.offset is not None, "no goal given"

    rospy.init_node("torso_movement")
    torso = TorsoStopController()
    if not torso.setup():
        rospy.fatal("could not initialize torso controller")
        exit(-1)

    if args.goal is not None:
        torso.move_to(args.goal)
    elif args.offset is not None:
        torso.move_rel(args.offset)
    else:
        rospy.logfatal("unknown error")