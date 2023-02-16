import re
import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from tf import TransformBroadcaster, TransformListener
import tf.transformations as trafo 

def pos2arr(p): return np.array([p.x, p.y, p.z])
def quat2arr(q): return np.array([q.x, q.y, q.z, q.w])
def trafo2homogeneous(pos, quat):
    T = trafo.quaternion_matrix(quat)
    T[:3,3]  = pos
    return T


PUBLISH_GRASPED = ["wood_02"]

class TrackerSub:
    robot_frame_name = "tiago"

    def __init__(self, br, otp, tracker_name):
        self.br = br
        self.otp = otp
        self.tracker_name = tracker_name
        rospy.Subscriber("/vrpn_client_node/"+self.tracker_name+"/pose", PoseStamped, self._tracker_pose_callback)

    def _tracker_pose_callback(self, m): 
        T = trafo2homogeneous(pos2arr(m.pose.position), quat2arr(m.pose.orientation))
        Tx = trafo.rotation_matrix(np.pi/2, [1,0,0])
        Txinv = trafo.rotation_matrix(-np.pi/2, [1,0,0])

        # T = Tz.dot(Tx.dot(T)).dot(Txinv).dot(Tzinv)

        T = Tx.dot(T).dot(Txinv)

        frames = ["%s_opti" % self.tracker_name.lower(), "optitrack"]
        if self.tracker_name.lower() == self.robot_frame_name:
            T = np.linalg.inv(T)
            T[:3,:3] = np.eye(3)
            frames.reverse()

        self.br.sendTransform(
            translation=T[:3,3],
            rotation=trafo.quaternion_from_matrix(T),
            time=rospy.Time.now(),
            child=frames[0],
            parent=frames[1]
        )

        if self.tracker_name.lower() in PUBLISH_GRASPED: self.otp.publish_gripper_tf(self.tracker_name.lower(), T)

class OptiTrackPublisher:

    tracker_names = []
    trackers = []
    br = TransformBroadcaster()
    world_frame = "base_footprint"
    grasping_frame = "gripper_grasping_frame"

    def __init__(self):
        self.li = TransformListener()
        print("waiting for transforms")
        for _ in range(6):
            try:
                self.li.waitForTransform(self.grasping_frame, self.world_frame, rospy.Time(0), rospy.Duration(3))
                break
            except Exception as e: print(e)

    def publish_gripper_tf(self, name, Two):
        (twg, Qwg) = self.li.lookupTransform(self.world_frame, self.grasping_frame, rospy.Time(0))
        Twg = trafo2homogeneous(twg, Qwg)

        Two[:3, 3] = [0, 0, 0]
        Tgo = np.linalg.inv(Twg)@Two

        self.br.sendTransform(
            translation=[0, 0, 0],
            rotation=trafo.quaternion_from_matrix(Tgo),
            time=rospy.Time.now(),
            child=f"{name}_grasped",
            parent=self.grasping_frame
        )

    def _refresh_tracker_list(self):
        # get topics from vrpn client
        topics = [t[0] for t in rospy.get_published_topics() if "vrpn" in t[0].lower()]
        
        for t in topics:
            # extract tracker name, skip if we have it already
            tracker_name = re.search(r"node/(.*)/pose", t).group(1)
            if tracker_name in self.tracker_names: continue 

            print("adding new tracker %s" % tracker_name)
            self.tracker_names.append(tracker_name)
            self.trackers.append(TrackerSub(self.br, self, tracker_name))

    def run(self):
        print("optitrack_publisher started!")
        i = 0
        r = rospy.Rate(2)

        while not rospy.is_shutdown():
            # periodically check for new trackers
            if i % 10 == 0: self._refresh_tracker_list()
            i += 1

            r.sleep()

if __name__ == "__main__":
    rospy.init_node("optitrack_publisher")
    otp = OptiTrackPublisher()
    otp.run()