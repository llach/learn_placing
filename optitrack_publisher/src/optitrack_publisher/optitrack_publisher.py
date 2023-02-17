import re
import rospy
import numpy as np

from threading import Lock
from geometry_msgs.msg import PoseStamped
from tf import TransformBroadcaster, TransformListener
import tf.transformations as trafo 

def pos2arr(p): return np.array([p.x, p.y, p.z])
def quat2arr(q): return np.array([q.x, q.y, q.z, q.w])
def trafo2homogeneous(pos, quat):
    T = trafo.quaternion_matrix(quat)
    T[:3,3]  = pos
    return T

class TrackerSub:
    robot_frame_name = "tiago"

    def __init__(self, br, tracker_name):
        self.br = br
        self.tracker_name = tracker_name
        self.ln = tracker_name.lower()
        rospy.Subscriber("/vrpn_client_node/"+self.tracker_name+"/pose", PoseStamped, self._tracker_pose_callback)

    def _tracker_pose_callback(self, m): 
        T = trafo2homogeneous(pos2arr(m.pose.position), quat2arr(m.pose.orientation))

        frames = ["%s_opti" % self.tracker_name.lower(), "optitrack"]
        if self.tracker_name.lower() == self.robot_frame_name:
            T = np.linalg.inv(T)
            frames.reverse()

        self.br.sendTransform(
            translation=T[:3,3],
            rotation=trafo.quaternion_from_matrix(T),
            time=rospy.Time.now(),
            child=frames[0],
            parent=frames[1]
        )

class OptiTrackPublisher:

    tracker_names = []
    trackers = []
    br = TransformBroadcaster()

    def _refresh_tracker_list(self):
        # get topics from vrpn client
        topics = [t[0] for t in rospy.get_published_topics() if "vrpn" in t[0].lower()]
        
        for t in topics:
            # extract tracker name, skip if we have it already
            tracker_name = re.search(r"node/(.*)/pose", t).group(1)
            if tracker_name in self.tracker_names: continue 

            print("adding new tracker %s" % tracker_name)
            self.tracker_names.append(tracker_name)
            self.trackers.append(TrackerSub(self.br, tracker_name))

    def run(self):
        print("optitrack_publisher started!")
        i = 0
        r = rospy.Rate(50)

        while not rospy.is_shutdown():
            # periodically check for new trackers
            if i % 50 == 0: self._refresh_tracker_list()
            i += 1

            r.sleep()

if __name__ == "__main__":
    rospy.init_node("optitrack_publisher")
    otp = OptiTrackPublisher()
    otp.run()