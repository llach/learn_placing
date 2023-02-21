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

placing_object_name = "pot2"

class TrackerSub:
    robot_frame_name = "tiago"

    def __init__(self, br, otp, tracker_name):
        self.br = br
        self.otp = otp
        self.tracker_name = tracker_name
        self.ln = tracker_name.lower()
        rospy.Subscriber("/vrpn_client_node/"+self.tracker_name+"/pose", PoseStamped, self._tracker_pose_callback)

    def _tracker_pose_callback(self, m): 
        T = trafo2homogeneous(pos2arr(m.pose.position), quat2arr(m.pose.orientation))

        # to have a valid TF tree, we need to publish the inverse from robot to OptiTrack origin
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

        if self.ln == placing_object_name: 
            # these two static TFs are needed by some components for evaluation
            self.otp.object_gripper_tf()
            self.br.sendTransform(
                translation=[0,0,0],
                rotation=[0,0,0,1],
                time=rospy.Time.now(),
                child="pot",
                parent=frames[0]
            )


class OptiTrackPublisher:

    tracker_names = []
    trackers = []
    br = TransformBroadcaster()
    world_frame = "base_footprint"
    grasping_frame = "gripper_grasping_frame"

    def __init__(self) -> None:
        self.li = TransformListener()
        rospy.loginfo("waiting for transform")
        for _ in range(6):
            try:
                self.li.waitForTransform(self.grasping_frame, self.world_frame, rospy.Time(0), rospy.Duration(3))
                break
            except Exception as e: print(e)

    def object_gripper_tf(self):
        try:
            (twg, Qwg) = self.li.lookupTransform(self.world_frame, self.grasping_frame, rospy.Time())
            (two, Qwo) = self.li.lookupTransform(self.world_frame, f"{placing_object_name}_opti", rospy.Time())
            Twg = trafo2homogeneous(twg, Qwg)
            Two = trafo2homogeneous(two, Qwo)
        except Exception as e:
            print(e)
            return

        Tgo = np.eye(4)
        Tgo[:3,:3] = np.linalg.inv(Twg)[:3,:3].dot(Two[:3,:3])

        self.br.sendTransform(
            translation=Tgo[:3,3],
            rotation=trafo.quaternion_from_matrix(Tgo),
            time=rospy.Time.now(),
            child="object",
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
        r = rospy.Rate(50)

        while not rospy.is_shutdown():
            # periodically check for new trackers
            if i % 50 == 0: self._refresh_tracker_list()
            i += 1

            # publish static 
            self.br.sendTransform(
                translation=[0.235, -0.0625, 0],
                rotation=trafo.quaternion_from_euler(0, 0, np.pi/2),
                time=rospy.Time.now(),
                child="tiago_opti",
                parent="torso_lift_link"
            )

            r.sleep()

if __name__ == "__main__":
    rospy.init_node("optitrack_publisher")
    otp = OptiTrackPublisher()
    otp.run()