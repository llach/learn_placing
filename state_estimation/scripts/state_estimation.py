#!/usr/bin/python
from collections import deque
import tf
import rospy
import numpy as np

from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Float64
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import TransformStamped
from tf.transformations import unit_vector, quaternion_multiply, quaternion_conjugate

def v2l(v):
    return [v.x, v.y, v.z]

def q2l(q):
    return [q.x, q.y, q.z, q.w]

def rotate_v(v, q):
    v = list(unit_vector(v))
    v.append(0.0) # vector as pure quaternion, i.e. normalized and 4D
    return quaternion_multiply(
        quaternion_multiply(q, v),
        quaternion_conjugate(q)
    )[:3]

class TagTransformator:

    MARKER_AXIS_UP = [0,-1,0]

    def __init__(self, cam_name, n_samples = 10, common_frame="camera_link"):
        self.cam_name = cam_name
        self.n_samples = n_samples
        self.common_frame = common_frame
        self.tag_topic = f"/tag_detections_{cam_name}"

        self.calibrated = False
        self.markers = deque(maxlen=self.n_samples)

        self.tag_sub = rospy.Subscriber(self.tag_topic, AprilTagDetectionArray, self.tag_cb)

    def tag_cb(self, am: AprilTagDetectionArray):
        nmarkers = len(am.detections)
        if nmarkers == 0: return

        # assumption: all markers are orientated the same way on the block
        # -> knowing the offset from one marker-Z will be the same for other markers

        timestamp = am.header.stamp
        markers = {}
        for d in am.detections:
            mid = d.id[0]
            markers.update({mid: []})

            tfs = TransformStamped()
            tfs.header = am.header
            tfs.child_frame_id = f"tag_{mid}_{self.cam_name}"
            tfs.transform.translation = d.pose.pose.pose.position
            tfs.transform.rotation = d.pose.pose.pose.orientation

            markers[mid].append(timestamp)
            markers[mid].append(tfs)
        self.markers.append(markers)

        # marker_vs = []
        # for m in am.detections:
        #     try:
        #         q = self.get_tag_tf(m.id[0])
        #         marker_vs.append(rotate_v(self.marker_axis_up, q))
        #     except:
        #         # rospy.logwarn(f"transform for marker {m.id[0]} not ready.")
        #         pass
        # if len(marker_vs) == 0:
        #     # rospy.logwarn(f"no successful marker transormation")
        #     pass
        #     # return

        # mean_v = np.mean(marker_vs, axis=0)
        # if not self.calibrated:
        #     print("starting TAG calibration")
        #     self.table_normal = mean_v
        #     self.calibrated = True
        #     print("TAG calibration done")
        #     return
        
        # angle = np.dot(mean_v, self.table_normal)

class StateEstimator:

    def __init__(self, cams = [], max_age = rospy.Duration(2)):
        self.max_age = max_age
        self.angle_pub = rospy.Publisher("/normal_angle", Float64, queue_size=1)
        self.calib_srv = rospy.Service("object_state_calibration", Empty, self.calibrate_cb)

        self.tts = []
        for cam in cams:
            self.tts.append(TagTransformator(cam))

        self.br = tf.TransformBroadcaster()

    def calibrate_cb(self, *args, **kwargs):
        r = rospy.Rate(50)

        self.calibrated = False
        while not self.calibrated: r.sleep()

        return EmptyResponse()

    def process(self):
        latest = rospy.Time.now()-self.max_age

        tfs = {}
        for tt in self.tts:             # cams
            for m in tt.markers:        # last N detections
                for mid, v in m.items():
                    if v[0] < latest: continue # ignore old markers

                    # store tf for this marker id if none is present
                    if mid not in tfs: 
                        tfs.update({mid: v[1]})
                        self.br.sendTransform(v2l(v[1].transform.translation), q2l(v[1].transform.rotation), v[1].header.stamp, f"tag_{mid}", v[1].header.frame_id)

if __name__ == "__main__":
    rospy.init_node("object_state_estimation")
    se = StateEstimator(["external_webcam"])
    
    r = rospy.Rate(50)
    while not rospy.is_shutdown():
        se.process()
        r.sleep()