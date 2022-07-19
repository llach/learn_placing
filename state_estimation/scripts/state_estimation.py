#!/usr/bin/python
import tf
import rospy
import numpy as np

from std_msgs.msg import Float64
from apriltag_ros.msg import AprilTagDetectionArray
from tf.transformations import unit_vector, quaternion_multiply, quaternion_conjugate

def rotate_v(v, q):
    v = list(unit_vector(v))
    v.append(0.0) # vector as pure quaternion, i.e. normalized and 4D
    return quaternion_multiply(
        quaternion_multiply(q, v),
        quaternion_conjugate(q)
    )[:3]


class StateEstimator:
    # TODO calibration service instead of doing it right away

    marker_axis_up = [0,-1,0]
    common_frame = "camera_link"

    def __init__(self):
        self.calibrated = False
        self.table_normal = None

        self.april_sub = rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.marker_cb)
        self.tfl = tf.TransformListener()
        self.angle_pub = rospy.Publisher("/normal_angle", Float64, queue_size=1)

    def marker_cb(self, am: AprilTagDetectionArray):
        if not self.calibrated and len(am.detections)<1:
            rospy.logwarn("couldn't calibrate, no markers present")
            return

        # average over up-pointing axis
        # TODO ideally, this calibration should happen on a per-marker basis and each
        # id should only be present once. difficult: what happens if we don't detect them during the calibration? when is calibration finished?
        marker_vs = []
        for m in am.detections:
            try:
                q = self.get_tag_tf(m.id[0])
                marker_vs.append(rotate_v(self.marker_axis_up, q))
            except:
                rospy.logwarn(f"transform for marker {m.id[0]} not ready.")
                # pass
        if len(marker_vs) == 0:
            rospy.logwarn(f"no successful marker transormation")
            # return

        mean_v = np.mean(marker_vs, axis=0)
        if not self.calibrated:
            self.table_normal = mean_v
            self.calibrated = True
            return
        
        angle = np.dot(mean_v, self.table_normal)
        # print((np.arccos(angle)*180)/np.pi)
        self.angle_pub.publish(angle)

    def get_tag_tf(self, num):
        return self.tfl.lookupTransform("camera_link", f"tag_{num}", rospy.Time(0))[1]


if __name__ == "__main__":
    rospy.init_node("object_state_estimation")

    se = StateEstimator()
    rospy.spin()