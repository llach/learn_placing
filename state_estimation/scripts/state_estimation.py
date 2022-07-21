#!/usr/bin/python
from collections import deque
import tf
import time
import rospy
import numpy as np

from threading import Lock
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Float64
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import TransformStamped
from tf.transformations import unit_vector, quaternion_multiply, quaternion_conjugate

def v2l(v):
    return np.array([v.x, v.y, v.z])

def q2l(q):
    return np.array([q.x, q.y, q.z, q.w])

def rotate_v(v, q):
    v = list(unit_vector(v))
    v.append(0.0) # vector as pure quaternion, i.e. normalized and 4D
    return quaternion_multiply(
        quaternion_multiply(q, v),
        quaternion_conjugate(q)
    )[:3]

class TagTransformator:

    MARKER_AXIS_UP = [0,-1,0]

    def __init__(self, cam_name, n_samples = 10, common_frame="camera_link", n_calibration_samples = 50):
        self.cam_name = cam_name
        self.n_samples = n_samples
        self.common_frame = common_frame
        self.tag_topic = f"/tag_detections_{cam_name}"

        self.calibrated = False
        self.should_calibrate = False
        self.calibration_samples = []
        self.n_calibration_samples = n_calibration_samples

        self.qoffset = None
        self.vstart = None

        self.l = Lock()

        self.markers = deque(maxlen=self.n_samples)
        self.tag_sub = rospy.Subscriber(self.tag_topic, AprilTagDetectionArray, self.tag_cb)

    def calibrate(self):
        self.qoffset = None
        self.calibrated = False
        self.should_calibrate = True
        self.calibration_samples = []

        print(f"staring calibration for cam {self.cam_name}")

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

            if self.calibrated:
                qcurrent = q2l(tfs.transform.rotation)
                vcurrent = rotate_v(self.MARKER_AXIS_UP, qcurrent)
                rad = np.dot(self.vstart, vcurrent)
                markers[mid].append(rad) # append the current orientation
            else:
                markers[mid].append(None) 
            
            if self.should_calibrate:
                # collect rotations that we use to calculate the 
                self.calibration_samples.append(q2l(tfs.transform.rotation))

                if len(self.calibration_samples) == self.n_calibration_samples:
                    self.qoffset = np.mean(self.calibration_samples, axis=0)
                    self.vstart = rotate_v(self.MARKER_AXIS_UP, self.qoffset)
                    self.calibrated = True
                    print(f"calibration for cam {self.cam_name} done")

        self.l.acquire()
        self.markers.append(markers)
        self.l.release()

class StateEstimator:

    def __init__(self, cams = [], max_age = rospy.Duration(2)):
        self.max_age = max_age
        self.angle_pub = rospy.Publisher("/normal_angle", Float64, queue_size=1)
        self.calib_srv = rospy.Service("object_state_calibration", Empty, self.calibrate_cb)

        self.tts = []
        for cam in cams:
            self.tts.append(TagTransformator(cam))

        self.calibrated = False
        self.br = tf.TransformBroadcaster()

    def calibrate_cb(self, *args, **kwargs):
        print("calibrating SE ...")
        for t in self.tts: t.calibrate()

        time.sleep(3)
        
        self.calibrated = np.any([tt.calibrated for tt in self.tts])
        print("SE calibration done:", self.calibrated, [tt.calibrated for tt in self.tts])
        
        return EmptyResponse()

    def process(self):
        latest = rospy.Time.now()-self.max_age

        tfs = {}
        angs = []
        final_ang = []
        for tt in self.tts: # loop over M cams ...

            angs.append([])
            tt.l.acquire()

            for m in tt.markers:            # ... and their last N detections ...
                for mid, v in m.items():    # ... and each marker per detection
                    if v[0] < latest: continue # ignore old markers

                    # store & publish tf for this marker id if none is present 
                    if mid not in tfs: 
                        tfs.update({mid: v[1]})
                        self.br.sendTransform(v2l(v[1].transform.translation), q2l(v[1].transform.rotation), v[1].header.stamp, f"tag_{mid}", v[1].header.frame_id)
                        
                    # store angle if one was calculated
                    if v[2] is not None:
                        angs[-1].append(v[2])
            tt.l.release()
            if len(angs[-1])>0:final_ang.append(np.mean(angs[-1]))
        final_ang = np.mean(final_ang)

        if self.calibrated: self.angle_pub.publish(final_ang)

if __name__ == "__main__":
    rospy.init_node("object_state_estimation")
    se = StateEstimator(["external_webcam", "webcam"])
    se.calibrate_cb()
    
    r = rospy.Rate(30)
    while not rospy.is_shutdown():
        se.process()
        r.sleep()