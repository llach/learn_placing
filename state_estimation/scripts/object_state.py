#!/usr/bin/python
from collections import deque
import tf
import time
import rospy
import numpy as np

from threading import Lock
from std_msgs.msg import Float64, String
from std_srvs.srv import Empty, EmptyResponse
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from tf.transformations import unit_vector, quaternion_multiply, quaternion_conjugate, quaternion_inverse, quaternion_slerp

from state_estimation.msg import ObjectStateEstimate

def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return np.squeeze(a / np.expand_dims(l2, axis))

def v2l(v):
    return np.array([v.x, v.y, v.z])

def q2l(q):
    return normalize(np.array([q.x, q.y, q.z, q.w]))

def vecs2quat(u, v):
    theta = np.dot(u,v) + np.sqrt(np.sqrt(np.linalg.norm(u) * np.linalg.norm(v)))
    q = np.concatenate([np.cross(u,v), [theta]])
    return normalize(q)

def rotate_v(v, q):
    v = list(unit_vector(v))
    v.append(0.0) # vector as pure quaternion, i.e. normalized and 4D
    return quaternion_multiply(
        quaternion_multiply(q, v),
        quaternion_conjugate(q)
    )[:3]

def qavg(quats):
    if len(quats)==1: return quats[0]

    n = len(quats)
    q = quats[0]
    for i in range(1, n):
        q = quaternion_slerp(q, quats[i], 1/n)
    return normalize(q)

def mid2axis(mid):
    if mid == 9:
        return [0,0,1]
    elif mid == 8:
        return [0,0,-1]
    else:
        return [0,-1,0]


class MarkerDetection:

    def __init__(self, mid, timestamp, transform):
        self.mid = mid
        self.timestamp = timestamp
        self.transform = transform

        # only set if we have calibration data
        self.angle = None
        self.voffset = None
        self.vcurrent = None


class TagTransformator:

    def __init__(self, cam_name, n_samples = 10, common_frame="camera_link", n_calibration_samples = 20):
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
        if nmarkers == 0: 
            self.l.acquire()
            self.markers.append(None)
            self.l.release()
            return

        # assumption: all markers are orientated the same way on the block
        # -> knowing the offset from one marker-Z will be the same for other markers

        timestamp = am.header.stamp
        markers = {}
        for d in am.detections:
            mid = d.id[0]
            # if mid == 9 or mid == 8: continue
            axis = mid2axis(mid)

            tfs = TransformStamped()
            tfs.header = am.header
            tfs.child_frame_id = f"tag_{mid}_{self.cam_name}"
            tfs.transform.translation = d.pose.pose.pose.position
            tfs.transform.rotation = d.pose.pose.pose.orientation

            m =  MarkerDetection(mid=mid, timestamp=timestamp, transform=tfs) 

            qcurrent = q2l(tfs.transform.rotation)
            vcurrent = rotate_v(axis, qcurrent)

            if self.calibrated:
                rad = np.dot(self.vstart, vcurrent)

                m.angle = rad
                m.voffset = self.vstart
                m.vcurrent = vcurrent
            
            if self.should_calibrate:
                self.calibration_samples.append(vcurrent)

                if len(self.calibration_samples) == self.n_calibration_samples:
                    self.vstart = normalize(np.mean(self.calibration_samples, axis=0))
                    self.calibrated = True
                    print(f"calibration for cam {self.cam_name} done")
            markers.update({m.mid:m})

        self.l.acquire()
        self.markers.append(markers)
        self.l.release()

class StateEstimator:

    def __init__(self, cams = [], max_age = rospy.Duration(2), n_calibration_samples = 50):
        self.max_age = max_age
        self.ose_pub = rospy.Publisher("/object_state_estimate", ObjectStateEstimate, queue_size=1)
        self.calib_srv = rospy.Service("/object_state_calibration", Empty, self.calibrate_cb)

        self.tts = []
        for cam in cams:
            self.tts.append(TagTransformator(cam, n_calibration_samples=n_calibration_samples))

        self.calibrated = False
        self.li = tf.TransformListener()
        self.br = tf.TransformBroadcaster()

        self.vref = None

    def calibrate_cb(self, *args, **kwargs):
        print("calibrating SE ...")
        for t in self.tts: t.calibrate()

        time.sleep(3)
        
        self.calibrated = np.any([tt.calibrated for tt in self.tts])
        print("SE calibration done:", self.calibrated, [tt.cam_name for tt in self.tts], [tt.calibrated for tt in self.tts])
        
        return EmptyResponse()

    def process(self):
        latest = rospy.Time.now()-self.max_age

        tfs = []

        angles = []
        voffsets = []
        vcurrents = []
        cameras = []

        self.vref = None

        for tt in self.tts: # loop over M cams ...

            angles.append([])
            vcurrents.append([])
            voffsets.append([])

            tt.l.acquire()
            for m in tt.markers:            # ... and their last N detections ...
                if m == None: continue      # ... skip non-detections ...
                for mid, md in m.items():    # ... and each marker per detection
                    if md.timestamp < latest: continue # ignore old markers
                    if tt.cam_name not in cameras: cameras.append(tt.cam_name)
                    # store & publish tf for this marker id if none is present 
                    if mid not in tfs: 
                        tfs.append(mid)
                        self.br.sendTransform(
                            v2l(md.transform.transform.translation), 
                            q2l(md.transform.transform.rotation), 
                            rospy.Time.now(), 
                            f"tag_{mid}", 
                            md.transform.header.frame_id)
                        
                    # store angle if one was calculated
                    if md.angle is not None:
                        angles[-1].append(md.angle)
                        vcurrents[-1].append(md.vcurrent)
                        voffsets[-1].append(md.voffset)
                    
                        
            tt.l.release()
        if np.any([len(a)>0 for a in angles]):
            mean_angs = [np.mean(a) for a in angles if len(a)>0]
            final_ang = np.mean(mean_angs)
        else:
            return
        
        ose = ObjectStateEstimate()

        vcmeans = [normalize(np.mean(v, axis=0)) for v in vcurrents if len(v)>0]
        voffs = [v[0] for v in voffsets if len(v)>0]

        ose.vcurrents = [Vector3(*v) for v in vcmeans]
        ose.voffsets = [Vector3(*v) for v in voffs]

        ose.angle = Float64(final_ang)
        ose.angles = [Float64(ma) for ma in mean_angs]

        ose.cameras = [String(ca) for ca in cameras]

        if len(ose.angles)>0:
            qps = [vecs2quat(vc, vo) for vc, vo in zip(vcmeans, voffs)]
            qp = qavg(qps)

            otf = TransformStamped()
            otf.header.frame_id = "base_link"
            otf.child_frame_id = "object"
            otf.transform.rotation = Quaternion(*qp)
            otf.transform.translation = Vector3(0,0,0)

            ose.transform = otf
            self.br.sendTransform(
                            v2l(otf.transform.translation), 
                            q2l(otf.transform.rotation), 
                            rospy.Time.now(),
                            "object", 
                            "base_link")

            try:
                (_,rotG) = self.li.lookupTransform('/base_link', '/gripper_grasping_frame', rospy.Time(0))
                rotGO = quaternion_multiply(qp, quaternion_inverse(rotG))
                rotGO = normalize(rotGO)

                self.br.sendTransform(
                            [0,0,0], 
                            rotGO,
                            rospy.Time.now(), 
                            "grasped_object", 
                            "gripper_grasping_frame")

            except (tf.LookupException, tf.ConnectivityException):
                print("unable to find gripper transform!")

        if self.calibrated: self.ose_pub.publish(ose)

if __name__ == "__main__":
    rospy.init_node("object_state_estimation")
    se = StateEstimator(["cam1", "cam2", "cam3"])
    se.calibrate_cb()
    
    r = rospy.Rate(60)
    while not rospy.is_shutdown():
        se.process()
        r.sleep()