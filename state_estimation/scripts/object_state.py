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

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def v2l(v):
    return np.array([v.x, v.y, v.z])

def q2l(q):
    return normalized(np.array([q.x, q.y, q.z, q.w]))[0]

def rotate_v(v, q):
    v = list(unit_vector(v))
    v.append(0.0) # vector as pure quaternion, i.e. normalized and 4D
    return quaternion_multiply(
        quaternion_multiply(q, v),
        quaternion_conjugate(q)
    )[:3]

def qavg(quats):
    n = len(quats)
    q = quats[0]
    for i in range(1, n):
        q = quaternion_slerp(q, quats[i], 1/n)
    return q

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
        self.qoffset = None
        self.voffset = None
        self.qcurrent = None
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
            if mid == 9 or mid == 8: continue
            axis = mid2axis(mid)

            tfs = TransformStamped()
            tfs.header = am.header
            tfs.child_frame_id = f"tag_{mid}_{self.cam_name}"
            tfs.transform.translation = d.pose.pose.pose.position
            tfs.transform.rotation = d.pose.pose.pose.orientation

            m =  MarkerDetection(mid=mid, timestamp=timestamp, transform=tfs) 

            if self.calibrated:
                qcurrent = q2l(tfs.transform.rotation)
                vcurrent = rotate_v(axis, qcurrent)
                rad = np.dot(self.vstart, vcurrent)

                m.angle = rad
                m.voffset = self.vstart
                m.qoffset = self.qoffset
                m.qcurrent = qcurrent
                m.vcurrent = vcurrent
            
            if self.should_calibrate:
                # collect rotations that we use to calculate the 
                self.calibration_samples.append(q2l(tfs.transform.rotation))

                if len(self.calibration_samples) == self.n_calibration_samples:
                    self.qoffset = np.mean(self.calibration_samples, axis=0)
                    # self.qoffset = qavg(self.calibration_samples)
                    self.vstart = rotate_v(axis, self.qoffset)
                    self.qoffset = qavg(self.calibration_samples)
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

        self.qref = None

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
        qoffsets = []
        voffsets = []
        qcurrents = []
        vcurrents = []
        cameras = []

        self.qref = None

        for tt in self.tts: # loop over M cams ...

            angles.append([])
            qcurrents.append([])
            vcurrents.append([])
            qoffsets.append([])
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

                    if self.qref is None: self.qref = md.vcurrent
                        
                    # store angle if one was calculated
                    if md.angle is not None:
                        angles[-1].append(md.angle)
                        vcurrents[-1].append(md.vcurrent)
                        qcurrents[-1].append(md.qcurrent)
                        qoffsets[-1].append(md.qoffset)
                        voffsets[-1].append(md.voffset)
                    print(tt.cam_name, mid, md.qcurrent, md.vcurrent, np.dot(md.vcurrent, self.qref), md.timestamp)
                    # if np.dot(md.qcurrent, self.qref)<0.9:
                        
            tt.l.release()
        if np.any([len(a)>0 for a in angles]):
            mean_angs = [np.mean(a) for a in angles if len(a)>0]
            final_ang = np.mean(mean_angs)
        else:
            return
        
        ose = ObjectStateEstimate()

        # ose.qcurrents = [Quaternion(*np.mean(q, axis=0)) for q in qcurrents if len(q)>0]
        ose.qcurrents = [Quaternion(*qavg(q)) for q in qcurrents if len(q)>0]
        ose.vcurrents = [Vector3(*np.mean(v, axis=0)) for v in vcurrents if len(v)>0]
        ose.qoffsets = [Quaternion(*q[0]) for q in qoffsets if len(q)>0]
        ose.voffsets = [Vector3(*v[0]) for v in voffsets if len(v)>0]

        ose.angle = Float64(final_ang)
        ose.angles = [Float64(ma) for ma in mean_angs]

        ose.cameras = [String(ca) for ca in cameras]

        if len(ose.angles)>0:
            qc  = [q2l(q) for q in ose.qcurrents]
            qo  = [q2l(q) for q in ose.qoffsets]
            qdiffs = [
                quaternion_multiply(c, quaternion_inverse(o)) 
                for c, o in zip(qc, qo)
            ]
            if len(qdiffs)==1: # there is a denormalization error here where |q|=0.98 TODO
                qp = qdiffs[0]
            else:
                print("yeah")
                qp = qavg(qdiffs)

            otf = TransformStamped()
            otf.header.frame_id = "base_link"
            otf.child_frame_id = "object"
            otf.transform.rotation = Quaternion(*qp)
            otf.transform.translation = Vector3(0,0,0)

            # print(q2l(otf.transform.rotation))

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
                rotGO = normalized(rotGO)[0]

                self.br.sendTransform(
                            [0,0,0], 
                            normalized(rotGO)[0],
                            rospy.Time.now(), 
                            "grasped_object", 
                            "gripper_grasping_frame")

            except (tf.LookupException, tf.ConnectivityException):
                print("unable to find gripper transform!")

        if self.calibrated: self.ose_pub.publish(ose)

if __name__ == "__main__":
    rospy.init_node("object_state_estimation")
    se = StateEstimator(["aukey", "webcam"])
    se.calibrate_cb()
    
    r = rospy.Rate(60)
    while not rospy.is_shutdown():
        se.process()
        r.sleep()