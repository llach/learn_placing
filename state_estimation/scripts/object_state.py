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
from tf.transformations import unit_vector, quaternion_multiply, quaternion_conjugate, quaternion_inverse, quaternion_slerp, quaternion_about_axis, quaternion_matrix, inverse_matrix, quaternion_from_matrix

from state_estimation.msg import ObjectStateEstimate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import numpy as np

class Arrow3D(FancyArrowPatch):

    def __init__(self, base, head, mutation_scale=20, lw=4, arrowstyle="-|>", color="r", **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), mutation_scale=mutation_scale, lw=lw, arrowstyle=arrowstyle, color=color, **kwargs)
        self._verts3d = list(zip(base,head))

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


x_ = [1,0,0]
y_ = [0,1,0]
z_ = [0,0,1]

Qx = lambda a: quaternion_about_axis(a, x_)
Qy = lambda a: quaternion_about_axis(a, y_)
Qz = lambda a: quaternion_about_axis(a, z_)

def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return np.squeeze(a / np.expand_dims(l2, axis))

def Tf2T(pos, rot):
    T = quaternion_matrix(rot)
    T[:3,3] = pos
    return T

def T2Tf(T):
    return T[:3,3], normalize(quaternion_from_matrix(T))

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

def mid2q(mid):
    if mid == 9:
        return Qy(-np.pi/2)
    elif mid == 8:
        return Qy(np.pi/2)
    else:
        return Qz(-np.pi/2)

class MarkerDetection:

    def __init__(self, mid, timestamp, transform):
        self.mid = mid
        self.timestamp = timestamp
        self.transform = transform

        # only set if we have calibration data
        self.angle = None
        self.voffset = None
        self.vcurrent = None
        self.qoffset = None
        self.qcurrent = None

TABLE_FRAME = "table"

class TagTransformator:


    def __init__(
        self, 
        cam_name, 
        n_samples = 10, 
        common_frame="camera_link", 
        n_calibration_samples = 20,
        publish_marker_tfs = True,
        publish_object_tf = True
    ):
        self.cam_name = cam_name
        self.n_samples = n_samples
        self.common_frame = common_frame
        self.publish_marker_tfs = publish_marker_tfs
        self.publish_object_tf = publish_object_tf
        self.tag_topic = f"/tag_detections_{cam_name}"

        self.calibrated = False
        self.should_calibrate = False
        self.calibration_samples = []
        self.n_calibration_samples = n_calibration_samples

        self.T = None
        self.Tinv = None
        self.stamp = None
        self.dist = None
        self.qoffset = None
        self.qcurrent = None
        self.voffset = None
        self.vcurrent = None
        self.translation = None

        self.l = Lock()
        self.br = tf.TransformBroadcaster()

        self.tag_sub = rospy.Subscriber(self.tag_topic, AprilTagDetectionArray, self.tag_cb, queue_size=1)

    def calibrate(self):
        self.T = None
        self.Tinv = None
        self.stamp = None
        self.dist = None
        self.qoffset = None
        self.qcurrent = None
        self.voffset = None
        self.vcurrent = None
        self.toffset = None
        self.tcurrent = None

        self.calibrated = False
        self.should_calibrate = True
        self.calibration_samples = []

        print(f"staring calibration for {self.cam_name}")

    def tag_cb(self, am: AprilTagDetectionArray):
        nmarkers = len(am.detections)
        if nmarkers == 0:
            self.stamp  = None
            return

        qs = {}
        vs = {}
        qbase = {}
        translations = []
        for d in am.detections:
            mid = d.id[0]
            qaxis = mid2q(mid)

            markerq = q2l(d.pose.pose.pose.orientation)
            pnormalq = normalize(quaternion_multiply(markerq, qaxis))

            qs.update({mid:pnormalq})
            vs.update({mid:rotate_v([1,0,0], pnormalq)})
            qbase.update({mid: markerq})

            translations.append(v2l(d.pose.pose.pose.position))

            if self.calibrated and self.publish_marker_tfs:
                self.br.sendTransform(
                            translations[-1], 
                            markerq, 
                            rospy.Time.now(),
                            f"tag_{mid}_{self.cam_name}", 
                            self.cam_name)

        meanq = qavg(list(qs.values()))
        meant = np.mean(translations, axis=0)
        vcurrent = rotate_v([1,0,0], meanq)
        
        self.l.acquire()
        if self.should_calibrate:
            self.calibration_samples.append([meanq, meant])

            if len(self.calibration_samples) == self.n_calibration_samples:
                self.calibration_samples = np.array(self.calibration_samples)

                self.qoffset = qavg(self.calibration_samples[:,0])
                self.toffset = np.mean(self.calibration_samples[:,1])
                self.voffset = vcurrent

                # this is the camera in table coordinates
                self.T = quaternion_matrix(self.qoffset)
                self.T[:3,3] = self.toffset
                self.Tinv = inverse_matrix(self.T)

                self.calibrated = True
                self.should_calibrate = False
                print(f"calibration for {self.cam_name} done")
        
        if self.calibrated:
            self.stamp = am.header.stamp

            self.qcurrent = meanq
            self.tcurrent = meant
            self.vcurrent = vcurrent
            self.angle = np.dot(vcurrent, self.voffset)
            self.dist = np.linalg.norm(self.tcurrent)

            if self.publish_object_tf:
                self.br.sendTransform(
                            self.tcurrent, 
                            self.qcurrent, 
                            rospy.Time.now(),
                            f"object_{self.cam_name}", 
                            self.cam_name)

        self.l.release()

    def publish_cam_tf(self):
        if self.Tinv is None: return
        self.br.sendTransform(
                        self.Tinv[:3,3], 
                        normalize(quaternion_from_matrix(self.Tinv)), 
                        rospy.Time.now(),
                        self.cam_name,
                        TABLE_FRAME
                    )

class StateEstimator:

    HEAD_CAM_FRAME = 'xtion_rgb_optical_frame'

    def __init__(self, cams = [], max_age = rospy.Duration(2), n_calibration_samples = 50, should_plot=False):
        self.max_age = max_age
        self.should_plot = should_plot

        self.ose_pub = rospy.Publisher("/object_state_estimate", ObjectStateEstimate, queue_size=1)
        self.calib_srv = rospy.Service("/object_state_calibration", Empty, self.calibrate_cb)

        self.tts = []
        for cam in cams:
            self.tts.append(TagTransformator(cam, n_calibration_samples=n_calibration_samples, publish_marker_tfs=False))
        self.head_tt = TagTransformator("head", n_calibration_samples=n_calibration_samples, publish_marker_tfs=False)

        self.calibrated = False
        self.li = tf.TransformListener()
        self.br = tf.TransformBroadcaster()

        for _ in range(5):
            try:
                if self.li.canTransform("base_footprint", self.HEAD_CAM_FRAME, rospy.Time(0)): break
                print("got head transform")
                break
            except:
                pass

        self.Ttable = None
        self.table_pos = []
        self.table_rot = []

        self.colors = [
            np.array([217,  93,  57])/255,
            np.array([239, 203, 104])/255,
            np.array([180, 159, 204])/255
        ][:len(self.tts)]

        self.has_plot = False

    def __del__(self):
        if self.has_plot:
            plt.close(self.fig)

    def calculate_Ttable(self):
        head2table = self.head_tt.T
        base2head = Tf2T(*self.li.lookupTransform("base_footprint", self.HEAD_CAM_FRAME, rospy.Time(0)))
        self.Ttable = base2head@head2table
        self.table_pos, self.table_rot = T2Tf(self.Ttable)
        pass

    def calibrate_cb(self, *args, **kwargs):
        print("calibrating SE ...")
        for t in self.tts: t.calibrate()
        self.head_tt.calibrate()

        time.sleep(3)
        
        self.calibrated = np.any([tt.calibrated for tt in self.tts])
        print("SE calibration done:", self.calibrated, [tt.cam_name for tt in self.tts], [tt.calibrated for tt in self.tts])
        if self.head_tt.calibrated:
            print("head calib successful")
            self.calculate_Ttable()
        else:
            print("!!! HEAD CALIBRATION FAILED !!!")
        return EmptyResponse()

    def process(self):
        latest = rospy.Time.now()-self.max_age
        calibrated_tts = [tt for tt in self.tts if tt.calibrated]

        if self.Ttable is not None:
            self.br.sendTransform(
                            self.table_pos, 
                            self.table_rot, 
                            rospy.Time.now(),
                            "table",
                            "base_footprint")

        if len(calibrated_tts)==0:
            print("warn: no cam calibrated")
            return

        angles = []
        qoffsets = []
        qcurrents = []
        voffsets = []
        vcurrents = []
        toffsets = []
        tcurrents = []
        cameras = []

        for tt in self.tts: # loop over M cams ...
            tt.publish_cam_tf()
            if tt.stamp is not None and tt.stamp>latest:
                angles.append(tt.angle)
                qoffsets.append(tt.qoffset)
                qcurrents.append(tt.qoffset)
                voffsets.append(tt.voffset)
                vcurrents.append(tt.vcurrent)
                toffsets.append(tt.toffset)
                tcurrents.append(tt.tcurrent)

                cameras.append(tt.cam_name)
            
        if len(angles)==0: 
            print("warn: no detection")
            return

        vdiffs = [list(vo-vc) for vo, vc in zip(voffsets, vcurrents)]
        qdiffs = [vecs2quat(vo, vc) for vo, vc in zip(voffsets, vcurrents)]
        vdlens = [np.linalg.norm(vd) for vd in vdiffs]
        # print(vdiffs)
        # print(vdlens)
        # print(angles)

        if self.should_plot:

            """
            live plotting: https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
            """
            if not self.has_plot:
                self.fig = plt.figure(figsize=(9.71, 8.61))
                self.ax = self.fig.add_subplot(111, projection='3d')
                
                plt.show(block=False)
                plt.draw()

                self.has_plot = True
            
            self.ax.clear()

            alim = [-1.2, 1.2]
            self.ax.set_xlim(alim)
            self.ax.set_ylim(alim)
            self.ax.set_zlim(alim)
            
            aalph = 0.9
            self.ax.add_artist(Arrow3D([0,0,0], [1,0,0], color=[1.0, 0.0, 0.0, aalph]))
            self.ax.add_artist(Arrow3D([0,0,0], [0,1,0], color=[0.0, 1.0, 0.0, aalph]))
            self.ax.add_artist(Arrow3D([0,0,0], [0,0,1], color=[0.0, 0.0, 1.0, aalph]))

            handles = []
            for vo, vc, col, tt in zip(voffsets, vcurrents, self.colors, calibrated_tts):
                self.ax.add_artist(Arrow3D([0,0,0], vo, color=list(col)+[0.7]))
                handles.append(self.ax.add_artist(Arrow3D([0,0,0], vc, color=list(col)+[1.0], label=f"{tt.cam_name} (d={tt.dist:.2f})"))
                )
            
            handles.append(
                self.ax.add_artist(Arrow3D([0,0,0], [0,0,-1], color=[0.0, 1.0, 1.0, 0.7], label="desired normal"))
            )

            for qd, tt, cosa in zip(qdiffs, calibrated_tts, angles):
                v = rotate_v([0,0,-1], qd)
                handles.append(
                    self.ax.add_artist(Arrow3D([0,0,0], v, color=[0.0, 1.0, 1.0, 1.0], label=f"{tt.cam_name} normal; cos(a)={cosa:.3f}"))
                )

            self.ax.legend(handles=handles)
            
            try:
                self.fig.tight_layout()
                self.fig.canvas.draw()
                plt.pause(0.05)
            except:
                print("plot closed")
                self.should_plot = False

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
            otf.header.stamp = rospy.Time.now()
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