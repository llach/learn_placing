#!/usr/bin/python
import rospy
import numpy as np

from collections import deque
from std_msgs.msg import Time
from std_srvs.srv import Empty, EmptyResponse
from state_estimation.msg import BoolHead
from geometry_msgs.msg import WrenchStamped, Vector3

"""
Topics & Parameters
"""
FT_TOPIC = "/wrist_ft"
BASE_TOPIC = "/table_contact"
CON_TOPIC = BASE_TOPIC+"/in_contact"
CON_TS_TOPIC = BASE_TOPIC+"/contact_timestamp"
DIFF_TOPIC = BASE_TOPIC+"/diff"
CALIBRATE_SERVER_TOPIC = BASE_TOPIC+"/calibrate"

N_CALIBRATION_SAMPLES = 25
M_SAMPLES = 3
STD_TRESH = 8

"""
global variables
"""
calibration_samples = []
calibrated = False
factor = 1.3

means = None
stds = None

con_pub = None
con_ts_pub = None

dws = deque(maxlen=M_SAMPLES)
ws = deque(maxlen=M_SAMPLES)

r = None

def wrench_to_vec(w):
    return [
        w.force.x,
        w.force.y,
        w.force.z,
    ]

def ft_cb(m):
    global stds
    global means
    global delta_ws
    global calibrated
    global calibration_samples
    
    global con_pub
    global diff_pub
    global con_ts_pub
    global diff_raw_pub

    in_contact = False

    delay = rospy.Time.now()-m.header.stamp
    # print(delay.to_sec())

    if not calibrated:
        if calibration_samples == []:
            print("starting FT calibration ...")

        if len(calibration_samples)<N_CALIBRATION_SAMPLES:
            calibration_samples.append(wrench_to_vec(m.wrench))
        else:
            means = np.mean(calibration_samples, axis=0)
            stds = np.std(calibration_samples, axis=0)
            calibrated = True
            print("FT calibration done!")
    else:
        w = wrench_to_vec(m.wrench)
        diff = np.abs(w-means)
        dws.append(diff)
        diff_raw_pub.publish(Vector3(*diff))
        if len(dws)<M_SAMPLES: return

        diff_pub.publish(Vector3(*np.median(dws, axis=0)))

        if np.any(diff>factor):
            print("contact detected:", diff)
            print(delay.to_sec())
            in_contact = True

    if in_contact:
        con_pub.publish(BoolHead(header=m.header, in_contact=True))
        con_ts_pub.publish(rospy.Time.now())
    else:
        con_pub.publish(BoolHead(header=m.header, in_contact=False))
        con_ts_pub.publish(rospy.Time(0))

def reset_calibration(*args, **kwargs):
    global r
    global means
    global calibrated
    global calibration_samples

    print("current means:",means)
    calibration_samples = []
    calibrated = False
    while not calibrated:
        r.sleep()
    print("new means:",means)

    return EmptyResponse()

rospy.init_node("ft_contact_detection")
r = rospy.Rate(50)

ft_sub = rospy.Subscriber(FT_TOPIC, WrenchStamped, ft_cb, queue_size=1)
con_pub = rospy.Publisher(CON_TOPIC, BoolHead, queue_size=1)
con_ts_pub = rospy.Publisher(CON_TS_TOPIC, Time, queue_size=1)
diff_pub = rospy.Publisher(DIFF_TOPIC, Vector3, queue_size=1)
diff_raw_pub = rospy.Publisher(DIFF_TOPIC+"_raw", Vector3, queue_size=1)
calibrate_srv = rospy.Service(CALIBRATE_SERVER_TOPIC, Empty, reset_calibration)

while not rospy.is_shutdown():
    r.sleep()
