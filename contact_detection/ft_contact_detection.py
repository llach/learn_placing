import rospy
import numpy as np

from collections import deque
from std_msgs.msg import Bool, Time
from std_srvs.srv import Empty, EmptyResponse
from geometry_msgs.msg import WrenchStamped

"""
Topics & Parameters
"""
FT_TOPIC = "/wrist_ft"
BASE_TOPIC = "/table_contact"
CON_TOPIC = f"{BASE_TOPIC}/in_contact"
CON_TS_TOPIC = f"{BASE_TOPIC}/contact_timestamp"
CALIBRATE_SERVER_TOPIC = f"{BASE_TOPIC}/calibrate"

N_CALIBRATION_SAMPLES = 150
M_SAMPLES = 5
STD_TRESH = 5

"""
global variables
"""
calibration_samples = []
calibrated = False

means = None
stds = None

con_pub = None
con_ts_pub = None

delta_ws = deque(maxlen=M_SAMPLES)

r = None

def wrench_to_vec(w: WrenchStamped):
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
    global con_ts_pub

    in_contact = False

    if not calibrated:
        if calibration_samples == []:
            print("starting calibration ...")

        if len(calibration_samples)<N_CALIBRATION_SAMPLES:
            calibration_samples.append(wrench_to_vec(m.wrench))
        else:
            means = np.mean(calibration_samples, axis=0)
            stds = np.std(calibration_samples, axis=0)
            calibrated = True
            print("calibration done!")
    else:
        delta_ws.append(np.abs(wrench_to_vec(m.wrench)-means))
        if len(delta_ws)>=M_SAMPLES:
            delta_w = np.median(delta_ws, axis=0) # we take the median over N_SAMPLES samples to avoid false-positives due to outliers
            for i, (v, std) in enumerate(zip(delta_w, stds)):
                if v>STD_TRESH*std:
                    print(f"element {i:.4f} detected contact ({v:.4f} > {STD_TRESH*std})")
                    in_contact = True

    if in_contact:
        con_pub.publish(True)
        con_ts_pub.publish(rospy.Time.now())
    else:
        con_pub.publish(False)
        con_ts_pub.publish(rospy.Time(0))

def reset_calibration(*args, **kwargs):
    global r
    global means
    global calibrated
    global calibration_samples

    print(f"current means: {means}")
    calibration_samples = []
    calibrated = False
    while not calibrated:
        r.sleep()
    print(f"new means: {means}")

    return EmptyResponse()

rospy.init_node("ft_contact_detection")
r = rospy.Rate(50)

ft_sub = rospy.Subscriber(FT_TOPIC, WrenchStamped, ft_cb, queue_size=1)
con_pub = rospy.Publisher(CON_TOPIC, Bool, queue_size=1)
con_ts_pub = rospy.Publisher(CON_TS_TOPIC, Time, queue_size=1)
calibrate_srv = rospy.Service(CALIBRATE_SERVER_TOPIC, Empty, reset_calibration)

while not rospy.is_shutdown():
    r.sleep()
