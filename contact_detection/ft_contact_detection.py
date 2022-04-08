import rospy
import numpy as np

from collections import deque
from std_msgs.msg import Bool, Time
from geometry_msgs.msg import WrenchStamped

FT_TOPIC = "/wrist_ft"
BASE_TOPIC = "/table_contact"
CON_TOPIC = f"{BASE_TOPIC}/in_contact"
CON_TS_TOPIC = f"{BASE_TOPIC}/contact_timestamp"

N_CALIBRATION_SAMPLES = 150
N_SAMPLES = 5
STD_TRESH = 8

calibration_samples = []
calibrated = False

means = None
stds = None

con_pub = None
con_ts_pub = None

delta_ws = deque(maxlen=N_SAMPLES)

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
        if len(delta_ws)>=N_SAMPLES:
            delta_w = np.median(delta_ws, axis=0) # we take the median over N_SAMPLES samples to avoid false-positives due to outliers
            for i, (v, std) in enumerate(zip(delta_w, stds)):
                if v>STD_TRESH*std:
                    print(f"element {i} detected contact ({v} > {STD_TRESH*std})")
                    in_contact = True

    if in_contact:
        con_pub.publish(True)
        con_ts_pub.publish(rospy.Time.now())
    else:
        con_pub.publish(False)
        con_ts_pub.publish(rospy.Time(0))


rospy.init_node("ft_contact_detection")

ft_sub = rospy.Subscriber(FT_TOPIC, WrenchStamped, ft_cb, queue_size=1)
con_pub = rospy.Publisher(CON_TOPIC, Bool, queue_size=1)
con_ts_pub = rospy.Publisher(CON_TS_TOPIC, Time, queue_size=1)

while not rospy.is_shutdown():
    pass
