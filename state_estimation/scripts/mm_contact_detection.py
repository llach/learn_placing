#!/usr/bin/env python
import rospy
import numpy as np

from threading import Lock
from std_msgs.msg import Time, Header, Float64, Float64MultiArray
from tactile_msgs.msg import TactileState
from std_srvs.srv import Empty, EmptyResponse
from state_estimation.msg import BoolHead


class MMContactDetector:
    N_CALIBRATION_SAMPLES = 50
    M_SAMPLES = 3

    calibrated = False

    BASE_TOPIC = "/table_contact"
    CON_TOPIC = BASE_TOPIC+"/in_contact"
    CON_TS_TOPIC = BASE_TOPIC+"/contact_timestamp"
    CALIBRATE_SERVER_TOPIC = BASE_TOPIC+"/calibrate"

    left_cal, right_cal = [], []
    left_m, right_m = 0, 0
    left_ref, right_ref = None, None
    
    callock = Lock()
    contact_thresh = 0.01

    def __init__(self) -> None:
        self.lsub = rospy.Subscriber("/tactile_left", TactileState, callback=self.lcb, queue_size=1)
        self.rsub = rospy.Subscriber("/tactile_right", TactileState, callback=self.rcb, queue_size=1)

        self.diffpub = rospy.Publisher("/mm_diff", Float64MultiArray, queue_size=1)
        self.threshpub = rospy.Publisher("/mm_diff/tresh", Float64, queue_size=1)

        self.con_pub = rospy.Publisher(self.CON_TOPIC, BoolHead, queue_size=1)
        self.con_ts_pub = rospy.Publisher(self.CON_TS_TOPIC, Time, queue_size=1)

        self.calibrate_srv = rospy.Service(self.CALIBRATE_SERVER_TOPIC, Empty, self.reset_calibration)
        
    def run(self):
            if not self.calibrated:
                with self.callock:
                    self.left_cal.append(self.left_m)
                    self.right_cal.append(self.right_m)
                    if len(self.left_cal)>self.N_CALIBRATION_SAMPLES and len(self.right_cal)>self.N_CALIBRATION_SAMPLES:
                        self.left_ref = np.mean(self.left_cal)
                        self.right_ref = np.mean(self.right_cal)

                        print(f"calibration done! means: {self.left_ref:.4f} | {self.right_ref:.4f}")
                        self.calibrated = True

                now =rospy.Time.now()
                head = Header(stamp=now)
                self.con_pub.publish(BoolHead(header=head, in_contact=False))
                self.con_ts_pub.publish(now)
                
            else:
                ldiff = np.abs(self.left_ref-self.left_m)
                rdiff = np.abs(self.right_ref-self.right_m)
                print(np.round(ldiff, 2), np.round(rdiff, 2))
                in_contact = ldiff>self.contact_thresh or rdiff>self.contact_thresh
                now =rospy.Time.now()
                head = Header(stamp=now)
                if in_contact:
                    print("contact detected!")
                    self.con_pub.publish(BoolHead(header=head, in_contact=True))
                    self.con_ts_pub.publish(now)
                else:
                    self.con_pub.publish(BoolHead(header=head, in_contact=False))
                    self.con_ts_pub.publish(rospy.Time(0))

                self.threshpub.publish(data=self.contact_thresh)
                self.diffpub.publish(data=[
                    ldiff,
                    rdiff,
                ])

    def lcb(self, msg):
        self.left_m = self.mmean(msg)

    def rcb(self, msg):
        self.right_m = self.mmean(msg)

    def mmean(self, msg):
        return 1-(np.mean(msg.sensors[0].values)/4094)

    def reset_calibration(self, _):
        print("recalibrating mm table contact detection ...")
        with self.callock:
            self.left_cal, self.right_cal = [], []
            self.left_ref, self.right_ref = None, None
            self.calibrated = False

        while not self.calibrated:
            rospy.Rate(50).sleep()

        return EmptyResponse()

if __name__ == "__main__":
    rospy.init_node("mm_contact_detection")
    mmd = MMContactDetector()

    r = rospy.Rate(100)

    while not rospy.is_shutdown():
        mmd.run()
        r.sleep()