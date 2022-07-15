from time import time
import rospy

from std_msgs.msg import Bool, Time
from std_srvs.srv import Empty

class ContactDetector:

    def __init__(self) -> None:
        self.initialized = False

        self.in_contact = False
        self.contact_ts = None

        self.con_sub = rospy.Subscriber("/table_contact/in_contact", Bool, self.contact_cb, queue_size=1)
        self.con_ts_sub = rospy.Subscriber("/table_contact/contact_timestamp", Time, self.contact_ts_cb, queue_size=1)

        self.calibrate_srv = rospy.ServiceProxy("/table_contact/calibrate", Empty)

    def _require_setup(self) -> bool:
        assert self.initialized, "ContactDetector not initialized, did you call setup()?"

    def setup(self, timeout: float = 5) -> bool:
        try:
            self.calibrate_srv.wait_for_service(timeout=timeout)
            self.initialized = True
        except rospy.exceptions.ROSException as e:
            rospy.logerr(f"could not connect to table contact calibration service:\n{e}")
            self.initialized = False
        return self.initialized
        
    def contact_cb(self, m: Bool):
        self.in_contact = m.data

    def contact_ts_cb(self, m: Time):
        self.contact_ts = m if m != rospy.Time(0) else None

    def calibrate(self):
        self._require_setup()
        self.calibrate_srv()
