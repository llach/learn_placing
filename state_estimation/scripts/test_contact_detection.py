#!/usr/bin/env python
import rospy

from std_msgs.msg import Header, Time
from std_srvs.srv import Empty, EmptyResponse
from state_estimation.msg import BoolHead

def s(x): return EmptyResponse()

rospy.init_node("test_contact_detection")

con_pub = rospy.Publisher("/table_contact/in_contact", BoolHead, queue_size=1)
con_ts_pub = rospy.Publisher("/table_contact/contact_timestamp", Time, queue_size=1)
calibrate_srv = rospy.Service("/table_contact/calibrate", Empty, s)

print("waiting for connections")
while con_pub.get_num_connections() < 1: rospy.Rate(10).sleep()

now = rospy.Time.now()
con_pub.publish(BoolHead(header=Header(stamp=now), in_contact=False))
con_ts_pub.publish(Time(now))

print("ready to take commands")
while not rospy.is_shutdown():
    a = input()
    if a.lower() == "q": break

    now = rospy.Time.now()
    print("publishing True")
    for _ in range(5):
        con_pub.publish(BoolHead(header=Header(stamp=now), in_contact=True))
        con_ts_pub.publish(Time(now))
        rospy.Rate(100).sleep()

    now = rospy.Time.now()
    con_pub.publish(BoolHead(header=Header(stamp=now), in_contact=False))
    con_ts_pub.publish(Time(now))
