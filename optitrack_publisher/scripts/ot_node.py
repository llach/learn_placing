#!/usr/bin/env python
import rospy 

from optitrack_publisher import OptiTrackPublisher

rospy.init_node("optitrack_publisher")
otp = OptiTrackPublisher()
otp.run()