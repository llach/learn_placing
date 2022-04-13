import tf
import rospy
import numpy as np

from std_msgs.msg import Header
from tf.transformations import quaternion_inverse, quaternion_matrix
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped, Pose

ppub = None
tfb = None
tfl = None
calibrated = False


def marker_cb(m: AprilTagDetectionArray):
    global tfb 
    global tfl
    global ppub

    _, ccol2cl_L = tfl.lookupTransform("camera_link", "tag_0", rospy.Time(0))

    if len(m.detections) > 0 and ccol2cl_L:
       
        CCOL2CL = quaternion_matrix(np.array(ccol2cl_L))[0:3,0:3]

        n0_in_CL = CCOL2CL@np.array([0,-1,0])

        p = PoseStamped()
        p.header.frame_id = "camera_link"
        p.header.stamp = rospy.Time.now()

        p.pose.position.x = n0_in_CL[0]
        p.pose.position.y = n0_in_CL[1]
        p.pose.position.z = n0_in_CL[2]

        p.pose.orientation.x = 0
        p.pose.orientation.y = 0
        p.pose.orientation.z = 0
        p.pose.orientation.w = 0

        ppub.publish(p)

        print(n0_in_CL, n0_in_CL.shape)
        pass
    

rospy.init_node("object_state_estimation")
rospy.Subscriber("/tag_detections", AprilTagDetectionArray, marker_cb)
ppub = rospy.Publisher("/table_pose", PoseStamped, queue_size=1)

tfb = tf.TransformBroadcaster(queue_size=1)
tfl = tf.TransformListener()

rospy.spin()