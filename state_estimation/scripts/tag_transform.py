import tf
import rospy
import numpy as np

from visualization_msgs.msg import Marker
from tf.transformations import unit_vector, quaternion_multiply, quaternion_conjugate
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped, Point

def rotate_v(v, q):
    v = list(unit_vector(v))
    v.append(0.0) # vector as pure quaternion, i.e. normalized and 4D
    return quaternion_multiply(
        quaternion_multiply(q, v),
        quaternion_conjugate(q)
    )[:3]

mpub = None
ppub = None
tfb = None
tfl = None
calibrated = False


m_axis_up = [0,-1,0]


def marker_cb(am: AprilTagDetectionArray):
    global tfb 
    global tfl
    global ppub
    global mpub

    _, Qtag2cam = tfl.lookupTransform("camera_link", "tag_0", rospy.Time(0))

    m = Marker()
    m.header.frame_id = "camera_link"
    m.type = Marker.ARROW
    m.action = Marker.ADD

    v = rotate_v(m_axis_up,Qtag2cam)
    print(np.linalg.norm(v))

    m.points = [
        Point(0,0,0),
        Point(*(0.2*v))
    ]

    m.scale.x = 0.03
    m.scale.y = 0.05

    m.color.r = 1.0
    m.color.g = 110/255
    m.color.b = 199/255
    m.color.a = 1.0

    mpub.publish(m)

rospy.init_node("object_state_estimation")
rospy.Subscriber("/tag_detections", AprilTagDetectionArray, marker_cb)
ppub = rospy.Publisher("/table_pose", PoseStamped, queue_size=1)
mpub = rospy.Publisher("/arrow", Marker, queue_size=1)

tfb = tf.TransformBroadcaster(queue_size=1)
tfl = tf.TransformListener()

rospy.spin()