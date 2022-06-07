
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl, MarkerArray
from interactive_markers.interactive_marker_server import InteractiveMarkerFeedback
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from tf import transformations as tf
import numpy

def cylinder(radius=0.02, len=0.1, color=ColorRGBA(1, 0, 0, 1), **kwargs):
    """Create a cylinder marker"""
    scale = Vector3(radius, radius, len)
    return Marker(type=Marker.CYLINDER, scale=scale, color=color, **kwargs)

def box(size=Vector3(0.1, 0.1, 0.1), color=ColorRGBA(1, 1, 1, 0.5), **kwargs):
    """Create a box marker"""
    return Marker(type=Marker.CUBE, scale=size, color=color, **kwargs)

def frame(T, scale=0.1, radius=None, frame_id='base_footprint', ns='frame', alpha=1.0):
    """Create a frame composed from three cylinders"""
    markers = []
    p = T[0:3, 3]

    defaults = dict(header=Header(frame_id=frame_id), ns=ns)
    if radius is None:
        radius = scale / 10

    xaxis = tf.quaternion_about_axis(numpy.pi / 2., [0, 1, 0])
    yaxis = tf.quaternion_about_axis(numpy.pi / 2., [-1, 0, 0])
    offset = numpy.array([0, 0, scale / 2.])

    m = cylinder(radius, scale, color=ColorRGBA(1, 0, 0, alpha), id=0, **defaults)
    q = tf.quaternion_multiply(tf.quaternion_from_matrix(T), xaxis)
    m.pose.orientation = Quaternion(*q)
    m.pose.position = Point(*(p + tf.quaternion_matrix(q)[:3, :3].dot(offset)))
    markers.append(m)

    m = cylinder(radius, scale, color=ColorRGBA(0, 1, 0, alpha), id=1, **defaults)
    q = tf.quaternion_multiply(tf.quaternion_from_matrix(T), yaxis)
    m.pose.orientation = Quaternion(*q)
    m.pose.position = Point(*(p + tf.quaternion_matrix(q)[:3, :3].dot(offset)))
    markers.append(m)

    m = cylinder(radius, scale, color=ColorRGBA(0, 0, 1, alpha), id=2, **defaults)
    m.pose.orientation = Quaternion(*tf.quaternion_from_matrix(T))
    m.pose.position = Point(*(p + T[:3, :3].dot(offset)))
    markers.append(m)
    return markers
