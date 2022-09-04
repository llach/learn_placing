from learn_placing.common.label_processing import rotate_v
import rospy
import numpy as np
import learn_placing.common.transformations as tf
from geometry_msgs.msg import Quaternion
from tf import TransformListener, TransformBroadcaster
from state_estimation.msg import ObjectStateEstimate

rospy.init_node("fake_object_state")

grasping_frame = "gripper_left_grasping_frame"
world_frame = "base_footprint"

br = TransformBroadcaster()
li = TransformListener()
li.waitForTransform(grasping_frame, world_frame, rospy.Time(0), rospy.Duration(3))

ospub = rospy.Publisher("/object_state_estimate", ObjectStateEstimate, queue_size=1)

r = rospy.Rate(20)

while not rospy.is_shutdown():
    Qgo = tf.quaternion_about_axis(np.pi+0.43, [0,1,0])

    br.sendTransform(
        [0,0,0], 
        Qgo, 
        rospy.Time.now(),
        "grasped_object", 
        grasping_frame
    )

    TRwg, Qwg = li.lookupTransform(world_frame, grasping_frame, rospy.Time(0))
    Qwo = tf.quaternion_multiply(
        Qwg,
        Qgo
    )

    br.sendTransform(
        [0,0,0], 
        Qwo, 
        rospy.Time.now(),
        "object", 
        world_frame
    )

    Qow = tf.quaternion_inverse(Qwo)

    u = [0,0,1]
    w = rotate_v([0,0,1], Qow)

    axis = np.cross(u,w)
    angle = np.arccos(np.dot(u,w))
    Qcorr = tf.quaternion_about_axis(angle, axis)

    br.sendTransform(
        [0,0,0], 
        Qcorr, 
        rospy.Time.now(),
        "object_corr", 
        "object"
    )

    ose = ObjectStateEstimate()
    ose.finalq = Quaternion(*Qwo)
    ospub.publish(ose)
    r.sleep()