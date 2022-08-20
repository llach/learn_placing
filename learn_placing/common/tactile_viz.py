import rospy

from cv_bridge import CvBridge
from tactile_msgs.msg import TactileState
from sensor_msgs.msg import Image

from learn_placing.common import preprocess_myrmex, mm2img, upscale_repeat

def pub_tactile_img(msg, pub):
    global bridge

    mm = preprocess_myrmex(msg.sensors[0].values)
    mm = upscale_repeat(mm, factor=35)
    img = mm2img(mm)[0]
    imgmsg = bridge.cv2_to_imgmsg(img, encoding="rgb8")
    pub.publish(imgmsg)

rospy.init_node("tactile_img_publisher")
bridge = CvBridge()

left_img_pub = rospy.Publisher("/tactile_left_image", Image, queue_size=1)
rospy.Subscriber("/tactile_left", TactileState, lambda x: pub_tactile_img(x, left_img_pub), queue_size=1)

right_img_pub = rospy.Publisher("/tactile_right_image", Image, queue_size=1)
rospy.Subscriber("/tactile_right", TactileState, lambda x: pub_tactile_img(x, right_img_pub), queue_size=1)

while not rospy.is_shutdown(): rospy.spin()