import rospy

from tf import TransformListener
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from wrist_motion.marker import frame
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Vector3, Quaternion, Transform, TransformStamped

from execute_placing.placing_planner import Tf2T

class OptiState:

    def __init__(self, publish_markers=True):
        self.publish_markers = publish_markers
        self.world_frame = "base_footprint"
        self.object_frame = "pot"
        self.grasping_frame = "gripper_left_grasping_frame"

        self.otfpub = rospy.Publisher("/opti_state", TFMessage, queue_size=10)
        self.markerpub = rospy.Publisher("/opti_state_markers", MarkerArray, queue_size=10)
        
        self.li = TransformListener()
        for _ in range(6):
            try:
                self.li.waitForTransform(self.world_frame, self.grasping_frame, rospy.Time(0), rospy.Duration(3))
                self.li.waitForTransform(self.world_frame, self.object_frame, rospy.Time(0), rospy.Duration(3))
                self.li.waitForTransform(self.grasping_frame, self.object_frame, rospy.Time(0), rospy.Duration(3))
                break
            except Exception as e:
                print(e)
        print("planner init done")

    def make_tf_stamped(self, t, q, fro, to, stamp):
        return TransformStamped(
            header=Header(stamp=stamp, frame_id=fro),
            child_frame_id=to,
            transform=Transform(
                translation=Vector3(*t), 
                rotation=Quaternion(*q)
            )
        )

    def pub_loop(self):
        now = rospy.Time.now()

        try:
            (twg, Qwg) = self.li.lookupTransform(self.world_frame, self.grasping_frame, rospy.Time(0))
            (two, Qwo) = self.li.lookupTransform(self.world_frame, self.object_frame, rospy.Time(0))
            (tgo, Qgo) = self.li.lookupTransform(self.grasping_frame, self.object_frame, rospy.Time(0))
        except Exception as e:
            print(f"could not get object transforms:\n{e}")
            return

        self.otfpub.publish(TFMessage(
            transforms=[
                self.make_tf_stamped(twg, Qwg, self.world_frame, self.grasping_frame, now),
                self.make_tf_stamped(two, Qwo, self.world_frame, self.object_frame, now),
                self.make_tf_stamped(tgo, Qgo, self.grasping_frame, self.object_frame, now)
            ]
        ))

        if self.publish_markers:
            Twg = Tf2T(twg, Qwg)
            Two = Tf2T(two, Qwo)
            Tgo = Tf2T([0,0,0], Qgo)

            self.markerpub.publish(MarkerArray(markers = [
                *frame(Twg, ns="Twg"),
                *frame(Two, ns="Two"),
                *frame(Twg@Tgo, ns="Tgo")
            ]))

if __name__ == "__main__":
    rospy.init_node("opti_state")
    os = OptiState()
    r = rospy.Rate(10)

    while not rospy.is_shutdown():
        os.pub_loop()
        r.sleep()