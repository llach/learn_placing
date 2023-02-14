import rospy

from threading import Lock
from tf import TransformListener
from std_srvs.srv import Empty, EmptyResponse
from state_estimation.msg import ObjectStateEstimate
from learn_placing.processing.bag2pickle import q2l
from execute_placing.placing_planner import PlacingPlanner
from learn_placing.common.transformations import quaternion_matrix

class PlacingOracle:
    """ vision-based placing with (partial?) object state knowledge
    """

    def __init__(self, optitrack=True):
        self.optitrack=optitrack
        self.os_t = None # latest timestamp
        self.os = None
        self.grasping_frame = "gripper_left_grasping_frame"
        self.world_frame = "base_footprint"
        self.oslock = Lock()

        print("creating planner")
        self.planner = PlacingPlanner()

        if not optitrack: 
            self.ossub = rospy.Subscriber("/object_state_estimate", ObjectStateEstimate, self.os_callback)
            print("waiting for object state ...")
            while self.os is None and not rospy.is_shutdown(): rospy.Rate(10).sleep()

        self.alignservice = rospy.Service("/placing_oracle/align", Empty, self.align_object)

        self.li = TransformListener()
        print("waiting for transform to object ...")
        for _ in range(6):
            try:
                self.li.waitForTransform(self.world_frame, "pot", rospy.Time(0), rospy.Duration(3))
                break
            except Exception as e:
                print(e)

        print("placing oracle setup done!")

    def os_callback(self, msg):
        # self.oslock.acquire()

        self.os = msg
        self.os_t = rospy.Time.now()

        # self.oslock.release()

    def align_object(self, _):
        print("aligning object ...")

        done = False
        while not done:
            inp = input("next? a=align; p=place\n")
            inp = inp.lower()

            if self.optitrack:
                try:
                    (_, Qwp) = self.li.lookupTransform(self.world_frame, "pot", rospy.Time(0))
                except Exception as e:
                    print(f"[ERROR] couldn't get pot TF: {e}")
                    return
            if inp == "a":
                if self.optitrack:
                    self.planner.align(quaternion_matrix(Qwp))
                else:
                    # self.oslock.acquire()
                    qdiff = q2l(self.os.finalq)
                    self.planner.align(quaternion_matrix(qdiff))
                    # self.oslock.release()
            elif inp == "p":
                self.planner.place()
            else:
                print("all done, bye")
                done = True
        return EmptyResponse()


if __name__ == "__main__":
    import time
    import rospy
    from sensor_msgs.msg import JointState
    rospy.init_node("placing_oracle")

    # INITIAL_STATE = [0.0, 1.07, 0.04, 1.87, 1.43, 0.28, 0.48, -0.92]
    # ACTIVE_JOINTS = ['torso_lift_joint', 'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint']

    # init_js = {jn: [jp] for jn, jp in zip(ACTIVE_JOINTS, INITIAL_STATE)}


    # joint_pub = rospy.Publisher('/move_group/fake_controller_joint_states', JointState, queue_size=10)
    # print("waiting for js subscriber")
    # while joint_pub.get_num_connections()<1:
    #     time.sleep(0.5)

    # joint_pub.publish(JointState(name=ACTIVE_JOINTS, position=INITIAL_STATE))

    po = PlacingOracle(optitrack=True)
    time.sleep(0.5)

    # r = rospy.Rate(20)
    # while not rospy.is_shutdown(): r.sleep()

    # po.align_object(0)
    # po.align_object(0)
    posrv = rospy.ServiceProxy("/placing_oracle/align", Empty)
    posrv()