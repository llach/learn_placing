import rospy

from threading import Lock
from std_srvs.srv import Empty, EmptyResponse
from state_estimation.msg import ObjectStateEstimate
from learn_placing.common import qO2qdiff, qavg
from learn_placing.processing.bag2pickle import q2l
from execute_placing.placing_planner import PlacingPlanner
from learn_placing.common.transformations import quaternion_matrix, Ry

class PlacingOracle:
    """ vision-based placing with (partial?) object state knowledge
    """

    def __init__(self):
        self.os_t = None # latest timestamp
        self.os = None
        self.oslock = Lock()

        self.planner = PlacingPlanner()

        self.ossub = rospy.Subscriber("/object_state_estimate", ObjectStateEstimate, self.os_callback)
        self.alignservice = rospy.Service("/placing_oracle/align", Empty, self.align_object)

        print("waiting for object state ...")
        while self.os is None and not rospy.is_shutdown(): rospy.Rate(10).sleep()

        print("placing oracle setup done!")

    def os_callback(self, msg):
        self.oslock.acquire()

        self.os = msg
        self.os_t = rospy.Time.now()

        self.oslock.release()

    def align_object(self, _):
        print("aligning object ...")
        self.oslock.acquire()

        qdiff = qO2qdiff(qavg([q2l(q) for q in self.os.qOs]))
        # self.planner.plan_placing(quaternion_matrix(qdiff))
        self.planner.plan_placing(Ry(0.73))

        self.oslock.release()
        return EmptyResponse()


if __name__ == "__main__":
    import time
    import rospy
    from sensor_msgs.msg import JointState

    INITIAL_STATE = [0.0, 1.07, 0.04, 1.87, 1.43, 0.28, 0.48, -0.92]
    ACTIVE_JOINTS = ['torso_lift_joint', 'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint']

    init_js = {jn: [jp] for jn, jp in zip(ACTIVE_JOINTS, INITIAL_STATE)}

    rospy.init_node("placing_oracle")

    joint_pub = rospy.Publisher('/move_group/fake_controller_joint_states', JointState, queue_size=10)
    print("waiting for js subscriber")
    while joint_pub.get_num_connections()<1:
        time.sleep(0.5)

    joint_pub.publish(JointState(name=ACTIVE_JOINTS, position=INITIAL_STATE))

    po = PlacingOracle()
    time.sleep(0.2)
    po.align_object(0)
    # posrv = rospy.ServiceProxy("/placing_oracle/align", Empty)
    # posrv()