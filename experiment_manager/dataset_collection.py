import sys
import rospy

from common import TorsoStopController
from common.cm_interface import load_controller, stop_controller, start_controller, is_running

"""
Parameters
"""
N_tries = 100
M = 10

"""
Preparation
"""

if len(sys.argv) >= 2:
    start_pos = float(sys.argv[1])

rospy.init_node("placing_data_collection")

rospy.loginfo("checking torso_stop_controller ...")
if not is_running("torso_stop_controller"):
    # this fails if torso_stop_controller is already running, but that's ok
    load_controller("torso_stop_controller")
    stop_controller("torso_controller")
    start_controller("torso_stop_controller")

    if not is_running("torso_stop_controller"):
        rospy.logfatal("couldn't start torso_stop_controller")
        exit(-1)
rospy.loginfo("torso_stop_controller running!")

rospy.loginfo("setting up TorsoStopController interface ...")
torso = TorsoStopController()
if not torso.setup():
    rospy.logfatal("couldn't setup torso stop controller!")
    exit(-1)

if start_pos is not None:
    rospy.loginfo("Moving to start position")
    torso.move_to(start_pos)
else:
    start_pos = torso.torso_pos

"""
Experiment Loop
"""
for i in range(N_tries):

    # step 1: move down 10cm or until contact
    torso.move_rel(-0.1, duration=10)

    # step 2: move up again
    torso.move_to(start_pos)