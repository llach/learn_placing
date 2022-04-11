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

start_pos = None
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
    if rospy.is_shutdown():
        rospy.logwarn("experiment execution was interrupted!")
        break

    rospy.loginfo(f"+++ Trial {i} +++")
    # TODO calibrate

    # step 1: move down 10cm or until contact
    # TODO measure time
    start = rospy.Time.now()
    torso.move_rel(-0.1, duration=10)
    dur = rospy.Time.now() - start
    rospy.loginfo(f"Took {dur.to_sec():.2f}sec to move down")

    # step 2: get on-contact timestep
    # TODO implement contact detection interface

    # step 3: move up again
    # TODO set duration to upward-movement time
    torso.move_to(start_pos)

    rospy.loginfo("")
rospy.loginfo("done.")