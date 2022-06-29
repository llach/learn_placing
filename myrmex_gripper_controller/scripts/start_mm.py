#!/usr/bin/python
import time
import rospy
from experiment_manager.common.cm_interface import safe_switch, is_running
rospy.init_node("start_mm")
# cm service is awaited in one of these methods
while not is_running("gripper_controller"): time.sleep(0.5)
safe_switch("gripper_controller", "myrmex_gripper_controller")

print("############################################")
print("######### MyrmexController started! ########")
print("############################################")