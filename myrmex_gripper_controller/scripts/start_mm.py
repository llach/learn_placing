#!/usr/bin/python
import rospy
from experiment_manager.common.cm_interface import safe_switch
rospy.init_node("start_mm")
# cm service is awaited in one of these methods
safe_switch("gripper_controller", "myrmex_gripper_controller")

print("############################################")
print("######### MyrmexController started! ########")
print("############################################")