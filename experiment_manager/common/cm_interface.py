#! /usr/bin/env python
# Wrappers around the services provided by MechanismControlNode

"""
taken from here:
https://github.com/ros-controls/ros_control/blob/0.5.0/controller_manager/src/controller_manager/controller_manager_interface.py

srv msgs changed, py3 CMI doesn't work ...
"""

import roslib; roslib.load_manifest('controller_manager')

import sys

import rospy
from controller_manager_msgs.srv import *

def list_controller_types():
    rospy.wait_for_service('controller_manager/list_controller_types')
    s = rospy.ServiceProxy('controller_manager/list_controller_types', ListControllerTypes)
    resp = s.call(ListControllerTypesRequest())
    for t in resp.types:
        print(t)

def reload_libraries(force_kill, restore = False):
    rospy.wait_for_service('controller_manager/reload_controller_libraries')
    s = rospy.ServiceProxy('controller_manager/reload_controller_libraries', ReloadControllerLibraries)

    list_srv = rospy.ServiceProxy('controller_manager/list_controllers', ListControllers)
    load_srv = rospy.ServiceProxy('controller_manager/load_controller', LoadController)
    switch_srv = rospy.ServiceProxy('controller_manager/switch_controller', SwitchController)

    print("Restore:", restore)
    if restore:
        originally = list_srv.call(ListControllersRequest())

    resp = s.call(ReloadControllerLibrariesRequest(force_kill))
    if resp.ok:
        print("Successfully reloaded libraries")
        result = True
    else:
        print("Failed to reload libraries. Do you still have controllers loaded?")
        result = False

    if restore:
        for c in originally.controllers:
            load_srv(c)
        to_start = []
        for c, s in zip(originally.controllers, originally.state):
            if s == 'running':
                to_start.append(c)
        switch_srv(start_controllers = to_start,
                   stop_controllers = [],
                   strictness = SwitchControllerRequest.BEST_EFFORT)
        print("Controllers restored to original state")
    return result


def _list_controllers():
    rospy.wait_for_service('controller_manager/list_controllers')
    s = rospy.ServiceProxy('controller_manager/list_controllers', ListControllers)
    resp = s.call(ListControllersRequest())
    return resp.controller

def list_controllers():
    controller = _list_controllers()
    if len(controller) == 0:
        print("No controllers are loaded in mechanism control")
    else:
        for c in controller:
            print('%s - %s ( %s )'%(c.name, c.claimed_resources, c.state))
        return controller


def load_controller(name):
    rospy.wait_for_service('controller_manager/load_controller')
    s = rospy.ServiceProxy('controller_manager/load_controller', LoadController)
    resp = s.call(LoadControllerRequest(name))
    if resp.ok:
        print("Loaded", name)
        return True
    else:
        print("Error when loading", name)
        return False

def unload_controller(name):
    rospy.wait_for_service('controller_manager/unload_controller')
    s = rospy.ServiceProxy('controller_manager/unload_controller', UnloadController)
    resp = s.call(UnloadControllerRequest(name))
    if resp.ok == 1:
        print("Unloaded %s successfully" % name)
        return True
    else:
        print("Error when unloading", name)
        return False

def start_controller(name):
    return start_stop_controllers([name], True)

def start_controllers(names):
    return start_stop_controllers(names, True)

def stop_controller(name):
    return start_stop_controllers([name], False)

def stop_controllers(names):
    return start_stop_controllers(names, False)

def start_stop_controllers(names, st):
    rospy.wait_for_service('controller_manager/switch_controller')
    s = rospy.ServiceProxy('controller_manager/switch_controller', SwitchController)
    start = []
    stop = []
    strictness = SwitchControllerRequest.STRICT
    if st:
        start = names
    else:
        stop = names
    resp = s.call(SwitchControllerRequest(start, stop, strictness))
    if resp.ok == 1:
        if st:
            print("Started %s successfully" % names)
        else:
            print("Stopped %s successfully" % names)
        return True
    else:
        if st:
            print("Error when starting ", names)
        else:
            print("Error when stopping ", names)
        return False

def is_running(name):
    controllers = _list_controllers()
    for c in controllers:
        if c.name == name:
            return c.state == 'running'
    return False