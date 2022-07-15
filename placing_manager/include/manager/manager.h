/* Author: Luca Lach
*/

#ifndef PLACING_MANAGER_H
#define PLACING_MANAGER_H

// std includes
#include <mutex>

// misc ros includes
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>

// ros msgs
#include <std_srvs/Empty.h>
#include <sensor_msgs/JointState.h>
#include <controller_manager_msgs/LoadController.h>
#include <controller_manager_msgs/ListControllers.h>
#include <controller_manager_msgs/SwitchController.h>
#include <control_msgs/FollowJointTrajectoryAction.h>

namespace placing_manager {

class PlacingManager{
public:
    PlacingManager(float initialTorsoQ = 0.25);

private:
    std::mutex jsLock_;

    bool initialized_ = false;

    int torsoIdx_ = -1;

    std::string baseFrame_ = "base_footprint";
    std::string torsoJointName_ = "torso_lift_joint";
    std::string torsoControllerName_ = "torso_controller";
    std::string torsoStopControllerName_ = "torso_stop_controller";

    float initialTorsoQ_;
    float currentTorsoQ_ = -1.0;

    ros::NodeHandle n_;

    ros::ServiceClient loadControllerSrv_;
    ros::ServiceClient listControllersSrv_;
    ros::ServiceClient switchControllerSrv_;

    actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> torsoAc_;

    float lerp(float a, float b, float f);
    void moveTorso(float targetQ, float duration, bool absolute = true);
    void jsCallback(const sensor_msgs::JointState::ConstPtr& msg);

    bool isControllerRunning(std::string name);
    bool ensureRunningController(std::string name, std::string stop);
};

} // namespace placing_manager 
#endif  // GAZE_MANAGER_H