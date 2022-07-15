/* Author: Luca Lach
*/

#ifndef PLACING_MANAGER_H
#define PLACING_MANAGER_H

// misc ros includes
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>

// ros msgs
#include <sensor_msgs/JointState.h>
#include <controller_manager_msgs/LoadController.h>
#include <controller_manager_msgs/ListControllers.h>
#include <controller_manager_msgs/SwitchController.h>
#include <control_msgs/FollowJointTrajectoryAction.h>

namespace placing_manager {

class PlacingManager{
public:
    PlacingManager();

private:
    std::mutex jsLock_;

    int torsoIdx_ = -1;

    std::string baseFrame_ = "base_footprint";
    std::string torsoJointName_ = "torso_lift_joint";

    float currentTorsoQ_ = -1.0;

    ros::ServiceClient loadControllerSrv_;
    ros::ServiceClient listControllersSrv_;
    ros::ServiceClient switchControllerSrv_;
    
    actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> torsoAc_;

    float lerp(float a, float b, float f);
    void moveTorso(float targetQ, float duration, bool absolute);
    void jsCallback(const sensor_msgs::JointState::ConstPtr& msg);

    bool isControllerRunning(std::string name);
};

} // namespace placing_manager 
#endif  // GAZE_MANAGER_H