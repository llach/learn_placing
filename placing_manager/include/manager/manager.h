/* Author: Luca Lach
*/

#pragma once

// std includes
#include <mutex>
#include <typeindex>

// misc ros includes
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>

// ros msgs
#include <std_msgs/Bool.h>
#include <std_srvs/Empty.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/JointState.h>
#include <tactile_msgs/TactileState.h>
#include <geometry_msgs/WrenchStamped.h>
#include <controller_manager_msgs/LoadController.h>
#include <controller_manager_msgs/ListControllers.h>
#include <controller_manager_msgs/SwitchController.h>
#include <control_msgs/FollowJointTrajectoryAction.h>

// local includes
#include <manager/topic_buffer.h>

namespace placing_manager {

class PlacingManager{
public:
    PlacingManager(float initialTorsoQ = 0.25);

    bool init(ros::Duration timeout);
    bool collectSample();
    
private:
    std::atomic<bool> paused_;
    bool initialized_ = false;

    int torsoIdx_ = -1;

    std::string baseFrame_ = "base_footprint";
    std::string torsoJointName_ = "torso_lift_joint";
    std::string torsoControllerName_ = "torso_controller";
    std::string torsoStopControllerName_ = "torso_stop_controller";

    float initialTorsoQ_;
    float currentTorsoQ_ = -1.0;

    ros::NodeHandle n_;
    ros::Rate waitRate_;

    ros::ServiceClient loadControllerSrv_;
    ros::ServiceClient listControllersSrv_;
    ros::ServiceClient switchControllerSrv_;

    actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> torsoAc_;

    float lerp(float a, float b, float f);
    
    void moveTorso(float targetQ, float duration, bool absolute = true);

    bool checkLastTimes(ros::Time n);
    bool isControllerRunning(std::string name);
    bool ensureRunningController(std::string name, std::string stop);

    std::mutex jsLock_;
    std::mutex mlLock_;
    std::mutex mrLock_;
    std::mutex ftLock_;
    std::mutex contactLock_;
    std::mutex objectStateLock_;

    ros::Subscriber jsSub_;
    ros::Subscriber myrmexLSub_;
    ros::Subscriber myrmexRSub_;
    ros::Subscriber ftSub_;
    ros::Subscriber contactSub_;
    ros::Subscriber objectStateSub_;

    std::vector<sensor_msgs::JointStateConstPtr> jsData_;
    std::vector<tactile_msgs::TactileStateConstPtr> mlData_;
    std::vector<tactile_msgs::TactileStateConstPtr> mrData_;
    std::vector<geometry_msgs::WrenchStampedConstPtr> ftData_;
    std::vector<std_msgs::BoolConstPtr> contactData_;
    std::vector<std_msgs::Float64ConstPtr> objectStateData_;

    std::vector<ros::Time> jsTime_;
    std::vector<ros::Time> mlTime_;
    std::vector<ros::Time> mrTime_;
    std::vector<ros::Time> ftTime_;
    std::vector<ros::Time> contactTime_;
    std::vector<ros::Time> objectStateTime_;

    ros::Time lastJsTime_;
    ros::Time lastMlTime_;
    ros::Time lastMrTime_;
    ros::Time lastFtTime_;
    ros::Time lastContactTime_;
    ros::Time lastObjectStateTime_;

    void jsCB(const sensor_msgs::JointState::ConstPtr& msg);
    void mmLeftCB(const tactile_msgs::TactileState::ConstPtr& msg);
    void mmRightCB(const tactile_msgs::TactileState::ConstPtr& msg);
    void ftCB(const geometry_msgs::WrenchStamped::ConstPtr& msg);
    void contactCB(const std_msgs::Bool::ConstPtr& msg);
    void objectStateCB(const std_msgs::Float64::ConstPtr& msg);
};

} // namespace placing_manager 