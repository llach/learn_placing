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

std::vector<std::string> dataTopics = {
    "/tactile_right",
    "/tactile_left",
    "/wrist_ft",
    "/table_contact/in_contact",
    "/normal_angle",
    "/joint_states"
};

std::vector<std::type_index> msgs = {
    typeid(tactile_msgs::TactileState)
};

// std::tuple<
//     TopicBuffer<tactile_msgs::TactileState>,
//     TopicBuffer<tactile_msgs::TactileState>,
//     TopicBuffer<geometry_msgs::WrenchStamped>,
//     TopicBuffer<std_msgs::Bool>,
//     TopicBuffer<std_msgs::Float64>,
//     TopicBuffer<sensor_msgs::JointState>
// > dataSources;


// std::vector<std::any> {
    // tactile_msgs::TactileState,
    // tactile_msgs::TactileState,
    // geometry_msgs::WrenchStamped,
    // std_msgs::Bool,
    // std_msgs::Float64,
    // sensor_msgs::JointState,
// };


// std::map<std::string, std::string> name2topic {
//     {"myrmex_right",    "/tactile_right"},
//     {"myrmex_left",     "/tactile_left"},
//     {"ft",              "/wrist_ft"},
//     {"contact",         "/table_contact/in_contact"},
//     {"object_state",    "/normal_angle"},
//     {"joint_states",    "/joint_states"},
// };

// std::map<std::string, std::type_index> name2msg {
//     {"myrmex_right",    tactile_msgs::TactileState},
//     {"myrmex_left",     tactile_msgs::TactileState},
//     {"ft",              geometry_msgs::WrenchStamped},
//     {"contact",         std_msgs::Bool},
//     {"object_state",    std_msgs::Float64},
//     {"joint_states",    sensor_msgs::JointState},
// };


class PlacingManager{
public:
    PlacingManager(float initialTorsoQ = 0.25);
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

    ros::ServiceClient loadControllerSrv_;
    ros::ServiceClient listControllersSrv_;
    ros::ServiceClient switchControllerSrv_;

    actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> torsoAc_;

    float lerp(float a, float b, float f);
    
    void moveTorso(float targetQ, float duration, bool absolute = true);
    void jsCallback(const sensor_msgs::JointState::ConstPtr& msg);

    bool isControllerRunning(std::string name);
    bool ensureRunningController(std::string name, std::string stop);

    std::mutex jsLock_;
    std::mutex mlLock_;
    std::mutex mrLock_;
    std::mutex ftLock_;
    std::mutex contactLock_;
    std::mutex objectStateLock_;

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

    void mmLeftCB(const tactile_msgs::TactileState::ConstPtr& msg);
    void mmRightCB(const tactile_msgs::TactileState::ConstPtr& msg);
    void ftCB(const geometry_msgs::WrenchStamped::ConstPtr& msg);
    void contactCB(const std_msgs::Bool::ConstPtr& msg);
    void objectStateCB(const std_msgs::Float64::ConstPtr& msg);
};

} // namespace placing_manager 