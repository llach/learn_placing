/* Author: Luca Lach
*/

#pragma once

// std includes
#include <mutex>
#include <chrono>
#include <time.h>
#include <thread>
#include <string>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <typeindex>

// misc ros includes
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <tf2_ros/transform_listener.h>
#include <actionlib/client/simple_action_client.h>

// ros msgs
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <std_srvs/Empty.h>
#include <std_msgs/Float64.h>
#include <tf2_msgs/TFMessage.h>
#include <sensor_msgs/JointState.h>
#include <state_estimation/BoolHead.h>
#include <tactile_msgs/TactileState.h>
#include <geometry_msgs/WrenchStamped.h>
#include <wrist_motion/PlanWristAction.h>
#include <placing_manager/ExecutePlacing.h>
#include <moveit_msgs/ExecuteTrajectoryAction.h>
#include <state_estimation/ObjectStateEstimate.h>
#include <controller_manager_msgs/LoadController.h>
#include <controller_manager_msgs/ListControllers.h>
#include <controller_manager_msgs/SwitchController.h>
#include <control_msgs/FollowJointTrajectoryAction.h>

// local includes
#include <manager/topic_buffer.h>

namespace placing_manager {

class PlacingManager{
public:
    PlacingManager(float initialTorsoQ = 0.35, float tableHeight_ = 0.72);

    bool init(ros::Duration timeout, bool initTorso);
    void sendSample();
    void flagSample();
    bool collectSample();
    bool isControllerRunning(std::string name);
private:
    // parameters
    int nFTRecalibrate_ = 5;
    ros::Duration dataCutOff_;

    std::atomic<bool> paused_;
    bool initialized_ = false;

    int torsoIdx_ = -1;
    int nSamples_ = 0;

    std::string lastSample_;
    std::string basePath_ = "/home/llach/placing_data/";

    std::string baseFrame_ = "base_footprint";
    std::string torsoJointName_ = "torso_lift_joint";
    std::string torsoControllerName_ = "torso_controller";
    std::string torsoStopControllerName_ = "torso_stop_controller";

    float initialTorsoQ_;
    float tableHeight_;
    float currentTorsoQ_ = -1.0;
    float torsoVel_ = 0.4; // sec/cm

    ros::Time contactTime_;
    ros::NodeHandle n_;
    ros::Rate waitRate_;

    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener tfListener_;

    ros::ServiceClient ftCalibrationSrv_;
    ros::ServiceClient loadControllerSrv_;
    ros::ServiceClient listControllersSrv_;

    actionlib::SimpleActionClient<wrist_motion::PlanWristAction> wristAc_;
    actionlib::SimpleActionClient<moveit_msgs::ExecuteTrajectoryAction> executeAc_;
    actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> torsoAc_;

    float lerp(float a, float b, float f);
    
    void moveTorso(float targetQ, float duration, bool absolute = true);

    ros::Time getContactTime();
    void pause();
    void unpause();
    void clearAll();
    bool reorientate();
    float getTorsoGoal(float &time);
    void storeSample();
    bool checkSamples(const ros::Time &n);
    bool checkLastTimes(ros::Time n);
    bool ensureRunningController(std::string name, std::string stop);

    TopicBuffer<tf2_msgs::TFMessage> bufferTf;
    TopicBuffer<sensor_msgs::JointState> bufferJs;
    TopicBuffer<state_estimation::BoolHead> bufferContact;
    TopicBuffer<tactile_msgs::TactileState> bufferMyLeft;
    TopicBuffer<tactile_msgs::TactileState> bufferMyRight;
    TopicBuffer<geometry_msgs::WrenchStamped> bufferFt;
    TopicBuffer<state_estimation::ObjectStateEstimate> bufferObjectState;

    std::mutex jsLock_;
    ros::Subscriber jsSub_;
    void jsCB(const sensor_msgs::JointState::ConstPtr& msg);
public:
    ros::ServiceClient switchControllerSrv;
};

} // namespace placing_manager 