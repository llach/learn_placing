#include <manager/manager.h>

using namespace std;
using namespace std_srvs;
using namespace control_msgs;
using namespace trajectory_msgs;
using namespace placing_manager;
using namespace controller_manager_msgs;

PlacingManager::PlacingManager(float initialTorsoQ) :
    waitRate_(50),
    initialTorsoQ_(initialTorsoQ),
    torsoAc_("/torso_stop_controller/follow_joint_trajectory", true),
    loadControllerSrv_(n_.serviceClient<LoadController>("/controller_manager/load_controller")),
    listControllersSrv_(n_.serviceClient<ListControllers>("/controller_manager/list_controllers")),
    switchControllerSrv_(n_.serviceClient<SwitchController>("/controller_manager/switch_controller"))
{
    ROS_INFO("checking torso controller ... ");
    if (!ensureRunningController(torsoStopControllerName_, torsoControllerName_)){
        ROS_FATAL("could not switch on torso stop controller");
        return; 
    }

    ROS_INFO("waiting for torso AC...");
    torsoAc_.waitForServer();

    ROS_INFO("creating data subscribers ...");
    jsSub_ = n_.subscribe("/joint_states", 1, &PlacingManager::jsCB, this);
    myrmexLSub_ = n_.subscribe("/tactile_left", 1, &PlacingManager::mmLeftCB, this);
    myrmexRSub_ = n_.subscribe("/tactile_right", 1, &PlacingManager::mmRightCB, this);
    ftSub_ = n_.subscribe("/wrist_ft", 1, &PlacingManager::ftCB, this);
    contactSub_ = n_.subscribe("/table_contact/in_contact", 1, &PlacingManager::contactCB, this);
    objectStateSub_ = n_.subscribe("/normal_angle", 1, &PlacingManager::objectStateCB, this);

    ROS_INFO("PlacingManager::PlacingManager() done");
    initialized_ = true;
}

bool PlacingManager::init(ros::Duration timeout){
    ros::Time start = ros::Time::now();

    while(not checkLastTimes(ros::Time::now()-ros::Duration(1))){
        waitRate_.sleep();
        if (ros::Time::now() > start+timeout) return false;
    }

    {
        std::lock_guard<std::mutex> l(jsLock_); 
        ROS_INFO("moving torso from %f to %f", currentTorsoQ_, initialTorsoQ_);
    }

    moveTorso(initialTorsoQ_, 1.0, true);
    return true;
}

bool PlacingManager::checkLastTimes(ros::Time n){
    {
        std::lock_guard<std::mutex> l(jsLock_);
        if (lastJsTime_ < n) {
            ROS_WARN("js not fresh");
            return false;
        }
    }
    {
        std::lock_guard<std::mutex> l(mlLock_);
        if (lastMlTime_ < n) {
            ROS_WARN("mm left not fresh");
            return false;
        }
    }
    {
        std::lock_guard<std::mutex> l(mrLock_);
        if (lastMrTime_ < n) {
            ROS_WARN("mm right not fresh");
            return false;
        }
    }
    {
        std::lock_guard<std::mutex> l(ftLock_);
        if (lastFtTime_ < n) {
            ROS_WARN("ft not fresh");
            return false;
        }
    }
    {
        std::lock_guard<std::mutex> l(contactLock_);
        if (lastContactTime_ < n) {
            ROS_WARN("contact not fresh");
            return false;
        }
    }
    {
        std::lock_guard<std::mutex> l(objectStateLock_);
        if (lastObjectStateTime_ < n) {
            ROS_WARN("object state not fresh");
            return false;
        }
    }
    return true;
}

bool PlacingManager::collectSample(){
    ROS_INFO("### collecting data sample ###");

    if (not checkLastTimes(ros::Time::now()-ros::Duration(1))) {
        ROS_ERROR("data not fresh");
        return false;
    }

    ROS_INFO("moving torso down towards the table ...");
    moveTorso(0.15, 3.0);

    ROS_INFO("move torso up again");
    moveTorso(initialTorsoQ_, 2.0); // TODO less time here -> faster

    return true;
}

/*
    TORSO CONTROLLER METHODS
*/

void PlacingManager::moveTorso(float targetQ, float duration, bool absolute){
    float startQ;
    {   
        std::lock_guard<std::mutex> l(jsLock_);
        startQ = currentTorsoQ_;
    }

    if (!absolute){
        targetQ = startQ + targetQ;
    }

    JointTrajectory jt;
    jt.header.frame_id = baseFrame_;
    jt.joint_names = {torsoJointName_};

    for (int i = 0; i < 5; i++){
        float t = (float) (i+1)/5; 

        JointTrajectoryPoint jp;
        jp.positions = {lerp(startQ, targetQ, t)};
        jp.time_from_start = ros::Duration(t*duration);

        jt.points.push_back(jp);
    }

    FollowJointTrajectoryGoal torsoGoal;
    torsoGoal.trajectory = jt;

    torsoAc_.sendGoalAndWait(torsoGoal);
}

float PlacingManager::lerp(float a, float b, float f) {
    return a + f * (b - a);
}

/*
    CONTROLLER MANAGER METHODS
*/
bool PlacingManager::ensureRunningController(std::string name, std::string stop){
    if (!isControllerRunning(name)){
        LoadController load;
        load.request.name = name;

        loadControllerSrv_.call(load);

        SwitchController sw;
        sw.request.stop_controllers = {stop};
        sw.request.start_controllers = {name};

        switchControllerSrv_.call(sw);
        return isControllerRunning(name);
    }
    return true;    
}

bool PlacingManager::isControllerRunning(std::string name) {
    ListControllers list;
    listControllersSrv_.call(list);

    for (auto& cs : list.response.controller) {
        if (cs.name == name && cs.state == "running") return true;
    }
    return false;
}


/*
    DATA CALLBACKS
*/

void PlacingManager::jsCB(const sensor_msgs::JointState::ConstPtr& msg)
{
    if (torsoIdx_ == -1){
        for (int i = 0; i < msg->name.size(); i++){
            if (msg->name[i] == torsoJointName_) torsoIdx_ = i;
        }
        if (torsoIdx_ == -1){
            ROS_FATAL("some joint index was not found: torso=%d", torsoIdx_);
            return;
        }
    }

    std::lock_guard<std::mutex> l(jsLock_);
    currentTorsoQ_ = msg->position[torsoIdx_];

    auto time = ros::Time::now();
    lastJsTime_ = time;

    if (paused_) return;

    jsData_.push_back(msg);
    jsTime_.push_back(time);

}

void PlacingManager::mmLeftCB(const tactile_msgs::TactileState::ConstPtr& msg){
    std::lock_guard<std::mutex> l(mlLock_);
    auto time = ros::Time::now();
    lastMlTime_ = time;

    if (paused_) return;

    mlData_.push_back(msg);
    mlTime_.push_back(time);
}

void PlacingManager::mmRightCB(const tactile_msgs::TactileState::ConstPtr& msg){
    std::lock_guard<std::mutex> l(mrLock_);
    auto time = ros::Time::now();
    lastMrTime_ = time;

    if (paused_) return;

    mrData_.push_back(msg);
    mrTime_.push_back(time);
}

void PlacingManager::ftCB(const geometry_msgs::WrenchStamped::ConstPtr& msg){
    std::lock_guard<std::mutex> l(ftLock_);
    auto time = ros::Time::now();
    lastFtTime_ = time;

    if (paused_) return;

    ftData_.push_back(msg);
    ftTime_.push_back(time);
}

void PlacingManager::contactCB(const std_msgs::Bool::ConstPtr& msg){
    std::lock_guard<std::mutex> l(contactLock_);
    auto time = ros::Time::now();
    lastContactTime_ = time;

    if (paused_) return;

    contactData_.push_back(msg);
    contactTime_.push_back(time);
}

void PlacingManager::objectStateCB(const std_msgs::Float64::ConstPtr& msg){
    std::lock_guard<std::mutex> l(objectStateLock_);
    auto time = ros::Time::now();
    lastObjectStateTime_ = time;

    if (paused_) return;

    objectStateData_.push_back(msg);
    objectStateTime_.push_back(time);
}

