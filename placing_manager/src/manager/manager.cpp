#include <manager/manager.h>

using namespace std;
using namespace rosbag;
using namespace std_srvs;
using namespace control_msgs;
using namespace trajectory_msgs;
using namespace placing_manager;
using namespace controller_manager_msgs;

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

PlacingManager::PlacingManager(float initialTorsoQ) :
    waitRate_(50),
    dataCutOff_(0.7),
    initialTorsoQ_(initialTorsoQ),
    bufferJs(n_,            "joint_states", "/joint_states"),
    bufferMyLeft(n_,        "myrmex_left", "/tactile_left"),
    bufferMyRight(n_,       "myrmex_right", "/tactile_right"),
    bufferFt(n_,            "ft", "/wrist_ft"),
    bufferContact(n_,       "contact", "/table_contact/in_contact"),
    bufferObjectState(n_,   "object_state", "/normal_angle"),
    jsSub_(n_.subscribe("/joint_states", 1, &PlacingManager::jsCB, this)),
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

    // jsSub_ = n_.subscribe("/joint_states", 1, &PlacingManager::jsCB, this);

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

void PlacingManager::pause(){
    bufferJs.paused_ = true;
    bufferMyLeft.paused_ = true;
    bufferMyRight.paused_ = true;
    bufferFt.paused_ = true;
    bufferContact.paused_ = true;
    bufferObjectState.paused_ = true;

    paused_ = true;
}
void PlacingManager::unpause(){
    bufferJs.paused_ = false;
    bufferMyLeft.paused_ = false;
    bufferMyRight.paused_ = false;
    bufferFt.paused_ = false;
    bufferContact.paused_ = false;
    bufferObjectState.paused_ = false;

    paused_ = false;
}

bool PlacingManager::checkLastTimes(ros::Time n){
    if (not bufferJs.isFresh(n)) return false;
    if (not bufferMyLeft.isFresh(n)) return false;
    if (not bufferMyRight.isFresh(n)) return false;
    if (not bufferFt.isFresh(n)) return false;
    if (not bufferContact.isFresh(n)) return false;
    if (not bufferObjectState.isFresh(n)) return false;

    return true;
}

ros::Time PlacingManager::getContactTime(){
    {
        std::lock_guard<std::mutex> l(bufferContact.m_);
        ROS_INFO_STREAM("got " << bufferContact.times_.size() << " samples ");
        for (size_t i = 0; i<bufferContact.times_.size(); i++){
            if (bufferContact.data_[i].data) return bufferContact.times_[i];
        }
    }
    ROS_FATAL("no contact detected!!");
    return ros::Time(0);
}

void PlacingManager::storeSample(ros::Time contactTime){
    std::string date = currentDateTime();

    ros::Time fromTime = contactTime - dataCutOff_;
    ros::Time toTime = contactTime + dataCutOff_;

    std::string file = "/home/llach/placing_data/"+date+".bag";
    ROS_INFO_STREAM("storing file at " << file);

    Bag bag;
    bag.open(file, bagmode::Write);
    
    bufferJs.storeData(bag, fromTime, toTime);
    bufferMyLeft.storeData(bag, fromTime, toTime);
    bufferMyRight.storeData(bag, fromTime, toTime);
    bufferFt.storeData(bag, fromTime, toTime);
    bufferContact.storeData(bag, fromTime, toTime);
    bufferObjectState.storeData(bag, fromTime, toTime);

    bag.close();
}

bool PlacingManager::collectSample(){
    ROS_INFO("### collecting data sample ###");

    if (not checkLastTimes(ros::Time::now()-ros::Duration(1))) {
        ROS_ERROR("data not fresh");
        return false;
    }

    unpause();
    ROS_INFO("moving torso down towards the table ...");
    moveTorso(0.15, 3.0);
    pause();

    ros::Time contactTime = getContactTime();
    if (contactTime != ros::Time(0)){
        ROS_INFO_STREAM("contact detected at " << contactTime);
        storeSample(contactTime);
    } else{
        ROS_FATAL("no contact -> can't store sample");
    }

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
}
