#include <manager/manager.h>

using namespace std;
using namespace rosbag;
using namespace std_msgs;
using namespace std_srvs;
using namespace control_msgs;
using namespace wrist_motion;
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

PlacingManager::PlacingManager(float initialTorsoQ, float tableHeight) :
    waitRate_(50),
    dataCutOff_(0.7),
    initialTorsoQ_(initialTorsoQ),
    tableHeight_(tableHeight),
    bufferJs(n_,            "joint_states", "/joint_states"),
    bufferMyLeft(n_,        "tactile_left", "/tactile_left"),
    bufferMyRight(n_,       "tactile_right", "/tactile_right"),
    bufferFt(n_,            "ft", "/wrist_ft"),
    bufferTf(n_,            "tf", "/tf"),
    bufferContact(n_,       "contact", "/table_contact/in_contact"),
    bufferObjectState(n_,   "object_state", "/object_state_estimate"),
    jsSub_(n_.subscribe("/joint_states", 1, &PlacingManager::jsCB, this)),
    wristAc_("/wrist_plan", true),
    executeAc_("/execute_trajectory", true),
    torsoAc_("/torso_stop_controller/follow_joint_trajectory", true),
    tfListener_(tfBuffer_),
    ftCalibrationSrv_(n_.serviceClient<std_srvs::Empty>("/table_contact/calibrate")),
    loadControllerSrv_(n_.serviceClient<LoadController>("/controller_manager/load_controller")),
    listControllersSrv_(n_.serviceClient<ListControllers>("/controller_manager/list_controllers")),
    switchControllerSrv(n_.serviceClient<SwitchController>("/controller_manager/switch_controller"))
{
    ROS_INFO("checking torso controller ... ");
    if (!ensureRunningController(torsoStopControllerName_, torsoControllerName_)){
        ROS_FATAL("could not switch on torso stop controller");
        return; 
    }

    ROS_INFO("waiting for torso AC ...");
    torsoAc_.waitForServer();

    // for unknown reasons, this doesn't work
    // ROS_INFO("waiting for execution AC ...");
    // executeAc_.waitForServer();

    ROS_INFO("waiting for FT calibration service ...");
    ftCalibrationSrv_.waitForExistence();

    ROS_INFO("setting up joint states subscriber ...");
    jsSub_ = n_.subscribe("/joint_states", 1, &PlacingManager::jsCB, this);

    try {
        tfBuffer_.canTransform("base_footprint", "gripper_grasping_frame", ros::Time(0), ros::Duration(5));
    } catch (tf2::TransformException &ex) {
        ROS_FATAL("Could not setup get transform: %s",ex.what());
        return;
    }

    ROS_INFO("PlacingManager::PlacingManager() done");
    initialized_ = true;
}

bool PlacingManager::init(ros::Duration timeout, bool initTorso){
   
    ros::Time start = ros::Time::now();

    while(not checkLastTimes(ros::Time::now()-ros::Duration(1))){
        waitRate_.sleep();
        if (ros::Time::now() > start+timeout) return false;
    }

    if (initTorso) {
        {
            std::lock_guard<std::mutex> l(jsLock_); 
            ROS_INFO("moving torso from %f to %f", currentTorsoQ_, initialTorsoQ_);
        }
        moveTorso(initialTorsoQ_, 1.0, true);
    }
    return true;
}

void PlacingManager::pause(){
    bufferJs.paused_ = true;
    bufferMyLeft.paused_ = true;
    bufferMyRight.paused_ = true;
    bufferFt.paused_ = true;
    bufferTf.paused_ = true;
    bufferContact.paused_ = true;
    bufferObjectState.paused_ = true;

    paused_ = true;
}
void PlacingManager::unpause(){
    bufferJs.paused_ = false;
    bufferMyLeft.paused_ = false;
    bufferMyRight.paused_ = false;
    bufferFt.paused_ = false;
    bufferTf.paused_ = false;
    bufferContact.paused_ = false;
    bufferObjectState.paused_ = false;

    paused_ = false;
}

bool PlacingManager::checkLastTimes(ros::Time n){
    if (not bufferJs.isFresh(n)) return false;
    if (not bufferMyLeft.isFresh(n)) return false;
    if (not bufferMyRight.isFresh(n)) return false;
    if (not bufferFt.isFresh(n)) return false;
    if (not bufferTf.isFresh(n)) return false;
    if (not bufferContact.isFresh(n)) return false;

    // we don't check the freshness here since one reading during the downwards movement is sufficient
    // if (not bufferObjectState.isFresh(n)) return false;

    return true;
}

bool PlacingManager::checkSamples(const ros::Time &n){
    if (!bufferJs.hasDataBefore(n)) return false;
    if (!bufferMyLeft.hasDataBefore(n)) return false;
    if (!bufferMyRight.hasDataBefore(n)) return false;
    if (!bufferFt.hasDataBefore(n)) return false;
    if (!bufferTf.hasDataBefore(n)) return false;
    if (!bufferContact.hasDataBefore(n)) return false;
    if (!bufferObjectState.hasDataBefore(n)) return false;

    return true;
}

void PlacingManager::clearAll(){
    bufferJs.clear();
    bufferMyLeft.clear();
    bufferMyRight.clear();
    bufferFt.clear();
    bufferTf.clear();
    bufferContact.clear();
    bufferObjectState.clear();
}

ros::Time PlacingManager::getContactTime(){
    {
        std::lock_guard<std::mutex> l(bufferContact.m_);
        ROS_INFO_STREAM("got " << bufferContact.times_.size() << " samples ");
        for (size_t i = 0; i<bufferContact.times_.size(); i++){
            if (bufferContact.data_[i].in_contact) return bufferContact.times_[i];
        }
    }
    ROS_FATAL("no contact detected!!");
    return ros::Time(0);
}

void PlacingManager::storeSample(ros::Time contactTime){
    lastSample_ = currentDateTime();

    ros::Time fromTime = contactTime - dataCutOff_;
    ros::Time toTime = contactTime + dataCutOff_;

    std::string file = basePath_+lastSample_+".bag";

    Bag bag;
    bag.open(file, bagmode::Write);
    
    bufferJs.storeData(bag, fromTime, toTime);
    bufferMyLeft.storeData(bag, fromTime, toTime);
    bufferMyRight.storeData(bag, fromTime, toTime);
    bufferFt.storeData(bag, fromTime, toTime);
    bufferTf.storeData(bag, fromTime, toTime);
    bufferContact.storeData(bag, fromTime, toTime);
    bufferObjectState.storeData(bag, fromTime, toTime);

    // store some bag metadata
    String s;

    s.data = "fromTime";
    bag.write("bag_times", fromTime, s);

    s.data = "contactTime";
    bag.write("bag_times", contactTime, s);
    
    s.data = "toTime";
    bag.write("bag_times", toTime, s);

    bag.close();

    ROS_INFO_STREAM("\033[1;36msample saved at " << file <<"\033[0m");
}

bool PlacingManager::reorientate(){
    ROS_INFO("planning arm reorientation ...");
    PlanWristGoal g;
    g.table_height = tableHeight_+0.1;

    wristAc_.sendGoalAndWait(PlanWristGoal());
    auto pwr = wristAc_.getResult();

    if (pwr->trajectory.joint_trajectory.points.size()==0){
        ROS_FATAL("planning failed!");
        return false;
    }

    moveit_msgs::ExecuteTrajectoryGoal executeGoal;
    executeGoal.trajectory = pwr->trajectory;
   
    ROS_INFO("executing arm reorientation ...");
    executeAc_.sendGoal(executeGoal);
    
    int waitSecs = (int) 1000*executeGoal.trajectory.joint_trajectory.points.back().time_from_start.toSec();
    ROS_INFO_STREAM("waiting for " << waitSecs << " milliseconds");
    std::this_thread::sleep_for(std::chrono::milliseconds(waitSecs));

    ROS_INFO("done.");
}

float PlacingManager::getTorsoGoal(float &time){
    geometry_msgs::TransformStamped t;
    try {
        t = tfBuffer_.lookupTransform("base_footprint", "gripper_grasping_frame", ros::Time(0), ros::Duration(5));
    } catch (tf2::TransformException &ex) {
        ROS_FATAL("Could not setup get transform: %s",ex.what());
        return -1.0;
    }
    float gripperHeight = t.transform.translation.z;
    float gripperTableDiff = gripperHeight-(tableHeight_+0.07);

    float desQ;
    {
        std::lock_guard<std::mutex> l(jsLock_);
        desQ = currentTorsoQ_-gripperTableDiff;
    }
    time = (gripperTableDiff*100)*torsoVel_;
    ROS_INFO("gripper height %f; gripper table diff %f; time %f; desQ %f", gripperHeight, gripperTableDiff, time, desQ);
    return desQ;
}
    
bool PlacingManager::collectSample(){
    ROS_INFO("### collecting data sample no. %d ###", nSamples_);

    ROS_INFO("recalibrating FT");
    std_srvs::Empty e;
    ftCalibrationSrv_.call(e);

    if (not checkLastTimes(ros::Time::now()-ros::Duration(1))) {
        ROS_ERROR("data not fresh");
        return false;
    }

    clearAll();
    unpause();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    ROS_INFO("moving torso down towards the table ...");

    ros::Time startMoving = ros::Time::now();

    float tTime;
    float tGoal = getTorsoGoal(tTime);

    moveTorso(tGoal, tTime);
    ros::Time stopMoving = ros::Time::now();
    ros::Duration moveDur = stopMoving - startMoving;

    // pause data recording
    pause();
    nSamples_++;

    ROS_INFO("got %d object samples", bufferObjectState.numData());

    ros::Time contactTime = getContactTime();
    bool hasSamples = checkSamples(contactTime);

    if (contactTime != ros::Time(0) && hasSamples){

        ROS_INFO_STREAM("contact detected at " << contactTime);
        storeSample(contactTime);

    } else if (contactTime == ros::Time(0) ) {
        ROS_FATAL_STREAM("\033[1;31mno contact -> can't store sample"<<"\033[0m");
    } else if (!hasSamples) {
        ROS_FATAL_STREAM("\033[1;31mmissing samples"<<"\033[0m");
    } else {
        ROS_FATAL_STREAM("\033[1;31munknown error"<<"\033[0m");
    }

    ROS_INFO("move torso up again ...");
    moveTorso(initialTorsoQ_, moveDur.toSec());

    reorientate();

    return true;
}

void PlacingManager::flagSample(){
    ROS_INFO_STREAM("flagging sample " << lastSample_);
    
    std::ofstream outfile;
    outfile.open(basePath_+"flagged.txt", std::ios_base::app);
    outfile << lastSample_ << "\n"; 
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

    ROS_INFO("moving torso to %f", targetQ);

    FollowJointTrajectoryGoal torsoGoal;
    torsoGoal.trajectory = jt;

    torsoAc_.sendGoalAndWait(torsoGoal);
    ROS_INFO("Action finished: %s",torsoAc_.getState().toString().c_str());
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
        sw.request.strictness = 0;

        switchControllerSrv.call(sw);
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
