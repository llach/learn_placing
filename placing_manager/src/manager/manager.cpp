#include <manager/manager.h>

using namespace std;
using namespace std_srvs;
using namespace control_msgs;
using namespace trajectory_msgs;
using namespace placing_manager;
using namespace controller_manager_msgs;

PlacingManager::PlacingManager(float initialTorsoQ) :
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

    ROS_INFO("moving torso to %f", initialTorsoQ_);
    moveTorso(initialTorsoQ_, 1.0, true);

    ROS_INFO("PlacingManager::PlacingManager() done");
    initialized_ = true;
}

bool collectSample(){

    return true;
}

/*
        DATA CALLBACKS
*/

void PlacingManager::jsCallback(const sensor_msgs::JointState::ConstPtr& msg)
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