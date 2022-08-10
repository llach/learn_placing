#include <ros/ros.h>
#include <manager/manager.h>

using namespace std;
using namespace ros;
using namespace placing_manager;

bool initPM(shared_ptr<PlacingManager> pm){
    pm = make_shared<PlacingManager>();
    if (not pm->init(ros::Duration(3))){
        ROS_ERROR("Failed to init pm");
        return false;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "placing_manager");

    AsyncSpinner spinner(4);
    spinner.start();

    shared_ptr<PlacingManager> pm;
    if (not initPM(pm)) return -1;

    int n;
    while (ros::ok()) {
        cout << "\nnext? (0=exit;1=flag;2-9=next)" << endl;
        cin >> n;
        if (n==0) return 0;
        if (n==1) {ROS_INFO("\n--- flagging sample ---\n"); pm->flagSample();}
        if (n==2) {ROS_INFO("\n+++ reinitializing PM +++\n"); initPM(pm);}
        
        bool success = pm->collectSample();
        if (!success) return -1;
    }
    return 0;
}