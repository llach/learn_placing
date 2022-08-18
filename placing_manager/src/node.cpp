#include <ros/ros.h>
#include <manager/manager.h>
#include <pal_common_msgs/EmptyAction.h>
#include <controller_manager_msgs/SwitchController.h>


using namespace std;
using namespace ros;
using namespace placing_manager;
using namespace pal_common_msgs;
using namespace controller_manager_msgs;


int main(int argc, char **argv)
{
    ros::init(argc, argv, "placing_manager");

    AsyncSpinner spinner(4);
    spinner.start();

    actionlib::SimpleActionClient<EmptyAction> gravAc("/gravity_compensation", true);

    shared_ptr<PlacingManager> pm = make_shared<PlacingManager>();
    pm->init(ros::Duration(3), false);

    int n;
    while (ros::ok()) {
        cout << "\nnext? (0=exit;1=flag;2=grav;3-9=next)" << endl;
        cin >> n;

        if (n==0) return 0;
        if (n==1) {
            ROS_INFO("\n--- flagging sample ---"); 
            pm->flagSample();
        } else if (n==2) {
            ROS_INFO("\n+++ disabling arm gravity ... +++");
            EmptyGoal eag;
            gravAc.sendGoal(eag);
            ROS_INFO("ok\ndone?");
            char c;
            cin >> c;
            gravAc.cancelGoal();

            ROS_INFO("switching torso controller ...");
            SwitchController swc;
            swc.request.start_controllers = {"torso_stop_controller"};
            swc.request.stop_controllers = {"torso_controller"};
            swc.request.strictness = 0;
            pm->switchControllerSrv.call(swc);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            ROS_INFO("reinitializing PM ...");
            pm = make_shared<PlacingManager>();
            pm->init(ros::Duration(3), false);
        } else {
            std::cout << "calling PM" << std::endl;
            bool success = pm->collectSample();
            if (!success) return -1;
        }
    }
    return 0;
}