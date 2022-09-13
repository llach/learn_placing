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

    int n = 99;
    while (ros::ok()) {
        cout << "\nnext? ["<< n <<"] (0=exit;1-5=collect;6=static;7=send;8=flag;9=grav)" << endl;
        cin >> n;

        if (n==0) return 0;
        if (n == 6 || n == 7) {
            ROS_INFO("\n--- static sample ---"); 
            pm->staticSample();
            ROS_INFO("\n--- sending sample ---"); 
            pm->sendSample();
        } else if (n==8) {
            ROS_INFO("\n--- flagging sample ---"); 
            pm->flagSample();
        } else if (n==9) {
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
        } else if (n >= 1 && n <=5) {
            // for (int j = 0; j<n;j++){
            std::cout << "calling PM" << std::endl;
            bool success = pm->collectSample();
            if (!success) return -1;
            // }
        } else {
            std::cout << "invalid n: " << n << std::endl;
        }
    }
    return 0;
}