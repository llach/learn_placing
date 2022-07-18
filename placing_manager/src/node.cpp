#include <ros/ros.h>
#include <manager/manager.h>

using namespace std;
using namespace ros;
using namespace placing_manager;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "placing_manager");

    AsyncSpinner spinner(4);
    spinner.start();

    PlacingManager pm;
    pm.init();
    
    int n;
    while (ros::ok()) {
        cout << "waiting" << endl;
        cin >> n;
        if (n==0) return 0;
        
        bool success = pm.collectSample();
        if (!success) return -1;
    }
    return 0;
}