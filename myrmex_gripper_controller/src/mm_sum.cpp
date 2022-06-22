#include "ros/ros.h"
#include "std_msgs/Int32.h"
#include "myrmex_gripper_controller/myrmex_processor.h"

using namespace std;
using namespace myrmex_gripper_controller;

int main(int argc, char **argv)
{
    int B = 0;
    bool normalize = false;

    ros::init(argc, argv, "mm_sum");
    
    ros::NodeHandle nh;

    MyrmexProcessor rmp("right", B, normalize);
    MyrmexProcessor lmp("left", B, normalize);

    ros::Publisher rpub = nh.advertise<std_msgs::Int32>("/tactile_right/sum", 0);
    ros::Publisher lpub = nh.advertise<std_msgs::Int32>("/tactile_left/sum", 0);

    ros::AsyncSpinner spinner(4);
    spinner.start();

    ros::Rate loop_rate(500);
    while (ros::ok()) {
        std_msgs::Int32 r;
        r.data = rmp.getTotalForce();
        cout << r.data << endl;
        rpub.publish(r);

        loop_rate.sleep();
    }
    return 0;
}
