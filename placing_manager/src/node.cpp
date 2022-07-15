#include <ros/ros.h>
#include <manager/manager.h>

using namespace placing_manager;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "placing_manager");
  
  ros::NodeHandle nh;
  ros::Rate loop_rate(10);

  PlacingManager pm;

  ros::AsyncSpinner spinner(4);
  spinner.start();

  while (ros::ok()) { loop_rate.sleep(); }
  return 0;
}