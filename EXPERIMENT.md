# Experiment Procedure
1. Turn on robot
2. (optional) sync new workspace
   1. [on experiment PC] `rosrun pal_deploy deploy.py 10.68.0.1 -y --cmake_args="-DDISABLE_PAL_FLAGS=TRUE"`
   2. [on robot PC] `pal_restart_deployer`
3. Run tiago camera: 
   1. ssh tiago-XXXc
   2. roslaunch state_estimation tiagocam.launch
4. [Optional] Run TIAGo's xtion with extrinsic calibration
   1. Stop head_manager, node_doctor_xtion and head_xtion
   2. on the robot, run `roslaunch openni2_launch openni2.launch rgb_camera_info_url:=file:///home/pal/.ros/camera_info/rgb_xtion.yaml depth_camera_info_url:=file:///home/pal/.ros/camera_info/depth_xtion.yaml publish_tf:=false camera:=xtion depth_registration:=false`
   3. `depth_registration:=false` makes sure extrinsic calibration is used
5. Put object on the table, make sure cameras have sight
6. All further shells assume ROS_MASTER_URI to be set to the robot (and ROS_IP to be set accordingly). To check run `rostopic hz /joint_states` on experiment PC, if messages come it's fine.
7. Start Myrmex: `roslaunch tiago_myrmex myrmex_readout.launch`
   1. if it says "nothing to read" or any other error, plug out sensors, plug back in and try again
8. Start components: `roslaunch placing_manager components.launch`
9.  Setup experiments: `rosrun placing_manager setup_experiment.py`
   1. `s` skips a step
   2. `q` quits the script
   3. `[ENTER]` confirms action

## Troubleshooting:

* If myrmex sensors don't deliver data, reconnect the USB cables of the PC
* when starting the OpenNi camera on the robot, it sometimes fails if the tiagocam is connected. Reconnecting the webcam solves the issue.

## Notes

ObjectVar2, GripperVar2 arm positions:
`[1.16, 0.13, 1.95, 1.10, 0.03, 0.37, 0.06]`