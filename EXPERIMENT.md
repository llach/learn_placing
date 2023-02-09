# Experiment Procedure OPTI
1. Turn on robot
2. (optional) sync new workspace from within docker (Ctrl+r -> ferrum -> ENTER):
   1. `cd ~/placing_ws && source devel/setup.bash`
   2. `rosrun pal_deploy deploy.py 10.68.0.1 -y --cmake_args="-DDISABLE_PAL_FLAGS=TRUE"`
   3. [on robot PC] `pal_restart_deployer`
3. Put object on the table, make sure cameras have sight
4. All further shells assume ROS_MASTER_URI to be set to the robot (and ROS_IP to be set accordingly) (alias `rostl`). To check run `rostopic hz /joint_states` on experiment PC, if messages arrive it's fine.
5. [Laptop] Start Myrmex: `roslaunch tiago_myrmex myrmex_readout.launch`
   1. the tape on the cables, the sensor-finger-mount and next to the USB port mean it's the sensor of the right finger
   2. if it says "nothing to read" or any other error, plug out sensors, plug back in and try again
6. [TIAGo]  Start OptiTrack: `roslaunch optitrack_bridge optitrack.launch`
7. [Laptop] Start components: `roslaunch placing_manager components.launch`
8. [Laptop] Setup experiments: `rosrun placing_manager setup_experiment.py`
   1. `s` skips a step
   2. `q` quits the script
   3. `[ENTER]` confirms action
9.  [TIAGo] Run rviz, load config `placing.rviz` (under File -> Recent Configs) and check whether opti frames are there and everything is aligned.
10. Make sure `/home/llach/placing_data` is empty (samples will be stored here)
11. [Laptop] Run sample collection `rosrun placing_manager node`

# Experiment Procedure CAMERAS
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
* open and close gripper: `/home/llach/.pyenv/versions/3.9.6/bin/python /home/llach/robot_mitm_ws/src/learn_placing/myrmex_gripper_controller/scripts/open_close.py`
  * `o` opens gripper
  * `c` closes gripper
  * `k` kills current goal **recommended** after closing for experiments to avoid gripper movements caused be force differences during tap

## Notes

ObjectVar2, GripperVar2 arm positions:
`[1.16, 0.13, 1.95, 1.10, 0.03, 0.37, 0.06]`

### GripperOptiTest

arm positions: `[1.29, 0.03, 2.29, 0.76, -0.29, 0.67, 0.75]`

arm_7:
* start (middle point): `0.75`
* min: `-0.19`
* max: `1.75`

* sometimes optitrack didn't detect the markers. the data collection node recognized this. keeping an eye on the optitrack state publisher terminal allows to detect vanished markers quickly
* in one specifc arm position, TIAGo's arm was shaking while moving which led to false positive table contacts. samples were flagged (manually)


# Experiment Procedure UPC

1. Connect switch to power
2. Power on OptiTrack PC, check calibration and re-calibrate if needed
3. TIAGo
   1. Connect Ethernet
   2. Start robot (if battery is depleted: F1; F10; ENTER, keyboard LEDs should indicate booting)
   3. `sudo ntpdate -u tiago-72c`; otherwise transforms won't work
   4. `rosrun optitrack_publisher ot_node.py`
   5. `roslaunch vrpn_client_ros sample.launch server:=192.168.141.20 ~broadcast_tf:=False`
   6. check TFs in rviz
4. Mac 
   1. Connect Ethernet, make sure it's connected in Settings
   2. 