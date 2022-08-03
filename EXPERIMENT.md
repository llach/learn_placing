# Experiment Procedure
1. Turn on robot
2. (optional) sync new workspace
   1. [on experiment PC] `rosrun pal_deploy deploy.py 10.68.0.1 -y --cmake_args="-DDISABLE_PAL_FLAGS=TRUE"`
   2. [on robot PC] `pal_restart_deployer`
3. Run tiago camera: 
   1. ssh tiago-XXXc
   2. sudo su
   3. roslaunch state_estimation onlyuvc.launch
4. Put object on the table, make sure cameras have sight
5. All further shells assume ROS_MASTER_URI to be set to the robot (and ROS_IP to be set accordingly). To check run `rostopic hz /joint_states` on experiment PC, if messages come it's fine.
6. Start Myrmex: `roslaunch tiago_myrmex myrmex_readout.launch`
7. Start components: `roslaunch placing_manager components.launch`
8. Setup experiments: `rosrun placing_manager setup_experiment.py`


### TODOs

* correct average of orientations (object_state)
* robust contact detection 
* correct AC usage