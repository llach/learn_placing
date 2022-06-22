
## Building

If you're using an older `ros_controllers` version where the JointTrajectoryController (JTC) wasn't inheritable (e.g. PAL's internal fork), you need to patch the JTC and rebuild it.

1. Install dependencies:
```
sudo apt install \
ros-melodic-four-wheel-steering-msgs \
pal-ferrum-pal-hardware-interfaces \
pal-ferrum-pal-hardware-interfaces-dev \
ros-melodic-urdf-geometry-parser
```
2. Apply patch in `ros_controllers` repository `git apply PATH_TO_MYRMEX_CONTROLLER/misc/jtc_inheritance.patch`
3. Run `catkin build` 

Now `myrmex_gripper_controller` should build without problems.

## Running on TIAGo

**Setup (needs to be done once)**

1. Copy startup files to TIAGo `cd misc && ./copy_startup_files.sh`
2. Deploy myrmex controller workspace (e.g. using `pal_deploy`) to copy controller and config files to robot
3. Reboot robot 

If the controller is changed (but no pal_startup files), it's sufficient to execute `pal_restart_deployer` to refresh the running controller without reboot.
In the WebCommander, you should see the myrmex controller entry we added under "Startup".
You can also verify that the startup scripts work by looking at the param server: `rosparam list | grep \/myrmex\_gripper`

**After each robot start**

Execute `cd misc && ./enable_myrmex_controller.sh`. This is only necessary if the script itself doesn't switch controllers itself.
