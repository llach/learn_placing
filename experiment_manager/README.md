# Experiment Scripts

These aim to automate the data collection / experiment process.
`cm_interface.py` uses the old controller manager interface that is required by TIAGos running ferrum.
The code is taken from [here](https://github.com/ros-controls/ros_control/blob/0.5.0/controller_manager/src/controller_manager/controller_manager_interface.py) and modified to be python3-compatible using [2to3](https://docs.python.org/3/library/2to3.html).
Note that the host PC needs to build `controller_manager_msgs` matching the version installed on TIAGo (in our case `0.5.0`).
This can be done as follows:
* clone [`ros_control`](https://github.com/ros-controls/ros_control)
* checkout required version, e.g. `git checkout 0.5.0`
* copy `controller_manager_msgs` to catkin workspace's source folder (avoid building the entire `ros_control` repo)
* build

## Data Collection 

**Setting**:

TIAGo's gripper is positioned above a table surface (e.g. by using gravity compensation and manually moving it).
An object is given to TIAGo, i.e. it's being held between TIAGo's fingers and the gripper is closed.

**Experiment Procedure**

1. Switch controllers (abort on failure)
2. If `loop_count % M == 0`: calibrate FT contact detection
3. Move torso down until contact
4. wait for $t_{\text{wait}}$ seconds
5. Move torso up to (starting position || pre-defined position)
6. If `loop_count < N`, go to `2.`

**Parameters**

* $N$ - number of total trials (100)
* $M$ - re-calibrate FT sensor after $M$ trials (10)
* $t_{\text{wait}}$ - seconds to wait after table contact (0.3)

**Observations**

* 28 trials take ~3:24min
* problem: gripper motors overheat, the grip loosens and the object slips too much when in contact