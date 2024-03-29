# Learn to Place on TIAGo <!-- omit in toc -->

- [Building and Execution Code for/on TIAGo](#building-and-execution-code-foron-tiago)
  - [Case I: Deploying on the robot](#case-i-deploying-on-the-robot)
  - [Case II: Running on an external PC](#case-ii-running-on-an-external-pc)
- [Components and Experiment Procedures](#components-and-experiment-procedures)
- [Dependencies](#dependencies)

## Building and Execution Code for/on TIAGo

If TIAGo is running a different Ubuntu version to what is used on development PCs (e.g. 18.04 vs 20.04), building software is a bit more complicated.
Software can be used with TIAGo in two different ways: deploying and running on TIAGo or building and running it locally (i.e. on an external PC).

In both cases, it's recommended to create an overlay workspace that's used to build dependencies **that will not modified during development**.
This is to a) avoid re-building all software when cleaning the development workspace and b) provide the same overlay for all development workspaces (everyone profits from fixes in the overlay workspace).

The overlay ws is created just like any other:
* open a new shell
* source the ROS / PAL distro that is being build upon (e.g. `source /opt/ros/melodic/setup.zsh`)
* create workspace and src folder `mkdir -p ~/overlay_ws/src`
* move / checkout dependencies into src folder
* go to ws root `cd ~/overlay_ws`
* run `catkin config install` (enables install space to use the ws's header files)
* build `catkin build`

Creating development workspaces is done like so:
* open a new shell
* create workspace and src folder `mkdir -p ~/dev_ws/src`
* within the workspace, run `catkin config --extend ~/overlay_ws/install`
* run `catkin build`
* from now on, source the ws's `dev_ws/devel/setup.zsh` and then build (like always)

### Case I: Deploying on the robot

It's important that the software is being linked to the same libraries that are present on the robot.
Using a Ubuntu 18.04 docker image containing the same packages and libraries to build it is the recommended solution.
Overlay and development workspaces can still be created on the host PC, they can be mounted by using docker's `-v` argument.
It's a good idea to mount the complete home of the docker's internal user account, this makes it possible to use `.bashrc` and shell history.
Software can be developed from within docker (launch `code` from within docker) and then it can be deployed to the robot.

### Case II: Running on an external PC

In this case, only the messages that are used to communicate to the robot need to be identical.
The only tricky part is to find which message version needs to be built.

On the robot, using `roscd MESSAGE_PACKAGE` we can have a look at the `package.xml` to find out what the package version is.
However, since PAL has internal forks, these versions might be different from the official ROS package counterparts.
So checking the message itself and comparing whether the correct ones are built is necessary.
To do so, execute, for example, `rosmsg show controller_manager_msgs/ControllerState` and compare message fields to the ones being built in the overlay ws.
They should be identical, otherwise the hashes will mismatch and ROS will throw a error during execution.


## Components and Experiment Procedures

<p align="center" >
  <img style="width: 70%" src="_resources/Tactile%20Placing%20Components.jpg" />
</p>

<p align="center">
  <img style="width: 60%" src="_resources/Experiment%20Procedures.jpg" />
</p>

## Dependencies

* [urdf](https://github.com/ubi-agni/urdf) 
* [urdfdom](https://github.com/ubi-agni/urdfdom) 
* [urdfdom headers](https://github.com/ubi-agni/urdfdom_headers)
* [tactile_filters](https://github.com/ubi-agni/tactile_filters)
* [tactile_toolbox](https://github.com/ubi-agni/tactile_toolbox)
  
* [moveit@1.0.6](https://github.com/ros-planning/moveit/tree/1.0.6)
  * [eigen_stl_containers](https://github.com/ros/eigen_stl_containers)
  * [geometric_shapes@0.6.3](https://github.com/ros-planning/geometric_shapes/tree/0.6.3)
  * [random_numbers@0.3.2](https://github.com/ros-planning/random_numbers/tree/0.3.2)
  * [srdfdom@0.5.2](https://github.com/ros-planning/srdfdom/tree/0.5.2)
  * [OMPL@main](https://github.com/ompl/ompl) (TIAGo version's tag has no `package.xml`, main worked on 10.02.2023)
  * [eigenpy@v2.5.0](https://github.com/stack-of-tasks/eigenpy/tree/v2.5.0)
  * [warehouse_ros@0.9.4](https://github.com/ros-planning/warehouse_ros/tree/0.9.4)
  * `catkin config --blacklist moveit_servo`
* [moveit_msgs@0.10.1](https://github.com/ros-planning/moveit_msgs/tree/0.10.1)
  * [object_rec_msgs@0.4.1](https://github.com/wg-perception/object_recognition_msgs)
  * [octomap_msgs@0.3.5](https://github.com/OctoMap/octomap_msgs/tree/0.3.5)
  * [pal_common_msgs@0.13.2]

rviz
* [pmb2_robot@3.0.13](https://github.com/pal-robotics/pmb2_robot/tree/3.0.13)
* [tiago_robot@2.0.54](https://github.com/pal-robotics/tiago_robot/tree/2.0.54)
* [pal_gripper@1.0.3](https://github.com/pal-robotics/pal_gripper/tree/1.0.3)