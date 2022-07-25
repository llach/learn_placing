cmake_minimum_required(VERSION 3.0.2)
project(wrist_motion)

find_package(catkin REQUIRED COMPONENTS
moveit_msgs
sensor_msgs
geometry_msgs
actionlib_msgs
message_generation
)

catkin_python_setup()

add_action_files(
  FILES
  PlanWrist.action
)

generate_messages(
  DEPENDENCIES
  moveit_msgs
  sensor_msgs
  geometry_msgs
  actionlib_msgs
)

catkin_package(CATKIN_DEPENDS)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/reorient_loop.py scripts/reorient_ac.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)