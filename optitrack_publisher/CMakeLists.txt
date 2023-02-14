cmake_minimum_required(VERSION 3.0.2)
project(optitrack_publisher)

find_package(catkin REQUIRED COMPONENTS
geometry_msgs
)

catkin_python_setup()

catkin_package(CATKIN_DEPENDS)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/ot_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# install(DIRECTORY launch
#         DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )