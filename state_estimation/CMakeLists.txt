cmake_minimum_required(VERSION 3.0.2)
project(state_estimation)

find_package(catkin REQUIRED COMPONENTS
)
# catkin_python_setup()

catkin_package(CATKIN_DEPENDS)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/april.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )