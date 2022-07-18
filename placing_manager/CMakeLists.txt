cmake_minimum_required(VERSION 3.0.2)

project(placing_manager)
add_compile_options(-std=c++14)


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_srvs
  actionlib
  sensor_msgs
  tactile_msgs
  control_msgs
  trajectory_msgs
  controller_manager_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
 CATKIN_DEPENDS roscpp std_srvs actionlib sensor_msgs tactile_msgs control_msgs trajectory_msgs controller_manager_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    SYSTEM
    ${catkin_INCLUDE_DIRS}
)
include_directories(
    include
)

# Declare a C++ library
add_library(${PROJECT_NAME}_lib
src/manager/manager.cpp
src/manager/topic_buffer.cpp
)

target_link_libraries(${PROJECT_NAME}_lib ${catkin_LIBRARIES})
add_dependencies(${PROJECT_NAME}_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Declare a C++ executable
# With catkin_make all packages are built within a single CMake context
# The recommended prefix ensures that target names across packages don't collide
add_executable(node src/node.cpp)

# Add cmake target dependencies of the executable
# same as for the library above
add_dependencies(node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Specify libraries to link a library or executable target against
target_link_libraries(node
  ${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
)

#############
## Install ##
#############

# Mark executables for installation
# See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
install(TARGETS node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)