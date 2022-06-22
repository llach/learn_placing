cmake_minimum_required(VERSION 2.8.3)
project(myrmex_gripper_controller)

# Default to C++14
if ("${CMAKE_VERSION}" VERSION_LESS "3.1")
    add_compile_options(-std=c++14)
else ()
    set(CMAKE_CXX_STANDARD 14)
endif ()

set(DISABLE_PAL_FLAGS "TRUE")

find_package(Eigen3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        tactile_msgs
        controller_interface
        joint_trajectory_controller
        )

catkin_package(
        CATKIN_DEPENDS
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME}_lib
)

include_directories(include ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

add_library(${PROJECT_NAME}_lib
        include/myrmex_gripper_controller/myrmex_processor.h
        include/myrmex_gripper_controller/myrmex_gripper_controller.h

        src/myrmex_gripper_controller/myrmex_processor.cpp
        src/myrmex_gripper_controller/myrmex_gripper_controller.cpp
        )

target_link_libraries(${PROJECT_NAME}_lib
        ${catkin_LIBRARIES}
        ${EIGEN3_LIBRARIES}
        )

add_executable(mm_sum
        src/mm_sum.cpp
)

target_link_libraries(mm_sum
        ${PROJECT_NAME}_lib
        ${catkin_LIBRARIES}
        ${EIGEN3_LIBRARIES}
)

# Install
install(DIRECTORY include
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

# Install library
install(TARGETS ${PROJECT_NAME}_lib
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
        )

install(DIRECTORY config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )

install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )

install(FILES ${PROJECT_NAME}_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
