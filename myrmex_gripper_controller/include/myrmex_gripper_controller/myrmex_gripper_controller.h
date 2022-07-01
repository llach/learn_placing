/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2020, Bielefeld University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Bielefeld University nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Luca Lach
*/

#ifndef MYRMEX_GRIPPER_CONTROLLER_MYRMEX_GRIPPER_CONTROLLER_H
#define MYRMEX_GRIPPER_CONTROLLER_MYRMEX_GRIPPER_CONTROLLER_H

#include <mutex>
#include <limits>
#include <atomic>

#include <myrmex_gripper_controller/myrmex_processor.h>
#include <myrmex_gripper_controller/MyrmexControllerDRConfig.h>

#include <std_msgs/Bool.h>
#include <std_srvs/Empty.h>
#include <dynamic_reconfigure/server.h>
#include <trajectory_interface/quintic_spline_segment.h>
#include <joint_trajectory_controller/joint_trajectory_controller.h>


namespace myrmex_gripper_controller {

// controller states
enum CONTROLLER_STATE {TRAJECTORY_EXEC, TRANSITION, FORCE_CTRL};

// sensor/joint states
enum SENSOR_STATE {NO_CONTACT, GOT_CONTACT, IN_CONTACT, GOAL, VIOLATED};

const std::map<SENSOR_STATE, std::string> STATE_STRING = {
        {NO_CONTACT, "no contact"},
        {GOT_CONTACT, "got contact"},
        {IN_CONTACT, "in contact"},
        {GOAL, "reached goal"},
        {VIOLATED, "violated goal constraints"}
};

typedef std::unique_ptr<MyrmexProcessor> MyrmexProcPtr;

class MyrmexGripperController
: public joint_trajectory_controller::JointTrajectoryController<trajectory_interface::QuinticSplineSegment<double>,
        hardware_interface::PositionJointInterface>
{
private:
    // myrmex readout 
    int B_;
    bool closing_;
    bool normalize_;
    std::vector<std::string> suffixes_;
    std::vector<MyrmexProcPtr> mm_procs_;

    // force control
    CONTROLLER_STATE state_ = TRAJECTORY_EXEC;
    std::vector<SENSOR_STATE> sensor_states_ = {NO_CONTACT, NO_CONTACT};

    bool goalMaintain_ = true;

    double Ki_ = 0;
    double Kp_ = 0;
    double deltaQ_ = 0;
    double deltaF_ = 0;
    double error_int_ = 0;
    double max_error_ = 0;
    double threshFac_ = 1.5;

    unsigned int k_ = 0;
    unsigned int f_sum_ = 0;
    unsigned int f_target_ = 0;
    unsigned int last_f_sum_ = 0;
    
    std::vector<unsigned int> forces_ = {0, 0};
    std::vector<unsigned int> last_forces_ = {0, 0};
    std::vector<unsigned int> force_thresholds_ = {200, 200};

    std::vector<double> u_ = {0, 0};
    std::vector<double> q_T_ = {0, 0};      // joint positions at first contact 
    std::vector<double> des_q_ = {0, 0};
    std::vector<double> last_u_ = {0, 0};

    unsigned int f_sum_threshold_ = force_thresholds_[0]+force_thresholds_[1];

    ros::Publisher debugPub_;
    ros::ServiceServer killService_;
    ros::ServiceServer calibrationService_;
    std::unique_ptr<dynamic_reconfigure::Server<myrmex_gripper_controller::MyrmexControllerDRConfig>> server_;
    dynamic_reconfigure::Server<myrmex_gripper_controller::MyrmexControllerDRConfig>::CallbackType f_;

    void publishDebugInfo();
    void updateSensorStates();
    bool checkControllerTransition();

    virtual void update(const ros::Time& time, const ros::Duration& period) override;
    virtual void goalCB(GoalHandle gh) override;

    bool kill(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);
    bool calibrate(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);
    void drCallback(myrmex_gripper_controller::MyrmexControllerDRConfig &config, uint32_t level);

protected:
    bool init(hardware_interface::PositionJointInterface* hw, ros::NodeHandle& root_nh,
              ros::NodeHandle& controller_nh) override;
};

} // namespace

#endif  // MYRMEX_GRIPPER_CONTROLLER_MYRMEX_GRIPPER_CONTROLLER_IMPL_H
