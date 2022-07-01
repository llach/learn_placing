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

#include <myrmex_gripper_controller/MyrmexControllerDebug.h>
#include <myrmex_gripper_controller/myrmex_gripper_controller.h>

namespace myrmex_gripper_controller {

bool MyrmexGripperController::init(hardware_interface::PositionJointInterface* hw,
                                   ros::NodeHandle& root_nh, ros::NodeHandle& controller_nh) {
    name_ = "TORSTOPC";
    ROS_INFO_NAMED(name_, "Initializing MyrmexGripperController ...");
    bool ret = 0;
    try {
        ret = JointTrajectoryController::init(hw, root_nh, controller_nh);
    } catch(std::runtime_error& e) {
        ROS_ERROR_STREAM_NAMED(name_, "Could not init JTC: " << e.what());
    }
    // print verbose errors
    this->verbose_ = true;

    ROS_INFO_NAMED(name_, "reloading action server ...");
    action_server_.reset(new ActionServer(controller_nh_, "follow_joint_trajectory",
                                            std::bind(&MyrmexGripperController::goalCB,   this, std::placeholders::_1),
                                            std::bind(&MyrmexGripperController::cancelCB, this, std::placeholders::_1),
                                            false));
    action_server_->start();

    ROS_INFO_NAMED(name_, "starting Myrmex listeners");

    if (!controller_nh_.getParam("B", B_)){{ROS_FATAL_NAMED(name_, "B missing");return false;}}
    if (!controller_nh_.getParam("normalize", normalize_)){{ROS_FATAL_NAMED(name_, "normalize missing");return false;}}

    suffixes_ = joint_trajectory_controller::internal::getStrings(controller_nh_, "suffixes");
    if (suffixes_.empty()) {ROS_FATAL_NAMED(name_, "suffixes missing");return false;}

    for (auto& suf : suffixes_){
      mm_procs_.push_back(std::make_unique<MyrmexProcessor>(suf, B_, normalize_));
    }

    ROS_INFO_NAMED(name_, "registering dynamic reconfigure callback ... ");
    server_ = std::make_unique<dynamic_reconfigure::Server<myrmex_gripper_controller::MyrmexControllerDRConfig>>(controller_nh);
    f_ = std::bind(&MyrmexGripperController::drCallback, this, std::placeholders::_1, std::placeholders::_2);
    server_->setCallback(f_);

    debugPub_ = root_nh.advertise<myrmex_gripper_controller::MyrmexControllerDebug>("/myrmex_controller_debug", 1);
    killService_ = controller_nh.advertiseService("kill", &MyrmexGripperController::kill, this);
    calibrationService_ = controller_nh.advertiseService("calibrate", &MyrmexGripperController::calibrate, this);

    ROS_INFO_NAMED(name_, "MyrmexGripperController init done! Started %ld processors.", mm_procs_.size());
    return ret;
}

bool MyrmexGripperController::checkControllerTransition(){
    for (auto& ss : sensor_states_)
        if (ss < GOT_CONTACT) return false;
    return true;
}

void MyrmexGripperController::updateSensorStates(){
    // NOTE we assume, that we don't loose contact once we acquired it -> there's no transition back from an in-contact state to no contact
    // usually, this doesn't happen in practice and if it does, some high-level component needs to analyse the situation and trigger replanning etc.
    // before trying to grasp again.
  for (int i = 0; i<forces_.size(); i++){
    SENSOR_STATE prevState = sensor_states_[i];

    if (sensor_states_[i] < GOAL) // if a joint has reached it's goal, we don't change its sensor state anymore
    { 
      if (last_forces_[i] <= force_thresholds_[i] && 
          forces_[i]      >  force_thresholds_[i]) 
      {
        sensor_states_[i] = GOT_CONTACT;
      } 
      else if (last_forces_[i] > force_thresholds_[i] && 
               forces_[i]      > force_thresholds_[i]) 
      {
        sensor_states_[i] = IN_CONTACT;
      }
    }

    if (sensor_states_[i] != prevState){
        if (prevState == NO_CONTACT) q_T_[i] = joints_[i].getPosition(); // we store the joint position at time of first contact
        ROS_DEBUG_STREAM_NAMED(name_, "myrmex_" << suffixes_[i] << " changed state from " 
        << STATE_STRING.at(prevState) << " to " << STATE_STRING.at(sensor_states_[i])
        << " with force " << forces_[i] << " and threshold " << force_thresholds_[i]); //
    }
  }
}

void MyrmexGripperController::update(const ros::Time& time, const ros::Duration& period) 
{
  realtime_busy_ = true;
  // Get currently followed trajectory
  TrajectoryPtr curr_traj_ptr;
  curr_trajectory_box_.get(curr_traj_ptr);
  Trajectory& curr_traj = *curr_traj_ptr;

  // Update time data
  TimeData time_data;
  time_data.time   = time;                                     // Cache current time
  time_data.period = period;                                   // Cache current control period
  time_data.uptime = time_data_.readFromRT()->uptime + period; // Update controller uptime
  time_data_.writeFromNonRT(time_data); // TODO: Grrr, we need a lock-free data structure here!

  for (int i = 0; i<forces_.size(); i++){
    forces_[i] = mm_procs_[i]->getTotalForce();
  }
  f_sum_ = forces_[0]+forces_[1];

  updateSensorStates();

  // update controller state
  if (state_ == CONTROLLER_STATE::TRANSITION) {
    // if the transition was detected last time, we enter force_control from here onwards
    state_ = CONTROLLER_STATE::FORCE_CTRL;
  } else if (state_ == CONTROLLER_STATE::TRAJECTORY_EXEC) {
    if (checkControllerTransition() && closing_) 
    {
      ROS_INFO_NAMED(name_, "CONTROLLER TRANSITION");
      state_ = CONTROLLER_STATE::TRANSITION;
    }
  }

  // NOTE: It is very important to execute the two above code blocks in the specified sequence: first get current
  // trajectory, then update time data. Hopefully the following paragraph sheds a bit of light on the rationale.
  // The non-rt thread responsible for processing new commands enqueues trajectories that can start at the _next_
  // control cycle (eg. zero start time) or later (eg. when we explicitly request a start time in the future).
  // If we reverse the order of the two blocks above, and update the time data first; it's possible that by the time we
  // fetch the currently followed trajectory, it has been updated by the non-rt thread with something that starts in the
  // next control cycle, leaving the current cycle without a valid trajectory.

  // Update current state and state error
  if (rt_active_goal_ && state_ != FORCE_CTRL)
  {
    for (unsigned int i = 0; i < joints_.size(); ++i)
    {
      current_state_.position[i] = joints_[i].getPosition();
      current_state_.velocity[i] = joints_[i].getVelocity();
      // There's no acceleration data available in a joint handle

      typename TrajectoryPerJoint::const_iterator segment_it = sample(curr_traj[i], time_data.uptime.toSec(), desired_joint_state_);
      if (curr_traj[i].end() == segment_it)
      {
        // Non-realtime safe, but should never happen under normal operation
        ROS_ERROR_NAMED(name_,
                        "Unexpected error: No trajectory defined at current time. Please contact the package maintainer.");
        return;
      }

      // don't move joint if it's in contact when closing the gripper 
      if (sensor_states_[i] > NO_CONTACT && closing_) desired_joint_state_.position[0] = q_T_[i];
      des_q_[i] = desired_joint_state_.position[0]; // debug should always show the desired value

      desired_state_.position[i] = desired_joint_state_.position[0];
      desired_state_.velocity[i] = desired_joint_state_.velocity[0];
      desired_state_.acceleration[i] = desired_joint_state_.acceleration[0]; ;

      state_joint_error_.position[0] = angles::shortest_angular_distance(current_state_.position[i],desired_joint_state_.position[0]);
      state_joint_error_.velocity[0] = desired_joint_state_.velocity[0] - current_state_.velocity[i];
      state_joint_error_.acceleration[0] = 0.0;

      state_error_.position[i] = angles::shortest_angular_distance(current_state_.position[i],desired_joint_state_.position[0]);
      state_error_.velocity[i] = desired_joint_state_.velocity[0] - current_state_.velocity[i];
      state_error_.acceleration[i] = 0.0;

      //Check tolerances
      const RealtimeGoalHandlePtr rt_segment_goal = segment_it->getGoalHandle();
      if (rt_segment_goal && rt_segment_goal == rt_active_goal_)
      {
        // Check tolerances
        if (time_data.uptime.toSec() < segment_it->endTime())
        {
          // Currently executing a segment: check path tolerances
          const joint_trajectory_controller::SegmentTolerancesPerJoint<Scalar>& joint_tolerances = segment_it->getTolerances();
          if (!checkStateTolerancePerJoint(state_joint_error_, joint_tolerances.state_tolerance))
          {
            if (verbose_)
            {
              ROS_ERROR_STREAM_NAMED(name_,"Path tolerances failed for joint: " << joint_names_[i]);
              checkStateTolerancePerJoint(state_joint_error_, joint_tolerances.state_tolerance, true);
            }

            if(rt_segment_goal && rt_segment_goal->preallocated_result_)
            {
              rt_segment_goal->preallocated_result_->error_code =
              control_msgs::FollowJointTrajectoryResult::PATH_TOLERANCE_VIOLATED;
              rt_segment_goal->setAborted(rt_segment_goal->preallocated_result_);
              rt_active_goal_.reset();
              successful_joint_traj_.reset();
            }
            else{
              ROS_ERROR_STREAM("rt_segment_goal->preallocated_result_ NULL Pointer");
            }
          }
        }
        else if (segment_it == --curr_traj[i].end())
        {
          if (verbose_)
            ROS_DEBUG_STREAM_THROTTLE_NAMED(1,name_,"Finished executing last segment, checking goal tolerances");

          // Controller uptime
          const ros::Time uptime = time_data_.readFromRT()->uptime;

          // Checks that we have ended inside the goal tolerances
          const joint_trajectory_controller::SegmentTolerancesPerJoint<Scalar>& tolerances = segment_it->getTolerances();
          const bool inside_goal_tolerances = checkStateTolerancePerJoint(state_joint_error_, tolerances.goal_state_tolerance);

          if (inside_goal_tolerances)
          {
            successful_joint_traj_[i] = 1;
          }
          else if (uptime.toSec() < segment_it->endTime() + tolerances.goal_time_tolerance)
          {
            // Still have some time left to meet the goal state tolerances
          }
          else
          {
            if (verbose_)
            {
              ROS_ERROR_STREAM_NAMED(name_,"Goal tolerances failed for joint: "<< joint_names_[i]);
              // Check the tolerances one more time to output the errors that occurs
              checkStateTolerancePerJoint(state_joint_error_, tolerances.goal_state_tolerance, true);
            }

            if(rt_segment_goal){
              rt_segment_goal->preallocated_result_->error_code = control_msgs::FollowJointTrajectoryResult::GOAL_TOLERANCE_VIOLATED;
              rt_segment_goal->setAborted(rt_segment_goal->preallocated_result_);
            }
            else
            {
              ROS_ERROR_STREAM("rt_segment_goal->preallocated_result_ NULL Pointer");
            }
            rt_active_goal_.reset();
            successful_joint_traj_.reset();
          }
        }
      }
    }
  }

  if (rt_active_goal_ && state_ == FORCE_CTRL)
  {
    double dt = period.toSec();

    deltaF_ = ((f_target_ - f_sum_) / k_);

    error_int_ += deltaF_ * dt;
    if (error_int_ > max_error_)
      error_int_ = max_error_;
    if (error_int_ < -max_error_)
      error_int_ = -max_error_;

   if(last_f_sum_ >= f_sum_threshold_ && f_sum_ < f_sum_threshold_){
     ROS_INFO_NAMED(name_, "reset ADD error integral");
     error_int_ = 0.0;
   }

    deltaQ_ = Kp_ * deltaF_ + Ki_ * error_int_;

    for (unsigned int i = 0; i < joints_.size(); ++i) {

      // There's no acceleration data available in a joint handle
      current_state_.position[i] = joints_[i].getPosition();
      current_state_.velocity[i] = joints_[i].getVelocity();
      
      u_[i] = -deltaQ_/2;
      des_q_[i] = current_state_.position[i] + u_[i];

      desired_joint_state_.position[0] = des_q_[i];
      desired_joint_state_.velocity[0] = (u_[i] - last_u_[i]) / (dt);

      desired_state_.position[i] = desired_joint_state_.position[0];
      desired_state_.velocity[i] = desired_joint_state_.velocity[0];
      desired_state_.acceleration[i] = desired_joint_state_.acceleration[0];


      state_joint_error_.position[0] =
              angles::shortest_angular_distance(current_state_.position[i], desired_joint_state_.position[0]);
      state_joint_error_.velocity[0] = desired_joint_state_.velocity[0] - current_state_.velocity[i];
      state_joint_error_.acceleration[0] = 0.0;

      state_error_.position[i] =
              angles::shortest_angular_distance(current_state_.position[i], desired_joint_state_.position[0]);
      state_error_.velocity[i] = desired_joint_state_.velocity[0] - current_state_.velocity[i];
      state_error_.acceleration[i] = 0.0;
    }
  }

  //If there is an active goal and all segments finished successfully then set goal as succeeded
  RealtimeGoalHandlePtr current_active_goal(rt_active_goal_);
  if (current_active_goal && current_active_goal->preallocated_result_ && successful_joint_traj_.count() == joints_.size())
  {
    current_active_goal->preallocated_result_->error_code = control_msgs::FollowJointTrajectoryResult::SUCCESSFUL;
    current_active_goal->setSucceeded(current_active_goal->preallocated_result_);
    rt_active_goal_.reset();
    successful_joint_traj_.reset();
  } 

  // Hardware interface adapter: Generate and send commands
  hw_iface_adapter_.updateCommand(time_data.uptime, time_data.period,
                                  desired_state_, state_error_);

  // Set action feedback
  if (rt_active_goal_ && rt_active_goal_->preallocated_feedback_)
  {
    rt_active_goal_->preallocated_feedback_->header.stamp          = time_data_.readFromRT()->time;
    rt_active_goal_->preallocated_feedback_->desired.positions     = desired_state_.position;
    rt_active_goal_->preallocated_feedback_->desired.velocities    = desired_state_.velocity;
    rt_active_goal_->preallocated_feedback_->desired.accelerations = desired_state_.acceleration;
    rt_active_goal_->preallocated_feedback_->actual.positions      = current_state_.position;
    rt_active_goal_->preallocated_feedback_->actual.velocities     = current_state_.velocity;
    rt_active_goal_->preallocated_feedback_->error.positions       = state_error_.position;
    rt_active_goal_->preallocated_feedback_->error.velocities      = state_error_.velocity;
    rt_active_goal_->setFeedback( rt_active_goal_->preallocated_feedback_ );
  }

  // update state storage
  for (int i = 0; i<forces_.size(); i++){
    last_u_[i] = u_[i];
    last_forces_[i] = forces_[i];
  }
  last_f_sum_ = f_sum_;

  // Publish state
  publishState(time_data.uptime);
  publishDebugInfo();
  realtime_busy_ = false;
}
  
void MyrmexGripperController::goalCB(GoalHandle gh) 
{
  ROS_INFO_NAMED(name_, "Received new action goal");
  closing_ = gh.getGoal()->trajectory.points.back().positions[0] < joints_[0].getPosition();
  if (closing_) {
    ROS_INFO_NAMED(name_, "closing gripper -> stopping on contact");
  } else {
    ROS_INFO_NAMED(name_, "opening to %f", gh.getGoal()->trajectory.points.back().positions[0]);
  }
  
  // reset internal parameters
  q_T_ = {0, 0};
  state_ = TRAJECTORY_EXEC;
  sensor_states_ = {NO_CONTACT, NO_CONTACT};

  JointTrajectoryController::goalCB(gh);
}

bool MyrmexGripperController::kill(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res)
{
    RealtimeGoalHandlePtr current_active_goal(rt_active_goal_);

    if (current_active_goal) {
        ROS_INFO_NAMED(name_, "Killing current goal ...");
        cancelCB(current_active_goal->gh_);
    } else {
        ROS_INFO_NAMED(name_, "No goal to kill!");
    }

    return true;
}

bool MyrmexGripperController::calibrate(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res)
{
    RealtimeGoalHandlePtr current_active_goal(rt_active_goal_);
    if (current_active_goal) {
        ROS_INFO_NAMED(name_, "Cannot calibrate sensors during goal execution");
        return false;
    }

    ROS_INFO_NAMED(name_, "starting sensor calibration process ...");
    mm_procs_[0]->startCalibration();
    mm_procs_[1]->startCalibration();

    // wait for calibration to be done
    while (!(mm_procs_[0]->is_calibrated && mm_procs_[1]->is_calibrated)) ros::Rate(5).sleep();
    
    // update thresholds based on the maximum observed value during the calibration process
    force_thresholds_[0] = static_cast<unsigned int>(mm_procs_[0]->maxDeviation * threshFac_);
    force_thresholds_[1] = static_cast<unsigned int>(mm_procs_[1]->maxDeviation * threshFac_);

    return true;
}

void MyrmexGripperController::drCallback(myrmex_gripper_controller::MyrmexControllerDRConfig &config, uint32_t level)
{
    ROS_INFO_NAMED(name_, "got reconfigure request!");

    k_ = config.k;
    Ki_ = config.Ki;
    Kp_ = config.Kp;
    f_target_ = config.f_target;
    goalMaintain_ = config.goal_maintain;
    force_thresholds_ = {config.force_threshold, config.force_threshold};
}

void MyrmexGripperController::publishDebugInfo()
{
    MyrmexControllerDebug mcd;
    mcd.header.stamp = ros::Time::now();

    mcd.k = k_;
    mcd.Ki = Ki_;
    mcd.Kp = Kp_;
    mcd.f_sum = f_sum_;
    mcd.delta_q = deltaQ_;
    mcd.delta_f = deltaF_;
    mcd.f_target = f_target_;
    mcd.max_error = max_error_;
    mcd.error_integral = error_int_;

    mcd.f = forces_;
    mcd.q_T = q_T_;
    mcd.des_q = des_q_;
    mcd.f_thresholds = force_thresholds_;

    mcd.bias = {mm_procs_[0]->getBias(), mm_procs_[1]->getBias()};

    debugPub_.publish(mcd);
}

} // namespace myrmex_gripper_controller

// Pluginlib
#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(myrmex_gripper_controller::MyrmexGripperController,
                       controller_interface::ControllerBase)