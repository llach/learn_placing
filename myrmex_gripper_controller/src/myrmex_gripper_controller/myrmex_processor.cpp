#include "myrmex_gripper_controller/myrmex_processor.h"

#include <cmath>
#include <vector>

using namespace std;
using namespace Eigen;
using namespace myrmex_gripper_controller;

MyrmexProcessor::MyrmexProcessor(std::string suffix, int B, bool normalize, ros::NodeHandle nh)
    : suffix_(suffix), B_(B), normalize_(normalize), nh_(nh)
{
    sub_ = nh_.subscribe("/tactile_"+suffix, 0, &MyrmexProcessor::tactileCallback, this);
    name_ = "myrmex_"+suffix+"_processor";
};

void MyrmexProcessor::tactileCallback(const tactile_msgs::TactileState::ConstPtr& msg)
{
    // convert sensor values to Eigen; Note: values coming in are ints represented as float. we convert them here back to ints
    std::vector<int> v(msg->sensors[0].values.begin(), msg->sensors[0].values.end());
    MatrixXi rawMyma = Map<MatrixXi>(v.data(), 16, 16);

    // zero force on a taxel -> readout is the maximum value. 
    // applying force reduces the value read. since, for us, force is increasing, we revert this here
    MatrixXi mymaAdjusted = MAX_READOUT_VALUE - rawMyma.array();

    // transform data such that it matches the physical sensor layout (it's just the transpose)
    MatrixXi myma = mymaAdjusted.transpose();
    
    // we allow for the processing to be done on a submatrix only
    // sometimes the values on the edges can give false positives if the foam is touching
    // hence this addition
    if(B_ > 0 && 2*B_<EDGE_LENGTH-2) {
        int LEN = EDGE_LENGTH - (2*B_);
        myma = myma.block(B_,B_, LEN, LEN);
    }

    // for debugging
    // cout << myma(0,0) <<  " | " << myma(myma.rows()-1, myma.cols()-1) << endl;

    {
        lock_guard<mutex> l(totalLock_);
        float msum = static_cast<float>(myma.sum());

        // avoid underflow of unsigned int
        totalForce_ = (msum/std::pow(10,4)) - bias_;
    }

    // calibration procedure. can be started by calling the `startCalibration()` member function 
    if (calibrate_)
    {
        if (nSamples_ == calibrationSamples_.size()-1) // collection done
        {
            lock_guard<mutex> l(totalLock_);

            // calculate bias
            bias_ = 0; 
            for (float s : calibrationSamples_) bias_ += s;
            bias_ = bias_/calibrationSamples_.size();

            maxDeviation -= bias_; // since maxDev is the maximum value of calibrationSamples_, this can never be negative -> no explicit check.

            calibrate_ = false;
            is_calibrated = true;

            std::cout << name_ << " " << bias_ << std::endl;
            ROS_INFO_NAMED(name_, "calibration done, bias: %f", bias_);
        } 
        else 
        {
            if (totalForce_ > maxDeviation) maxDeviation = totalForce_;
            calibrationSamples_[nSamples_] = totalForce_;
        }
        nSamples_++;
    } 
}

void MyrmexProcessor::startCalibration(){nSamples_ = 0; bias_ = 0; maxDeviation = 0; calibrate_ = true; is_calibrated = false;}

float MyrmexProcessor::getBias(){lock_guard<mutex> l(totalLock_); return bias_;}
float MyrmexProcessor::getTotalForce(){lock_guard<mutex> l(totalLock_); return totalForce_;}
