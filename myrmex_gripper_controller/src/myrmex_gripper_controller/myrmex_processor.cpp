#include "myrmex_gripper_controller/myrmex_processor.h"

#include <vector>

using namespace std;
using namespace Eigen;
using namespace myrmex_gripper_controller;

MyrmexProcessor::MyrmexProcessor(std::string suffix, int B, bool normalize, ros::NodeHandle nh)
    : suffix_(suffix), B_(B), normalize_(normalize), nh_(nh)
{
    sub_ = nh_.subscribe("/tactile_"+suffix, 0, &MyrmexProcessor::tactileCallback, this);
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

    lock_guard<mutex> l(totalLock_);
    totalForce_ = myma.sum();
}

unsigned int MyrmexProcessor::getTotalForce(){lock_guard<mutex> l(totalLock_); return totalForce_;}