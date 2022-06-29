#include "mutex"
#include "atomic"
#include "ros/ros.h"
#include "eigen3/Eigen/Core"
#include "tactile_msgs/TactileState.h"

namespace myrmex_gripper_controller {

// constants
const int MAX_READOUT_VALUE = 4095;
const int EDGE_LENGTH = 16;
const int MAX_IDX = EDGE_LENGTH - 1;

class MyrmexProcessor
{
public:
    MyrmexProcessor(std::string suffix, int B = 0, bool normalize = false, ros::NodeHandle nh = ros::NodeHandle());

    std::atomic<bool> is_calibrated = { false };

    void startCalibration();

    unsigned int getBias();
    unsigned int getTotalForce();

private:
    // params
    int B_;
    bool normalize_;
    std::string suffix_;

    // internal members
    std::string name_;
    std::mutex totalLock_;
    unsigned int bias_ = 0;
    unsigned int totalForce_ = 0;

    unsigned short nSamples_ = 0;
    std::atomic<bool> calibrate_ = { false };
    std::vector<int> calibrationSamples_ = std::vector<int>(4000);

    // ros members
    ros::NodeHandle nh_;
    ros::Subscriber sub_;

    void tactileCallback(const tactile_msgs::TactileState::ConstPtr& msg);
}; // MyrmexProcessor

} // namespace