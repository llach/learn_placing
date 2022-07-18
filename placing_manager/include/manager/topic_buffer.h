#pragma once

#include <mutex>
#include <ros/ros.h>

namespace placing_manager {

template<class T>
class TopicBuffer {
public:
    TopicBuffer(ros::NodeHandle nh, std::string topic);

private:
    std::mutex m_;
    std::string topic_;
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    std::vector<T&> data_;
    ros::Time lastTimestamp_;

    void callback(const T&);
};

}// namespace
