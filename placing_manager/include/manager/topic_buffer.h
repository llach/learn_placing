#pragma once

#include <mutex>
#include <atomic>

#include <ros/ros.h>

namespace placing_manager {

template<class T>
class TopicBuffer {
public:
    TopicBuffer(ros::NodeHandle nh, std::string topic, std::string name);

private:
    std::mutex m_;
    std::string name_;
    std::string topic_;
    ros::NodeHandle nh_;
    ros::Subscriber sub_;

    std::vector<T&> data_;
    ros::Time lastTimestamp_;
    std::atomic<bool> paused_;

    void callback(const T&);
};

}// namespace
