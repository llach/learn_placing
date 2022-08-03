#pragma once

#include <mutex>
#include <atomic>

#include <ros/ros.h>
#include <rosbag/bag.h>

namespace placing_manager {

template<class T>
class TopicBuffer {
public:
    TopicBuffer(ros::NodeHandle nh, const std::string &name, const std::string &topic) : 
        nh_(nh),
        name_(name),
        topic_(topic),
        paused_(true)
    {
        sub_ = nh_.subscribe(topic_, 1, &TopicBuffer<T>::callback, this);
    };

    void storeData(rosbag::Bag &bag, const ros::Time &fromTime, const ros::Time &toTime) {
        std::lock_guard<std::mutex> l(m_);
        for (size_t i = 0; i<data_.size(); i++){
            if (times_[i] > fromTime && times_[i] < toTime){
                bag.write(name_, times_[i], data_[i]);
            }
        }
        times_.clear();
        data_.clear();
    }

    bool isFresh(const ros::Time &n){
        std::lock_guard<std::mutex> l(m_);
        if (lastTimestamp_ < n) {
            ROS_WARN_STREAM(name_ << " not fresh");
            return false;
        }
        ROS_DEBUG_STREAM(name_ << " fresh at " << lastTimestamp_);
        return true;
    }

    int numData(){
        if (data_.size()==0) ROS_INFO("%s has no data!", name_);
        return (int) data_.size();
    }

    std::mutex m_;
    std::vector<T> data_;
    std::atomic<bool> paused_;
    std::vector<ros::Time> times_;

private:
    std::string name_;
    std::string topic_;
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Time lastTimestamp_ = ros::Time(0);

    void callback(const boost::shared_ptr<T const> &msg){
        std::lock_guard<std::mutex> l(m_);
        ros::Time t = ros::Time::now();
        lastTimestamp_ = t;

        if (paused_) return;

        data_.push_back(*msg);
        times_.push_back(t);
    }
};

}// namespace
