#include <manager/topic_buffer.h>

using namespace std;
using namespace ros;
using namespace placing_manager;

template <class T>
TopicBuffer<T>::TopicBuffer(NodeHandle nh, string topic, string name) : 
    nh_(nh),
    name_(name),
    topic_(topic),
    paused_(true)
{
    sub_ = nh_.subscribe(topic_, 1, &TopicBuffer<T>::callback, this);
}

template <class T>
void TopicBuffer<T>::callback(const T& msg)
{
    if (paused_) return;

    std::lock_guard<std::mutex> l(m_);
    data_.push_back(msg);
    lastTimestamp_ = ros::Time::now();
}