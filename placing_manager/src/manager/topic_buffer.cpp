#include <manager/topic_buffer.h>

using namespace std;
using namespace ros;
using namespace placing_manager;

template <class T>
TopicBuffer<T>::TopicBuffer(NodeHandle nh, string topic) : 
    nh_(nh),
    topic_(topic)
{

}

template <class T>
void TopicBuffer<T>::callback(const T& msg)
{

}