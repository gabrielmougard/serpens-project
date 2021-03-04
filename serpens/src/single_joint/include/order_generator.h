#ifndef ORDER_GENERATOR_H
#define ORDER_GENERATOR_H

#include <ros/ros.h>
#include "single_joint/PositionOrder.h"
#include "single_joint/PositionFeedback.h"

namespace single_joint {
namespace {
class OrderGenerator {
    public:
        OrderGenerator(ros::NodeHandle *nh, OfflineNodeWatcher *watcher);
        ~OrderGenerator();
        void run();
    private:
        int _msg_id_counter; 
        float _max_theta_m_p;
        float _theta_ld_min;
        float _theta_ld_max;
        float _theta_ld_resolution;

        ros::NodeHandle *_nh;
        ros::ServiceClient _order_pub;
        ros::ServiceServer _feedback_sub;
        ros::ServiceServer _reset_order;
        
        void feedbackHandler(const PositionFeedback::ConstPtr& msg);
        void publishOrder(PositionOrder *order);
        PositionOrder generate();
        PositionOrder next(PositionOrder *sent, PositionFeedback *received);
};
} // namespace
} // namespace single_joint

#endif // ORDER_GENERATOR_H