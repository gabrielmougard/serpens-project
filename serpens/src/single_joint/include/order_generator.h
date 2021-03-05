#ifndef ORDER_GENERATOR_H
#define ORDER_GENERATOR_H

#include <ros/ros.h>
#include "single_joint/PositionOrder.h"
#include "single_joint/PositionFeedback.h"

namespace single_joint {
namespace {
class OrderGenerator {
    public:
        OrderGenerator(ros::NodeHandle *nh);
        ~OrderGenerator();
        void Run();
    private: 
        // Order boundaries
        float theta_ld_min_;
        float theta_ld_max_;
        float theta_ld_resolution_;
        float theta_l_min_;
        float theta_l_max_;
        float theta_l_resolution_;
        float theta_m_min_;
        float theta_m_max_;
        float theta_m_resolution_;
        float theta_m_p_min_;
        float theta_m_p_max_;
        float theta_m_p_resolution_;
        // Position and Feedback order
        PositionOrder *position_order_;
        int position_order_count_;
        PositionFeedback *position_feedback_;
        int position_feedback_count_;
        // ROS related
        ros::NodeHandle *nh_;
        ros::ServiceClient position_order_client_;
        ros::ServiceServer position_feedback_server_;
        // Methods
        void FeedbackHandler(PositionFeedback::Request &req, PositionFeedback::Response &res);
        void SendOrder();
        PositionOrder Generate();
        PositionOrder Next(PositionOrder &sent, PositionFeedback &received);
};
} // namespace
} // namespace single_joint

#endif // ORDER_GENERATOR_H