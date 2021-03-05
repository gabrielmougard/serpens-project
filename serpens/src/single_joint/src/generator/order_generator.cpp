#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits>

#include <ros/ros.h>

#include "single_joint/order_generator.h"

namespace single_joint {
namespace {

OrderGenerator::OrderGenerator(ros::NodeHandle *nh) {      
    nh_ = nh:
    
    nh_->getParam("/order_generator/theta_ld_min", theta_ld_min_);
    nh_->getParam("/order_generator/heta_ld_max", theta_ld_max_);
    nh_->getParam("/order_generator/theta_ld_resolution", theta_ld_resolution_);
    nh_->getParam("/order_generator/theta_l_min", theta_l_min_);
    nh_->getParam("/order_generator/heta_l_max", theta_l_max_);
    nh_->getParam("/order_generator/theta_l_resolution", theta_l_resolution_);
    nh_->getParam("/order_generator/theta_m_min", theta_m_min_);
    nh_->getParam("/order_generator/theta_m_max", theta_m_max_);
    nh_->getParam("/order_generator/theta_m_resolution", theta_m_resolution_);
    nh_->getParam("/order_generator/theta_m_p_min", theta_m_p_min_);
    nh_->getParam("/order_generator/theta_m_p_max", theta_m_p_max_);
    nh_->getParam("/order_generator/theta_m_p_resolution", theta_m_p_resolution_);
 

    position_order_client_ = nh_->serviceClient<PositionOrder>("position_order");
    position_feedback_server_ = nh_->advertiseService("position_feedback", OrderGenerator::FeedbackHandler);

    position_order_ = NULL;
    position_order_count_ = 0;
    position_feedback_ = NULL;
    position_feedback_count_ = 0;

}

void OrderGenerator::Run() {
    // Only the first time the node is initialized.
    if (position_order_ == NULL && position_feedback_ == NULL) {
        position_order_ = OrderGenerator::Generate();
        OrderGenerator::SendOrder();
    }

    ros::spin();
}

void OrderGenerator::SendOrder() {
    if (position_order_client_.call(&position_order_)) {
        position_order_count_ += 1;
    } else {
        ROS_INFO("The position order service client can't reach the server.");
    }
}

void OrderGenerator::FeedbackHandler(PositionFeedback::Request &req, PositionFeedback::Response &res) {
    position_feedback_->request.stamp = req.stamp;
    position_feedback_->request.msg_id = req.msg_id;
    position_feedback_->request.theta_m_update = req.theta_m_update;
    position_feedback_->request.theta_l_update = req.theta_l_update;
    position_feedback_->request.should_reset = req.should_reset;
    
    ROS_INFO("
        PositionFeedback received :\n
        \tstamp : %d,\n
        \tmsg_id : %d,\n
        \ttheta_m_update: %f,\n
        \ttheta_l_update: %f,\n
        \tshoudle_reset: %d,\n
    ", (int)req.stamp, (int)req.msg_id, (float)req.theta_m_update,
    (float)req.theta_l_update, (int)req.should_reset);

    if (req.should_reset) {
        position_feedback_ = NULL;
        position_order_ = OrderGenerator::Generate();
        OrderGenerator::SendOrder();
    } else {
        position_order_ = OrderGenerator::Next();
        OrderGenerator::SendOrder();
    }
}


PositionOrder OrderGenerator::Generate() {
    /*
    Random initialization of the input parameters.
    - 'theta_ld' : will be fixed during the whole episode ('epsilon' is computed using 'theta_ld' and the moving value of 'theta_l').
    - 'theta_l' : will be randomly initialized (a joint is not necessarily straight at the beginning) but will vary through time.
    - 'theta_m' : same as 'theta_l'.
    - 'theta_m_p' ; we randomly decide of which rotationnal speed we want at the beginning. It may vary through time as well.
    */
    float rand_theta_ld = RandomSample(theta_ld_min_, theta_ld_max_, theta_ld_resolution_);
    float rand_theta_l = RandomSample(theta_l_min_, theta_l_max_, theta_l_resolution_);
    float rand_theta_m = RandomSample(theta_m_min_, theta_m_max_, theta_m_resolution_);
    float rand_theta_m_p = RandomSample(theta_m_p_min_, theta_m_p_max_, theta_m_p_resolution_);
    
    // set the values of the differential parameters.
    float theta_l_p = numeric_limits<float>::max();
    float epsilon = numeric_limits<float>::max();
    float epsilon_p = numeric_limits<float>::max();

    // return the Service to send    
    PositionOrder order;
    order.request.stamp = ros::Time::now();
    order.request.msg_id = position_order_count_;
    order.request.theta_ld = rand_theta_ld;
    order.request.theta_l = rand_theta_l;
    order.request.theta_l_p = theta_l_p;
    order.request.theta_m = rand_theta_m;
    order.request.theta_m_p = rand_theta_m_p;
    order.request.epsilon = epsilon;
    order.request.epsilon_p = epsilon_p;

    return order;
}

PositionOrder OrderGenerator::Next() {

        int delta = sent->stamp.nsec - received->stamp.nsec;
        single_joint_episode_generator::JointOrder next;
        _msg_id_counter += 1;
        next.time = ros::Time::now();
        next.msg_id = _msg_id_counter;
        next.theta_ld = sent->theta_ld; // This is the target parameter so it does not change
        next.theta_l = received->theta_l_update;
        next.theta_l_p = (received->theta_l_update - sent->theta_l) / (float)delta;
        next.theta_m = received->theta_m_update;
        next.theta_m_p = (received->theta_m_update - sent->theta_m) / (float)delta;
        next.epsilon = sent->theta_ld - received->theta_l_update;
        next.epsilon_p = ((sent->theta_ld - received->theta_l_update) - sent->epsilon) / (float)delta;
        return next;

}

} // namespace
} // namespace single_joint

float RandomSample(float min_val, float max_val, float resolution) {
    srand(time(NULL)); //initialize the random seed
    int l = (int)((max_val - min_val) / resolution);
    float dist[l];
    for (int i = 0; i < l; i++) {
        dist[i] =  min_val + (float)i*resolution;
    }
    int rand_idx = rand() % l; //generates a random number between 0 and l
    return dist[l];
}