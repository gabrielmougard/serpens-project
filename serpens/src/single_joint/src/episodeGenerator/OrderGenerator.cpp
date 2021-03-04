#include <ros/ros.h>
#include "single_joint_episode_generator/OrderGenerator.h"

OrderGenerator::OrderGenerator(
    ros::NodeHandle *nh,
    single_joint_episode_generator::OfflineNodeWatcher *watcher) {
        _msg_id_counter = 0;       
        _nh = nh:
        _nh->getParam("/order_generator/max_theta_m_p", _max_theta_m_p);
        _nh->getParam("/order_generator/theta_ld_min", _theta_ld_min);
        _nh->getParam("/order_generator/heta_ld_max", _theta_ld_max);
        _nh->getParam("/order_generator/theta_ld_resolution", _theta_ld_resolution);

        // Declare publisher, create publisher 'single_joint_episode_generator' using the 'JointOrder' 
	    // message file from the 'ros_tutorials_topic' package. The topic name is
	    // '/single_joint/input_to_rl/joint_order' and the size of the publisher queue is set to 100.
        _order_pub = _nh->advertise<JointOrder>("/single_joint/input_to_rl/joint_order", 100);
        _nh->serviceClient<single_joint_episode_generator::SrvTutorial>("ros_tutorial_srv");

        _feedback_sub = _nh->subscribe("/single_joint/actuator_to_input/joint_feedback", 100, OrderGenerator::feedbackHandler);
        _reset_order_srv = _nh->advertiseService("", OrderGenerator::resetOrder);
}

void OrderGenerator::run() {
    //1) The first time it is called, just generate a random input message 
    //(whose the boundaries are controlled by the parameter server)


    //2) 
}

void OrderGenerator::publishOrder() {

}

void OrderGenerator::feedbackHandler() {

}

void OrderGenerator::resetOrder() {
    
}

single_joint_episode_generator::JointOrder OrderGenerator::generate() {

}

single_joint_episode_generator::JointOrder next(
    single_joint_episode_generator::JointOrder *sent,
    single_joint_gz_simulator::JointFeedback *received) {

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