#include <ros/ros.h>
#include "single_joint_episode_generator/JointOrder.h"
#include "single_joint_gz_simulator/JointFeedback.h"

class OrderGenerator {
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
        
        void feedbackHandler(const single_joint_gz_simulator::JointFeedback::ConstPtr& msg);
        void publishOrder(single_joint_episode_generator::JointOrder *order);
        single_joint_episode_generator::JointOrder generate();
        single_joint_episode_generator::JointOrder next(
            single_joint_episode_generator::JointOrder *sent,
            single_joint_gz_simulator::JointFeedback *received
        )

    public:
        OrderGenerator(ros::NodeHandle *nh, single_joint_episode_generator::OfflineNodeWatcher *watcher);
        ~OrderGenerator();
        void run();

}