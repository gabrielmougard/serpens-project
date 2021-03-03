#include <ros/ros.h>

class OrderGenerator {
    private:
        ros::NodeHandle *nh;
        ros::Publisher *order_pub;
        ros::Subscriber *feedback_sub;
        
    public:
        OrderGenerator(ros::NodeHandle *nh, *single_joint_episode_generator::OfflineNodeWatcher *watcher);
        ~OrderGenerator();
        void run();
        void publishOrder(single_joint_episode_generator::JointOrder *order);
        void feedbackHandler(const single_joint_gz_simulator::JointFeedback::ConstPtr& msg);
        single_joint_episode_generator::JointOrder generate();

}