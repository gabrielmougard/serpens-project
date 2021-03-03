// ROS Default Header File
#include "ros/ros.h"
// JointOrder Message File Header
// The header file is automatically created when building the package.
#include "single_joint_episode_generator/JointOrder.h"
#include "single_joint_episode_generator/OrderGenerator.h"
#include "single_joint_episode_generator/OfflineNodeWatcher.h"

using namespace single_joint_episode_generator;

int main(int *argc, char **argv) {
	ros::init(argc, argv, "single_joint_episode_generator"); // Initializes Node Name
	ros::NodeHandle nh; // Node handle declaration for communication with ROS system
	// Declare publisher, create publisher 'single_joint_episode_generator' using the 'JointOrder' 
	// message file from the 'ros_tutorials_topic' package. The topic name is
	// '/single_joint/input_to_rl/joint_order' and the size of the publisher queue is set to 100. 
	ros::Publisher episode_order_pub = nh.advertise<JointOrder>("/single_joint/input_to_rl/joint_order", 100);
	OfflineNodeWatcher *watcher = new OfflineNodeWatcher();
	Generator *g = new Generator(nh, watcher);
	ROS_INFO("Starting the OrderGenerator ...");
	g.run();
	ROS_INFO("Closing the OrderGenerator ...");
	return 0;
}
