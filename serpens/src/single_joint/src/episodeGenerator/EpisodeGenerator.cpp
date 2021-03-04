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
	OfflineNodeWatcher *watcher = new OfflineNodeWatcher();
	Generator *g = new Generator(nh, watcher);
	ROS_INFO("Starting the OrderGenerator ...");
	g.run();
	ROS_INFO("Closing the OrderGenerator ...");
	return 0;
}
