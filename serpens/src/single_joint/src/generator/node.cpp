#include "ros/ros.h"

#include "single_joint/JointOrder.h"
#include "single_joint/OrderGenerator.h"
#include "single_joint/OfflineNodeWatcher.h"

using namespace single_joint;

int main(int *argc, char **argv) {
	ros::init(argc, argv, "generator");
	ros::NodeHandle nh;
	//OfflineNodeWatcher *watcher = new OfflineNodeWatcher();
	// Generator *g = new Generator(nh, watcher);
	// ROS_INFO("Starting the OrderGenerator ...");
	// g.run();
	// ROS_INFO("Closing the OrderGenerator ...");
	// return 0;
}