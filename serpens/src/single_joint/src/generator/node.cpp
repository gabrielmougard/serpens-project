#include "ros/ros.h"

#include "single_joint/JointOrder.h"
#include "single_joint/order_generator.h"

using namespace single_joint;

int main(int *argc, char **argv) {
	ros::init(argc, argv, "generator");
	ros::NodeHandle nh;
	OrderGenerator *g = new OrderGenerator(nh);
	ROS_INFO("Starting the OrderGenerator ...");
	g.run();
	return 0;
}