# gzactuator node

The purpose of this node is to simulate the single joint in Gazebo so that it can feedback the results to the `generator` node.
This node can also send information (as ROS Services) to the `rainbow` node.
This node contains the URDF definition of the joint as well as an OpenAI Gym environment (which is wrapped around a ROS Service server layer to communicate with the `rainbow` node in order to tell whether or not an episode has converged and thus give the order to reset the episode) 