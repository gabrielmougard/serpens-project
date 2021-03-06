#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The main function for the Reinforcement training node.
"""

import rospy

from single_joint.srv import PositionOrder

def callback(req):
    """Handle subscriber data."""
    rospy.loginfo(
        (
            "Packet received:\n" +
            "\tstamp : %i\n" +
            "\tmsg_id : %d\n" +
            "\ttheta_ld : %f rad\n" +
            "\ttheta_l : %f rad\n" +
            "\ttheta_l_p : %f rad/s\n" +
            "\ttheta_m : %f rad\n" +
            "\ttheta_m_p : %f rad/s\n" +
            "\tepsilon : %f\n" +
            "\tepsilon_p : %f\n"
        ),
        req.stamp.nsecs,
        req.msg_id,
        req.theta_ld,
        req.theta_l,
        req.theta_l_p,
        req.theta_m,
        req.theta_m_p,
        req.epsilon,
        req.epsilon_p
    )
    # TODO : call the RL agent here.


if __name__ == "__main__":
    # Initialize the node and name it.
    rospy.init_node("rainbow")
    # Configure Service server 
    s = rospy.Service('position_order', PositionOrder, callback)
    # Start the node.
    rospy.spin()