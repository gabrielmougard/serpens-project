#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The main function for the Generator node
"""

import rospy

from single_joint.srv import PositionOrder
from generator.order_generator import OrderGenerator

if __name__ == "__main__":
    # Initialize the node and name it.
    rospy.init_node("generator")
    # Start the OrderGenerator 
    OrderGenerator()
    rospy.spin()