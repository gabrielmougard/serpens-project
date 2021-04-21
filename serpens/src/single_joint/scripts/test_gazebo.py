#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

class Test:
    def __init__(self):
        pass

        
    def talk_and_listen(self):
        pub = rospy.Publisher('/single_joint/link_motor_effort/command', Float64, queue_size=10)
        rospy.Subscriber('/single_joint/joint_states', JointState, self.joint_states_callback)

        rospy.init_node('test_gazebo')
        rate = rospy.Rate(10) # 10Hz
        while not rospy.is_shutdown():
            torque_value = random.uniform(-2.5, 2.5) # between 0.0 N.m and 2.5 N.m
            rospy.loginfo("The torque value is : {}".format(torque_value))
            pub.publish(torque_value)
            rate.sleep()

    def joint_states_callback(self, data):
        rospy.loginfo("joint states : {}".format(data))
        
if __name__ == '__main__':
    try:
        t = Test()
        t.talk_and_listen()
    except rospy.ROSInterruptException:
        pass
