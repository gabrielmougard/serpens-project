#!/usr/bin/env python

import copy
import math
import os
import time
import rospy

from gym.utils import seeding
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from rosgraph_msgs.msg import Clock

from core.robot_gazebo_env import RobotGazeboEnv


class JointEnv(RobotGazeboEnv):
    def __init__(self):
        self.robot_name_space = "single_joint"
        self._torque_pub = rospy.Publisher('/{}/link_motor_effort/command'.format(self.robot_name_space), Float64, queue_size=1)
        

        rospy.Subscriber("/{}/joint_states".format(self.robot_name_space), JointState, self.joints_callback)

        self.controllers_list = ['link_motor_effort']

        self.reset_controls = True

        # Seed the environment
        self._seed()
        self.steps_beyond_done = None

        super(JointEnv, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=self.reset_controls
        )


    def joints_callback(self, data):
        self.joints = data


    def _seed(self, seed=5048795115606990371):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def init_internal_vars(self, init_torque_value):
        self.torque = [init_torque_value]
        self.joints = None


    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while (self._torque_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to _torque_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_torque_pub Publisher Connected")
        rospy.logdebug("All Publishers READY")


    # def _check_all_systems_ready(self, init=True):
    #     """
    #     Checks that all the sensors, publishers and other simulation systems are
    #     operational.
    #     """
    #     try:
    #         self.base_position = rospy.wait_for_message("/{}/joint_states".format(self.robot_name_space), JointState, timeout=1.0)
    #         rospy.logdebug("Current /{}/joint_states READY=>".format(self.robot_name_space))
    #     except:
    #         rospy.logerr("Current /{}/joint_states not ready yet, retrying for getting joint_states".format(self.robot_name_space))
    #     rospy.logdebug("ALL SYSTEMS READY")


    def move_joints(self, joint_array):
        """
        Apply random external torque to existing torque order
        and publish it to gazebo.
        """
        joint_value = Float64()
        joint_value.data = joints_array[0] + self.episode_random_external_torque
        rospy.logdebug("Torque join value : "+str(joint_value))
        self._torque_pub.publish(joint_value)


    def get_clock_time(self):
        self.clock_time = None
        while self.clock_time is None and not rospy.is_shutdown():
            try:
                self.clock_time = rospy.wait_for_message("/clock", Clock, timeout=1.0)
                rospy.logdebug("Current clock_time READY=>" + str(self.clock_time))
            except:
                rospy.logdebug("Current clock_time not ready yet, retrying for getting Current clock_time")
        return self.clock_time
