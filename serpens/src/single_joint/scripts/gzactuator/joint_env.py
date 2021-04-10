#!/usr/bin/env python

import gym
import rospy
import roslaunch
import time
import numpy as np
from gym import utils, spaces
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gym.utils import seeding
from gym.envs.registration import register
import copy
import math
import os

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkState
from rosgraph_msgs.msg import Clock

from core.robot_gazebo_env import RobotGazeboEnv


class JointEnv(RobotGazeboEnv):
    def __init__(self, control_type):
        self.publishers_array = []
        self._base_pub = rospy.Publisher('/snakejoint_v0/base_joint_velocity_controller/command', Float64, queue_size=1)
        self._pole_pub = rospy.Publisher('/snakejoint_v0/joint_velocity_controller/command', Float64, queue_size=1)
        self.publishers_array.append(self._base_pub)
        self.publishers_array.append(self._pole_pub)

        rospy.Subscriber("/snakejoint_v0/joint_states", JointState, self.joints_callback)

        self.control_type = control_type
        if self.control_type == "velocity":
            self.controllers_list = [
                'joint_state_controller',
                'pole_joint_velocity_controller',
                'foot_joint_velocity_controller',
            ]
                                    
        elif self.control_type == "position":
            self.controllers_list = [
                'joint_state_controller',
                'pole_joint_position_controller',
                'foot_joint_position_controller',
            ]
                                    
        elif self.control_type == "effort":
            self.controllers_list = [
                'joint_state_controller',
                'pole_joint_effort_controller',
                'foot_joint_effort_controller',
            ]

        self.robot_name_space = "snakejoint_v0"
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


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # RobotEnv methods
    def _env_setup(self, initial_qpos):
        self.init_internal_vars(self.init_pos)
        self.set_init_pose()
        self.check_all_systems_ready()


    def init_internal_var(self, init_pos_value):
        #TODO

    def check_publishers_connection(self):
        #TODO

    def _check_all_systems_ready(self, init=True):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        #TODO

    def move_joints(self, joint_array):
        #TODO

    def get_clock_time(self):
        #TODO

    
