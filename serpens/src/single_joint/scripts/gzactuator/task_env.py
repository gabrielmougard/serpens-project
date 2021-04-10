#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The communication layer with the gzactuator node where
is defined the Gym Environment of the single joint.
"""

from gym import utils
from joint_env import JointEnv
from gym.envs.registration import register
from gym import error, spaces
import rospy
import math
import numpy as np

register(
    id="SnakeJoint-v0",
    entry_point="single_joint.gzactuator.task_env:SnakeJoint",
    max_episode_steps=1000, # TODO: call rospy.get_param() instead
)

class SnakeJoint(JointEnv):
    def __init__(self):
        # TODO

    def get_params(self):
        # TODO

    def _set_action(self, action):
        # TODO

    def _get_obs(self):
        # TODO

    def _is_done(self, observation):
        # TODO

    def _compute_reward(self, observations, done):
        # TODO

    def _init_env_variables(self):
        # TODO

    def _set_init_pose(self):
        # TODO
