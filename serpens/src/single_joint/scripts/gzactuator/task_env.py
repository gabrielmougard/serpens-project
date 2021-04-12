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
    """
    Observation:
        Type: Box(8)
        Num     Observation               Min                               Max
        0       theta_ld                  -theta_ld_max                     theta_ld_max
        1       theta_l                   -theta_l_max                      theta_l_max
        2       theta_l_p                 -Inf                              Inf
        3       theta_m                   -theta_m_max                      theta_m_max
        4       theta_m_p                 -theta_m_p_max                    theta_m_p_max
        5       tau_ext                   -tau_ext_max                      tau_ext_max
        6       epsilon                   -(theta_ld_max + theta_l_max)     theta_ld_max + theta_l_max
        7       epsilon_p                 -Inf                              Inf

    Actions:
        Type: Discrete(8)
        Num   Action
        0     decrease torque with very large step
        1     decrease torque with large step
        2     decrease torque with medium step
        3     decrease torque with small step
        4     increase torque with small step
        5     increase torque with medium step
        6     increase torque with large step
        7     increase torque with very large step

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting States:
        All observations are assigned a uniform random value in [-max_bound..max_bound]
        except for theta_l_p and epsilon_p which are set to +Inf.

    Episode termination:
        (Divergence)
            (epsilon > 50 deg && mean_10_step(abs(epsilon_p)) < 1 deg/s ) 
            OR
            episode_length > 200
        (Convergence)
            (
                avg_score > 195.0 &&
                mean_10_step(epsilon) < 3 deg &&
                mean_10_step(abs(epsilon_p)) < 1 deg/s
            ) over 100 consecutive frames 
    """
    def __init__(self):
        
        self.get_params()

        self.action_space = spaces.Discrete(self.n_actions)

        boundaries = np.array([
            self.theta_ld_max,
            self.theta_l_max,
            np.finfo(np.float32).max,
            self.theta_m_max,
            self.theta_m_p_max,
            self.tau_ext_max,
            self.theta_ld_max + self.theta_l_max,
            np.finfo(np.float32).max
        ])

        self.observation_space = spaces.Box(
            -boundaries,
            boundaries,
            dtype=np.float32
        )

        JointEnv.__init__(
            self, control_type=self.control_type
        )


    def get_params(self):
        # Get configuration parameters
        self.n_actions = rospy.get_param('/generator/n_actions')
        self.theta_ld_max = rospy.get_param("/generator/theta_ld_max")
        self.theta_ld_resolution = rospy.get_param("/generator/theta_ld_resolution")
        self.theta_l_max = rospy.get_param("/generator/theta_l_max")
        self.theta_l_resolution = rospy.get_param("/generator/theta_l_resolution")
        self.theta_m_max = rospy.get_param("/generator/theta_m_max")
        self.theta_m_resolution = rospy.get_param("/generator/theta_m_resolution")
        self.theta_m_p_max = rospy.get_param("/generator/theta_m_p_max")
        self.theta_m_p_resolution = rospy.get_param("/generator/theta_m_p_resolution")
        self.control_type = rospy.get_param('/generator/control_type')
        self.torque_step = rospy.get_param('/generator/torque_step')

        # Variables divergence/convergence conditions
        self.max_allowed_epsilon =  rospy.get_param('/generator/max_allowed_epsilon')
        self.max_ep_length =  rospy.get_param('/generator/max_ep_length')
        self.min_allowed_epsilon_p =  rospy.get_param('/generator/min_allowed_epsilon_p')


    def _set_action(self, action):
        
        # Take action
        if action == 0: # decrease torque with very large step
            self.torque[0] -= self.torque_step * 50
        elif action == 1: # decrease torque with large step
            self.torque[0] -= self.torque_step * 10
        elif action == 2: # decrease torque with medium step
            self.torque[0] -= self.torque_step * 5
        elif action == 3: # decrease torque with small step
            self.torque[0] -= self.torque_step
        elif action == 4: # increase torque with small step
            self.torque[0] += self.torque_step
        elif action == 5: # increase torque with medium step
            self.torque[0] += self.torque_step * 5
        elif action == 6: # increase torque with large step
            self.torque[0] += self.torque_step * 10
        elif action == 7: # increase torque with very large step
            self.torque[0] += self.torque_step * 50


        self.move_joint(self.torque)
        rospy.sleep(self.running_step) # Wait for some time
    
    
    def _get_obs(self):
        data = self.joint
        obs = [
            # TODO
        ]
        return np.array(obs)


    def _is_done(self, observation):
        data = self.joint

        done = bool(
            observation[6] > self.max_allowed_epsilon or
            abs(observation[7]) < self.min_allowed_epsilon_p
        )
        return done
        

    def _compute_reward(self, observations, done):
        """
        Gives more points for staying upright, gets data from given observations to avoid
        having different data than other previous functions
        :return:reward
        """
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Joint just diverged
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0
        return reward


    def _init_env_variables(self):
         """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.steps_beyond_done = None


    def _set_init_pose(self):
        """
        Sets joints to random initial position
        :return:
        """

        self.check_publishers_connection()

        # Reset Internal pos variable
        self.init_internal_vars(self.init_pos)
        self.move_joints(self.pos)
