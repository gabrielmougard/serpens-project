#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The main function for the Reinforcement training node.
"""

import rospy

from rainbow import RainbowAgent
from gzactuator.task_env import SnakeJoint 

if __name__ == "__main__":
    # Initialize the node and name it.
    rospy.init_node("rainbow")

    n_episodes = rospy.get_param('/rainbow/episodes_training')
    memory_size = rospy.get_param('/rainbow/memory_size')
    batch_size = rospy.get_param('/rainbow/batch_size')
    target_update = rospy.get_param('/rainbow/target_update')
    gamma = rospy.get_param('/rainbow/gamma')
    alpha = rospy.get_param('/rainbow/alpha')
    beta = rospy.get_param('/rainbow/beta')
    prior_eps = rospy.get_param('/rainbow/prior_eps')
    categorical_v_min = rospy.get_param('/rainbow/categrorical_v_min')
    categorical_v_max = rospy.get_param('/rainbow/categrorical_v_max')
    categorical_atom_size = rospy.get_param('/rainbow/categrorical_atom_size')
    n_step = rospy.get_param('/rainbow/n_step')
    convergence_window = rospy.get_param('/rainbow/convergence_window')
    convergence_window_epsilon_p = rospy.get_param('/rainbow/convergence_window_epsilon_p')
    convergence_avg_score = rospy.get_param('/rainbow/convergence_avg_score')
    convergence_avg_epsilon = rospy.get_param('/rainbow/convergence_avg_epsilon')
    convergence_avg_epsilon_p = rospy.get_param('/rainbow/convergence_avg_epsilon_p')
    model_name = rospy.get_param('/rainbow/model_name')
    num_frames = rospy.get_param('/rainbow/num_frames')

    joint_env = SnakeJoint()

    agent = RainbowAgent(
        joint_env,
        memory_size,
        batch_size,
        target_update,
        gamma,
        alpha,
        beta,
        prior_eps,
        categorical_v_min,
        categorical_v_max,
        categorical_atom_size,
        n_step,
        convergence_window,
        convergence_window_epsilon_p, 
        convergence_avg_score,
        convergence_avg_epsilon,
        convergence_avg_epsilon_p,
        model_name
    )

    agent.train(num_frames)
