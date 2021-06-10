#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The main function for the Reinforcement training node.
"""
import time

import rospy
import torch
from gym.utils import seeding

from ppo.agent import PPOAgent
from ppo.network import MlpPolicy
from gzactuator.joint_env import SnakeJoint 

if __name__ == "__main__":
    # Initialize the node and name it.
    rospy.init_node("rainbow")

    # memory_size = rospy.get_param('/rainbow/memory_size')
    # batch_size = rospy.get_param('/rainbow/batch_size')
    # target_update = rospy.get_param('/rainbow/target_update')
    # gamma = rospy.get_param('/rainbow/gamma')
    # alpha = rospy.get_param('/rainbow/alpha')
    # beta = rospy.get_param('/rainbow/beta')
    # prior_eps = rospy.get_param('/rainbow/prior_eps')
    # categorical_v_min = rospy.get_param('/rainbow/categorical_v_min')
    # categorical_v_max = rospy.get_param('/rainbow/categorical_v_max')
    # categorical_atom_size = rospy.get_param('/rainbow/categorical_atom_size')
    # n_step = rospy.get_param('/rainbow/n_step')
    # convergence_window = rospy.get_param('/rainbow/convergence_window')
    # convergence_window_epsilon_p = rospy.get_param('/rainbow/convergence_window_epsilon_p')
    # convergence_avg_score = rospy.get_param('/rainbow/convergence_avg_score')
    # convergence_avg_epsilon = rospy.get_param('/rainbow/convergence_avg_epsilon')
    # convergence_avg_epsilon_p = rospy.get_param('/rainbow/convergence_avg_epsilon_p')
    # model_name = rospy.get_param('/rainbow/model_name')
    # num_frames = rospy.get_param('/rainbow/num_frames')

    joint_env = SnakeJoint()

    gamma = 0.99
    plot_every = 10
    update_freq = 1
    k_epoch = 3
    learning_rate = 0.02
    lmbda = 0.95
    eps_clip = 0.2
    v_coef = 1
    entropy_coef = 0.01
    memory_size = 400
    checkpoint_interval = 50 # every 50 episodes

    agent = PPOAgent(
        joint_env,
        memory_size,
        gamma,
        plot_every,
        update_freq,
        k_epoch,
        learning_rate,
        lmbda,
        eps_clip,
        v_coef,
        entropy_coef,
        checkpoint_interval
    )

    train = False

    # Train loop
    if train:
        agent.train()
    else:
        # Inference
        inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MlpPolicy(action_size=joint_env.action_space.n).to(inference_device)
        model.load_state_dict(torch.load("saved_models"))
        model.eval()
        np_random, _ = seeding.np_random(5048795115606990371)
        # Decide of a random constant order to hold
        theta_ld_max = rospy.get_param("/rainbow/theta_ld_max")
        order = np_random.uniform(-theta_ld_max, theta_ld_max)
        state = None
        step = 0
        while True:
            state, _, _, step = agent.predict(model, order, step, state=state)
            rospy.loginfo(f"[INFERENCE] epsilon: {state[6]}, step: {step}")
