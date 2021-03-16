#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The definition of the RainbowAgent.
It is the central part of the RL loop.
"""
import rospy
import gym
import numpy as np
import tensorflow as tf
from tf.keras.optimizer import Adam

from prioritized_replay_buffer import PrioritizedReplayBuffer
 
class RainbowAgent:
    """
    Rainbow Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment (connected to Gazebo node)
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
            state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """
    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
    ):
        """
        Initialization.

        Args:
            env_client (GymEnvClient): ROS client to an openAI Gym environment server
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma

        # Selecting computing device
        n_gpu = len(tf.config.list_physical_devices('GPU'))
        rospy.loginfo("Number of GPU detected : " + str(n_gpu))
        if n_gpu > 0:
            rospy.loginfo("Switching to single GPU mode : /device:GPU:0")
            self.used_device = "/device:GPU:0"
        else:
            rospy.loginfo("No GPU detected. Switching to single CPU mode : /device:CPU:0")
            self.used_device = "/device:CPU:0"

        # PER
        # memory for 1-step learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha
        )

        # memory for N-step learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = tf.linspace(self.v_min, self.v_max, self.atom_size, name="support")

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, self.atom_size, self.support, name="dqn"
        )
        self.dqn_target = Network(
            obs_dim, action_dim, self.atom_size, self.support, name="dqn_target"
        )

        tf.saved_model.save(self.dqn, "./dqn")
        self.dqn_target = tf.saved_model.load("dqn")

        # optimizer
        self.optimizer = Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='AdamOptimizer'
        )

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False


    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = tf.math.argmax(self.dqn(
            tf.constant(state, dtype=tf.float32)
        ), name="argmax_selected_action")

        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        # TODO

    def update_model(self) -> tf.Tensor:
        # TODO
    
    def train(self, num_frames: int, plotting_interval: int = 200):
        # TODO

    def test(self) -> List[np.ndarray]:
        # TODO

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> tf.Tensor:
        # TODO

    def _target_hard_update(self):


    def _plot(self, frame_idx: int, scores: List[float], losses: List[float]):

