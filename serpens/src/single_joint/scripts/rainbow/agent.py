#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The definition of the RainbowAgent.
It is the central part of the RL loop.
"""
from typing import Dict, List, Tuple

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
        
        # Convert to numpy ndarray datatype
        selected_action = selected_action.numpy()

        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """
        Take an action and return the response of the env.
        """
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]

            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition
            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
        return next_state, reward, done


    def update_model(self) -> tf.Tensor:
        """
        Update the model by gradient descent
        """
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = tf.constant(
            samples["weights"].reshape(-1, 1),
            dtype=tf.float32,
            name="update_model_weights"
        )
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)


        with tf.GradientTape() as tape:
            # PER: importance of sampling before average
            loss = tf.math.reduced_mean(elementwise_loss * weights)

            # N-step Learning loss
            # We are going to combine 1-ste[ loss and n-step loss so as to
            # prevent high-variance.
            if self.use_n_step:
                gamma = self.gamma ** self.n_step
                samples = self.memory_n.sample_batch_from_idxs(indices)
                elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
                elementwise_loss += elementwise_loss_n_loss

                # PER: importance of sampling before average
                loss = tf.math.reduced_mean(elementwise_loss * weights)
        
        dqn_variables = self.dqn.trainable_variables
        gradients = tape.gradient(loss, dqn_variables)
        gradients, _ = tf,clip_by_global_norm(gradients, 10.0)
        self.optimizer.apply_gradients(zip(gradients, dqn_variables))

        # PER: update priorities
        loss_for_prior = elementwise_loss.numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.numpy().ravel()


    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        update_cnt = 0
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
            
            # NoisyNet: removed decrease of epsilon
            
            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses)
                
        self.env.close()


    def test(self) -> List[np.ndarray]:
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        return frames


    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> tf.Tensor:
        with tf.device(self.used_device):
            state = tf.constant(samples["obs"], dtype=tf.float32)
            next_state = tf.constant(samples["next_obs"], dtype=tf.float32)
            action = tf.constant(samples["acts"], dtype=tf.float32)
            reward = tf.reshape(tf.constant(samples["rews"], dtype=tf.float32), [-1, 1])
            done = tf.reshape(tf.constant(samples["done"], dtype=tf.float32), [-1, 1])

            # Categorical DQN algorithm
            delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

            # Double DQN
            next_action = tf.math.argmax(self.dqn(next_state), axis=1)
            next_dist = tf.norm(self.dqn_target(next_state), ord=2)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = tf.clip_by_value(t_z, clip_value_min=self.v_min, clip_value_max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = tf.dtypes.cast(tf.math.floor(b), tf.float64)
            u = tf.dtypes.cast(tf.math.ceil(b), tf.float64)

            offset = (
                tf.broadcast_to(
                    tf.expand_dims(
                        tf.dtypes.cast(
                            tf.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size),
                            tf.float64
                        ),
                        axis=1
                    ),
                    [self.batch_size, self.atom_size]
                )
            )

            proj_dist = tf.zeros(tf.shape(next_dist), tf.float32)

            proj_dist = tf.tensor_scatter_nd_add(
                proj_dist, # input tensor
                tf.dtypes.cast(l + offset, tf.int64), # indices
                (next_dist * (u - b)) # updates
            )

            proj_dist = tf.tensor_scatter_nd_add(
                proj_dist,
                tf.dtypes.cast(u + offset, tf.int64), # indices
                (next_dist * (b - l)) # updates
            )

        dist = self.dqn.dist(state)
        log_p = tf.math.log(dist[range(self.batch_size), action])
        elementwise_loss = tf.math.reduce_sum(-(proj_dist * log_p), axis=1)

        return elementwise_loss


    def _target_hard_update(self):
        """Hard update: target <- local."""
        tf.saved_model.save(self.dqn, "./dqn")
        self.dqn_target = tf.saved_model.load("dqn")


    def _plot(self, frame_idx: int, scores: List[float], losses: List[float]):
        # TODO : Maybe call our Custom Tensorboard class here...
