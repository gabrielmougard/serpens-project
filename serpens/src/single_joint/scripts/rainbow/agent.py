#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The definition of the RainbowAgent.
It is the central part of the RL loop.
"""
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from collections import deque
from statistics import mean
from tqdm import tqdm

import rospy
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from cpprb import ReplayBuffer, PrioritizedReplayBuffer

from rainbow.network import Network
from rainbow.tensorboard import RainbowTensorBoard
from rainbow.util import TqdmToLogger

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
tf.get_logger().setLevel(logging.ERROR) # Allow only tensorflow error logs to be shown
tqdm_out = TqdmToLogger(logger,level=logging.INFO)


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
        # Convergence parameters
        convergence_window: int = 100,
        convergence_window_epsilon_p: int = 10, 
        convergence_avg_score: float = 195.0,
        convergence_avg_epsilon: float = 0.0524, # 3 degs converted to rads
        convergence_avg_epsilon_p: float = 0.0174, # 1 deg/s converted to rad/s
        # Tensorboard parameters
        model_name: str = "snake_joint",
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
        physical_devices = tf.config.list_physical_devices('GPU') 
        n_gpu = len(physical_devices)
        rospy.loginfo("Number of GPU detected : " + str(n_gpu))
        if n_gpu > 0:
            rospy.loginfo("Switching to single GPU mode : /device:GPU:0")
            self.used_device = "/device:GPU:0"
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            rospy.loginfo("No GPU detected. Switching to single CPU mode : /device:CPU:0")
            self.used_device = "/device:CPU:0"

        # PER
        # memory for 1-step learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            memory_size,
            {
                "obs": {"shape": (obs_dim,)},
                "act": {"shape": (1,)},
                "rew": {},
                "next_obs": {"shape": (obs_dim,)},
                "done": {}
            },
            alpha=alpha    
        )

        # memory for N-step learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                memory_size,
                {
                    "obs": {"shape": (obs_dim,)},
                    "act": {"shape": (1,)},
                    "rew": {},
                    "next_obs": {"shape": (obs_dim,)},
                    "done": {}
                },
                Nstep={
                    "size": n_step,
                    "gamma": gamma,
                    "rew": "rew",
                    "next": "next_obs"
                }
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

        # optimizer
        self.optimizer = Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='AdamOptimizer'
        )

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        # Custom tensorboard object
        self.tensorboard = RainbowTensorBoard(
            log_dir="single_joint_logs/{}-{}".format(
                model_name,
                datetime.now().strftime("%m-%d-%Y-%H_%M_%S")
            )
        )
        # Convergence criterion
        self.convergence_window = convergence_window
        self.convergence_window_epsilon_p = convergence_window_epsilon_p
        self.convergence_avg_score = convergence_avg_score 
        self.convergence_avg_epsilon = convergence_avg_epsilon
        self.convergence_avg_epsilon_p = convergence_avg_epsilon_p

        # model checkpoint object
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.dqn_target)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory="single_joint_ckpts", max_to_keep=5
        )


    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = tf.math.argmax(self.dqn(
            tf.constant(state.reshape(1, state.shape[0]), dtype=tf.float32)
        ), axis=-1, name="argmax_selected_action")
        
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
                idx = self.memory_n.add(
                    **dict(
                        zip(["obs", "act", "rew", "next_obs", "done"], self.transition)
                    )
                )
                one_step_transition = [ v[idx] for _,v in self.memory_n.get_all_transitions().items()] if idx else None

            # 1-step transition
            else:
                one_step_transition = self.transition
            # add a single step transition
            if one_step_transition:
                self.memory.add(
                    **dict(
                        zip(["obs", "act", "rew", "next_obs", "done"], one_step_transition)
                    )
                )
        return next_state, reward, done


    def update_model(self) -> tf.Tensor:
        """
        Update the model by gradient descent
        """
        # PER needs beta to calculate weights
        samples = self.memory.sample(self.batch_size, beta=self.beta)
        weights = tf.constant(
            samples["weights"].reshape(-1, 1),
            dtype=tf.float32,
            name="update_model_weights"
        )
        indices = samples["indexes"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)


        with tf.GradientTape() as tape:
            # PER: importance of sampling before average
            loss = tf.math.reduce_mean(elementwise_loss * weights)

            # N-step Learning loss
            # We are going to combine 1-ste[ loss and n-step loss so as to
            # prevent high-variance.
            if self.use_n_step:
                gamma = self.gamma ** self.n_step
                samples = {k: [v[i] for i in indices] for k,v in self.memory_n.get_all_transitions().items()}
                elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
                elementwise_loss += elementwise_loss_n_loss

                # PER: importance of sampling before average
                loss = tf.math.reduce_mean(elementwise_loss * weights)
        
        dqn_variables = self.dqn.trainable_variables
        gradients = tape.gradient(loss, dqn_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.optimizer.apply_gradients(zip(gradients, dqn_variables))

        # PER: update priorities
        loss_for_prior = elementwise_loss.numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.numpy().ravel()


    def train(self, num_frames: int):
        """Train the agent."""
        self.is_test = False

        state = self.env.reset()
        update_cnt = 0
        scores = deque(maxlen=self.convergence_window)
        joint_epsilon = deque(maxlen=self.convergence_window)
        joint_epsilon_p = deque(maxlen=self.convergence_window_epsilon_p)
        score = 0 # cumulated reward
        episode_length = 0
        episode_cnt = 0

        for frame_idx in tqdm(range(1, num_frames + 1), file=tqdm_out):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward
            episode_length += 1

            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            print("epsilon_p is {}".format(state[7]))
            print("epsilon is {}".format(state[6]))

            if done:
                print("done")
                # to be used for convergence criterion
                scores.append(score) 
                joint_epsilon.append(state[6])
                joint_epsilon_p.append(state[7])
                #

                state = self.env.reset()
                self.tensorboard.update_stats(
                    score={
                        "data": score,
                        "desc": "Score (or cumulated rewards) for an episode - episode index on x-axis."
                    },
                    episode_length={
                        "data": episode_length,
                        "desc": "Episode length (in frames)"
                    },
                    final_epsilon={
                        "data": state[6],
                        "desc": "Value of epsilon = abs(theta_ld - theta_l) at the last frame of an episode"
                    },
                    final_epsilon_p={
                        "data": state[7],
                        "desc": "Value of d(epsilon)/dt at the last frame of an episode"
                    }
                )
                score = 0
                episode_length = 0
                episode_cnt += 1

                # check convergence criterion
                converged = bool(
                    len(scores) == self.convergence_window and # be sure the score buffer is full
                    len(joint_epsilon) == self.convergence_window and # same for epsilon buffer
                    len(joint_epsilon_p) == self.convergence_window and # same for epsilon_p buffer
                    mean(scores) > self.convergence_avg_score and 
                    mean(joint_epsilon) < self.convergence_avg_epsilon and
                    mean(joint_epsilon_p) < self.convergence_avg_epsilon_p
                )
                if converged:
                    rospy.loginfo("Ran {} episodes. Solved after {} trials".format(episode_cnt, frame_idx))
                    return

            #  if training is ready
            if self.memory.get_stored_size() >= self.batch_size:
                loss = self.update_model()
                # plotting loss every frame
                self.tensorboard.update_stats(
                    loss={
                        "data": loss[0],
                        "desc": "Loss value."
                    }
                )
                update_cnt += 1
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()
                    # checkpointing of target model (only if the loss decrease)
                    self.checkpoint_manager.save()


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
        
        rospy.loginfo("score: ", score)
        self.env.close()
        
        return frames


    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> tf.Tensor:
        with tf.device(self.used_device):
            state = tf.constant(samples["obs"], dtype=tf.float32)
            next_state = tf.constant(samples["next_obs"], dtype=tf.float32)
            action = tf.constant(samples["act"], dtype=tf.float32)
            reward = tf.reshape(tf.constant(samples["rew"], dtype=tf.float32), [-1, 1])
            done = tf.reshape(tf.constant(samples["done"], dtype=tf.float32), [-1, 1])

            # Categorical DQN algorithm
            delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

            # Double DQN
            next_action = tf.math.argmax(self.dqn(next_state), axis=1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = tf.gather_nd(
                next_dist,
                [[i, next_action.numpy()[0]] for i in range(self.batch_size)]
            )

            t_z = reward + (1 - done) * gamma * self.support
            t_z = tf.clip_by_value(t_z, clip_value_min=self.v_min, clip_value_max=self.v_max)
            b = tf.dtypes.cast((t_z - self.v_min) / delta_z, tf.float64)
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

            proj_dist = tf.zeros(tf.shape(next_dist), tf.float64)
            # casting
            next_dist = tf.dtypes.cast(next_dist, tf.float64)

            proj_dist = tf.tensor_scatter_nd_add(
                tf.reshape(proj_dist, [-1]), # input tensor
                tf.reshape(tf.dtypes.cast(l + offset, tf.int64), [-1, 1]), # indices
                tf.reshape((next_dist * (u - b)), [-1]) # updates
            )

            proj_dist = tf.tensor_scatter_nd_add(
                proj_dist,
                tf.reshape(tf.dtypes.cast(u + offset, tf.int64), [-1, 1]), # indices
                tf.reshape((next_dist * (b - l)), [-1]) # updates
            )
            proj_dist = tf.reshape(proj_dist, [self.batch_size, self.atom_size])

        dist = self.dqn.dist(state)
        #log_p = tf.math.log(dist[range(self.batch_size), action])
        log_p = tf.dtypes.cast(
            tf.math.log(
                tf.gather_nd(
                    dist,
                    [[i, tf.dtypes.cast(tf.reshape(action, [-1]), tf.int32).numpy()[i]] for i in range(self.batch_size)]
                )
            ),
            tf.float64
        )
        elementwise_loss = tf.math.reduce_sum(-(proj_dist * log_p), axis=1)

        return tf.dtypes.cast(elementwise_loss, tf.float32)


    def _target_hard_update(self):
        """Hard update: target <- local."""
        tf.saved_model.save(self.dqn, "single_joint_dqn")
        self.dqn_target = tf.saved_model.load("single_joint_dqn")
