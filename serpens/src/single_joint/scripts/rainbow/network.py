#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The topology of the Deep-Q-Network that
we use for our RainbowAgent.


We also make use of a NoisyLayer, which has
proven to be efficient in DQN-based
Reinforcement Learning algorithms.

https://arxiv.org/pdf/1706.10295.pdf
"""

import tensorflow as tf

from tensorflow_addons.layers import NoisyDense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.activations import relu, softmax


class Network(tf.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        atom_size: int,
        support: tf.Tensor,
        **kwargs
    ):
        """
        Initialization
        """
        super(Network, self).__init__(**kwargs)

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = Sequential()
        self.feature_layer.add(Input(shape=(in_dim,)))
        self.feature_layer.add(Dense(128, activation='relu'))
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyDense(128, name="advantage_hidden_layer")
        self.advantage_layer = NoisyDense(out_dim * atom_size, name="advantage_layer")

        # set value layer
        self.value_hidden_layer = NoisyDense(128, name="value_hidden_layer")
        self.value_layer = NoisyDense(atom_size, name="value_layer")


    @tf.function
    def __call__(self, x):
        dist = self.dist(x)
        q = tf.reduce_sum(dist * self.support, axis=2)
        return q


    @tf.function
    def dist(self, x):
        """
        Get distribution for atoms
        """
        feature = self.feature_layer(x)
        adv_hid = relu(self.advantage_hidden_layer(feature))
        val_hid = relu(self.value_hidden_layer(feature))
        advantage = tf.reshape(self.advantage_layer(adv_hid), [-1, self.out_dim, self.atom_size])
        value = tf.reshape(self.value_layer(val_hid), [-1, 1, self.atom_size])
        q_atoms = value + advantage - tf.math.reduce_mean(advantage, axis=1, keepdims=True)
        dist = softmax(q_atoms, axis=-1)
        return tf.clip_by_value(dist, clip_value_min=1e-3, clip_value_max=float('inf')) # for avoiding NaNs


    @tf.function
    def reset_noise(self):
        """
        Reset all noisy layers.
        """
        self.advantage_hidden_layer._reset_noise()
        self.advantage_layer._reset_noise()
        self.value_hidden_layer._reset_noise()
        self.value_layer._reset_noise()
