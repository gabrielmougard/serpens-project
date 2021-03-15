#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The topology of the Deep-Q-Network that
we use for our RainbowAgent.
"""

import tensorflow as tf

from tf.keras import Sequential
from tf.keras.layers import Input, Dense
from tf.keras.activations import relu, softmax

from noisy_layer import NoisyLayer


class Network(tf.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        atom_size: int,
        support: tf.Tensor
    ):
        """
        Initialization
        """
        super(Network, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = Sequential()
            .add(Input(shape=(in_dim,)))
            .add(Dense(128, activation='relu'))
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def __call__(self, x):
        dist = self.dist(x)
        q = tf.reduce_sum(dist * self.support, axis=2)
        return q


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


    def reset_noise(self):
        """
        Reset all noisy layers.
        """
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()
