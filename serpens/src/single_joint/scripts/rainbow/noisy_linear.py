#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NoisyLinear has proven to be efficient in DQN-based
Reinforcement Learning algorithms.

https://arxiv.org/pdf/1706.10295.pdf
"""
import math

import tensorflow as tf

class NoisyLinear(tf.Module):
    """Noisy linear module for NoisyNet.
            
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (tf.Variable): mean value weight parameter
        weight_sigma (tf.Variable): std value weight parameter
        bias_mu (tf.Variable): mean value bias parameter
        bias_sigma (tf.Variable): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init


        # Choose a uniform initializer
        mu_range = 1 / math.sqrt(self.in_features)
        mu_initializer = tf.random_uniform_initializer(
            minval=-mu_range, maxval=mu_range, seed=None
        )
        constant_initializer = tf.constant_initializer(value=(self.std_init / math.sqrt(self.in_features)))

        # Initialize layer parameter
        self.weight_mu = tf.Variable(mu_initializer(shape=[out_features, in_features], dtype=tf.float32, name="weight_mu"))
        self.weight_sigma = tf.Variable(constant_initializer(shape=[out_features, in_features], dtype=tf.float32, name="weight_sigma"))
        self.bias_mu = tf.Variable(mu_initializer(shape=[out_features], dtype=tf.dtypes.float32, name="bias_mu"))
        self.bias_sigma = tf.Variable(constant_initializer(shape=[out_features], dtype=tf.float32, name="bias_sigma"))

        self.weight_epsilon = tf.Variable(
            tf.zeros([out_features, in_features], dtype=tf.dtypes.float32, name="weight_epsilon"),
            trainable=False
        )

        self.bias_epsilon = tf.Variable(
            tf.zeros([out_features], dtype=tf.dtypes.float32, name="bias_epsilon"),
            trainable=False
        )

        self.reset_noise()


    def reset_noise(self):
        """
        Make new noises
        """
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.assign(tf.tensordot(epsilon_out, epsilon_in, axes=0))
        self.bias_epsilon.assign(epsilon_out)


    @staticmethod
    def scale_noise(size: int) -> tf.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=size), dtype=tf.dtypes.float32)
        return tf.math.multiply(tf.math.sign(x), tf.math.sqrt(tf.math.abs(x)))
