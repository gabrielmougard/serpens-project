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

        self.weight_mu = tf.Variable(
            tf.zeros([out_features, in_features], dtype=tf.dtypes.float32, name="weight_mu")
        )
        self.weight_sigma = tf.Variable(
            tf.zeros([out_features, in_features], dtype=tf.dtypes.float32, name="weight_sigma")
        )
        self.weight_epsilon = tf.Variable(
            trainable=False,
            tf.zeros([out_features, in_features], dtype=tf.dtypes.float32, name="weight_epsilon")
        )
        self.bias_mu = tf.Variable(
            tf.zeros([out_features], dtype=tf.dtypes.float32, name="bias_mu")
        )
        self.bias_sigma = tf.Variable(
            tf.zeros([out_features], dtype=tf.dtypes.float32, name="bias_sigma")
        )
        self.bias_epsilon = tf.Variable(
            trainable=False,
            tf.zeros([out_features], dtype=tf.dtypes.float32, name="bias_epsilon")
        )

        self.reset_parameters()
        self.reset_noise()


    # def reset_parameters(self):
    #     """Reset trainable network parameters (factorized gaussian noise)."""
    #     mu_range = 1 / math.sqrt(self.in_features)
