#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:59:11 2020

@author: yaoleixu
"""

# ============================================================================
# Import libraries
import numpy as np
import tensorflow as tf

tf.random.set_seed(0)
    

# ============================================================================
# General recurrent tensor network class
class GeneralRGTN (tf.keras.layers.Layer):
    
    # Initialize terms
    def __init__(
        self, 
        units, 
        damping=0.9,
        bi_directional=False,
        **kwargs,
    ):

        # Save variables
        self.units = units
        self.damping = damping
        self.bi_directional = bi_directional

        # Additional variables
        self.time_steps = None
        self.layer_norm = tf.keras.layers.LayerNormalization()

        super(GeneralRGTN, self).__init__(**kwargs)
        
    def get_time_adj(self):
        
        # Adjacency matrix as a constant variable (N x N)
        A = np.zeros((self.time_steps, self.time_steps))
        for row_idx in range(A.shape[0]):
            for col_idx in range(A.shape[1]):
                if col_idx > row_idx:
                    A[row_idx, col_idx] = self.damping**(col_idx-row_idx)

        # Bi-directional graph
        if self.bi_directional:
            A = A / 2 + A.T / 2
            
        return tf.Variable(tf.cast(A.T, tf.float32), trainable=False)

    # Define weights
    def build(self, input_shape):
        
        # Get time variable
        self.time_steps = input_shape[1]
        
        # Create a trainable weight variable for this layer of shape (f_i x f_o)
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            trainable=True,
        )
        
        # Recurrent kernel weights
        self.p_kernel = self.add_weight(
            name='p_kernel',
            shape=(self.units, self.units),
            trainable=True,
        )

        # Time adjacency matrix
        self.time_adj = self.get_time_adj()
                
        # Be sure to call this at the end
        super(GeneralRGTN, self).build(input_shape)

    # Forward pass
    def call(self, x):
        
        # Contractions
        x = tf.tensordot(x, self.kernel, [[-1], [0]])  # feature map
        x_filt = tf.tensordot(x, self.p_kernel, [[-1], [0]])
        x_filt = tf.tensordot(self.time_adj, x_filt, [[-1], [1]])
        x_filt = tf.transpose(x_filt, [1, 0] + [d for d in range(2, len(x.shape))])
        y = self.layer_norm(x + x_filt)
            
        return y


# ============================================================================
# Special recurrent tensor network class
class SpecialRGTN (tf.keras.layers.Layer):
    
    # Initialize terms
    def __init__(
        self, 
        units, 
        damping=0.9, 
        bi_directional=False,
        **kwargs
    ):
        
        # Save variables
        self.units = units  # number of hidden units
        self.damping = damping  # time-steps damping
        self.bi_directional = bi_directional  # bi directional time-graph
        
        # Additional variables
        self.time_steps = None
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
        super(SpecialRGTN, self).__init__(**kwargs)
        
    def get_time_adj(self):
        
        # Adjacency matrix as a constant variable (N x N)
        A = np.zeros((self.time_steps, self.time_steps))
        for row_idx in range(A.shape[0]):
            for col_idx in range(A.shape[1]):
                if col_idx > row_idx:
                    A[row_idx, col_idx] = self.damping**(col_idx-row_idx)

        # Bi-directional graph
        if self.bi_directional:
            A = A / 2 + A.T / 2
            
        return tf.Variable(tf.cast(A.T, tf.float32), trainable=False)

    def build(self, input_shape):
        
        # Get number of time-steps per sample
        self.time_steps = input_shape[1]
        
        # Create a trainable weight variable for this layer of shape (f_i x f_o)
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            trainable=True
        )
        
        # Time adjacency matrix
        self.time_adj = self.get_time_adj()
                
        # Be sure to call this at the end
        super(SpecialRGTN, self).build(input_shape)

    def call(self, x):

        # Contractions
        x = tf.tensordot(x, self.kernel, axes=[[-1],[0]]) # feature map
        x_filt = tf.tensordot(self.time_adj, x, axes=[[-1], [1]]) # graph filter
        x_filt = tf.transpose(x_filt, [1, 0] + [d for d in range(2, len(x.shape))])
        y = self.layer_norm(x + x_filt)
                    
        return y
    
    
# ============================================================================
# Tensor-Train Fully-Connected Layer from Tensorizing Neural Networks
class TensorTrainLayer (tf.keras.layers.Layer):
    
    # define initial variables needed for implementation
    def __init__(self, tt_ips, tt_ops, tt_ranks, bias_bool=False, **kwargs):

        # Tensor Train Variables.
        self.tt_ips = np.array(tt_ips)
        self.tt_ops = np.array(tt_ops)
        self.tt_ranks = np.array(tt_ranks)
        self.num_dim = np.array(tt_ips).shape[0]
        self.param_n = np.sum(self.tt_ips*self.tt_ops*self.tt_ranks[1:]*self.tt_ranks[:-1])
        self.bias_bool = bias_bool
  
        super(TensorTrainLayer, self).__init__(**kwargs)

    # define weights for each core
    def build(self, input_shape):

        # Initalize weights for the TT FCL. Note that Keras will pass the optimizer directly on these core parameters
        self.cores = []
        for d in range(self.num_dim):
            if d == 0: my_shape = (self.tt_ips[d], self.tt_ops[d], self.tt_ranks[d+1])
            elif d == self.num_dim-1: my_shape = (self.tt_ranks[d], self.tt_ips[d], self.tt_ops[d])
            else: my_shape = (self.tt_ranks[d], self.tt_ips[d], self.tt_ops[d], self.tt_ranks[d+1])
            
            self.cores.append(self.add_weight(name='tt_core_{}'.format(d),
                                              shape=my_shape,
                                              initializer='uniform',
                                              trainable=True))
        
        # Bias vector
        if self.bias_bool:
            self.bias = self.add_weight(name='bias',
                                        shape=self.tt_ops,
                                        initializer='uniform',
                                        trainable=True)
        # Be sure to call this at the end
        super(TensorTrainLayer, self).build(input_shape)

    # Implementing the layer logic
    def call(self, x, mask=None):

        w = self.cores[0]
        for d in range(1, self.num_dim):
            w = tf.tensordot(w, self.cores[d], [[-1],[0]])

        output = tf.tensordot(x, w, [[i for i in range(1, 3+1)], [i for i in range(0, 2*3, 2)]])
        
        if self.bias_bool: output = output + self.bias

        return output

    # Compute input/output shapes
    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(self.tt_ops))
