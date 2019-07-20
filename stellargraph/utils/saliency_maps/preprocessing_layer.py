# -*- coding: utf-8 -*-
#
# Copyright 2018-2019 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocessing as a layer in GCN. This is to ensure that the GCN model is differentiable in an end-to-end manner.
"""


import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer
import keras


class GraphPreProcessingLayer(Layer):
    """
    This class implements the pre-processing of adjacency matrices in GCN. We implement it in tensorflow so that
    while computing the saliency maps, we are able to calculate the gradients in an end-to-end way. 
    We currently only support this for tensorflow backend.

    Args:
    output_dim (int pair): The output shape of the pre-processed adjacency matrix.
    """

    def __init__(self, output_dim=(2708, 2708), **kwargs):
        if K.backend() != "tensorflow":
            raise TypeError("Only tensorflow backend is currently supported.")
        self.output_dims = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.d_inv_sqrt_var = self.add_weight(
            name="d_inv_sqrt_var",
            shape=(self.output_dims[0],),
            initializer=keras.initializers.zeros(),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, adj):
        """
            The adjacency matrix pre-processing in tensorflow.
            This function applies the matrix transformations on the adjacency matrix, which are required by GCN.
            GCN requires that the input adjacency matrix should be symmetric, with self-loops, and normalized.

            Args:
                adj (Numpy array): the adjacency matrix to transform.

            Returns:
                The tensor of the transformed adjacency matrix.
        """

        # Build a symmetric adjacency matrix.
        adj_T = tf.transpose(adj)
        adj = (
            adj
            + tf.multiply(
                adj_T, tf.where(adj_T > adj, tf.ones_like(adj), tf.zeros_like(adj))
            )
            - tf.multiply(
                adj, tf.where(adj_T > adj, tf.ones_like(adj), tf.zeros_like(adj))
            )
        )
        # Add self loops.
        adj = adj + tf.eye(tf.shape(adj)[0])
        # Normalization
        rowsum = tf.reduce_sum(adj, 1)
        p = tf.fill(tf.shape(rowsum), -0.5)
        d_inv_sqrt = tf.pow(rowsum, p)
        self.d_inv_sqrt_var = tf.assign(self.d_inv_sqrt_var, d_inv_sqrt)
        tf.assign(
            self.d_inv_sqrt_var,
            tf.where(
                tf.is_inf(self.d_inv_sqrt_var),
                tf.zeros_like(self.d_inv_sqrt_var),
                self.d_inv_sqrt_var,
            ),
        )
        d_mat_inv_sqrt = tf.diag(self.d_inv_sqrt_var)
        adj_normalized = tf.matmul(tf.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj_normalized
