# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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


from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow import keras
import numpy as np


class SymmetricGraphPreProcessingLayer(Layer):
    """
    This class implements the pre-processing of adjacency matrices in GCN. We implement it in tensorflow so that
    while computing the saliency maps, we are able to calculate the gradients in an end-to-end way.
    We currently only support this for tensorflow backend.

    Args:
    num_of_nodes (int pair): The number of nodes in the graph.
    """

    def __init__(self, num_of_nodes, **kwargs):
        self.output_dims = (num_of_nodes, num_of_nodes)
        super().__init__(**kwargs)

    def build(self, input_shape):
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
        adj = adj + tf.linalg.diag(tf.ones(adj.shape[0]) - tf.diag_part(adj))
        # Normalization
        rowsum = tf.reduce_sum(adj, 1)
        d_mat_inv_sqrt = tf.diag(tf.rsqrt(rowsum))
        adj_normalized = tf.matmul(tf.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj_normalized


class GraphPreProcessingLayer(Layer):
    """
    This class implements the pre-processing of adjacency matrices in GCN. We implement it in tensorflow so that
    while computing the saliency maps, we are able to calculate the gradients in an end-to-end way.
    We currently only support this for tensorflow backend.

    Args:
    num_of_nodes (int pair): The number of nodes in the graph.
    """

    def __init__(self, num_of_nodes, **kwargs):
        self.output_dims = (num_of_nodes, num_of_nodes)
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, adj):
        """
            The adjacency matrix pre-processing in tensorflow.
            This function applies the matrix transformations on the adjacency matrix, which are required by GCN.
            GCN requires that the input adjacency matrix has self-loops and is normalized.

            Args:
                adj (Numpy array): the adjacency matrix to transform.

            Returns:
                The tensor of the transformed adjacency matrix.
        """
        if K.is_sparse(adj):  # isinstance(adj, tf.SparseTensor):
            raise RuntimeError(
                "Tensorflow adjacency matrix normalization not implemented for sparse matrices."
            )

        else:
            # Add self loops.
            adj = adj + tf.linalg.diag(tf.ones(adj.shape[0]) - tf.linalg.diag_part(adj))

            # Normalization
            rowsum = tf.reduce_sum(adj, 1)
            d_mat_inv_sqrt = tf.linalg.diag(tf.math.rsqrt(rowsum))
            adj_normalized = tf.matmul(tf.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
            return adj_normalized
