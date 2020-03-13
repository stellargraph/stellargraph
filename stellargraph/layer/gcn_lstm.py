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

import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import (
    activations,
    initializers,
    constraints,
    regularizers,
    Sequential,
    Model,
)
from tensorflow.keras.layers import Input, Layer, Lambda, Dropout, Reshape, LSTM, Dense

from ..mapper import FullBatchGenerator
from .misc import SqueezedSparseConversion
from .preprocessing_layer import GraphPreProcessingLayer


class GraphConvolution(Layer):

    """
    Graph Convolution (GCN) Keras layer.
    The implementation is based on the keras-gcn github repo https://github.com/tkipf/keras-gcn.

    Original paper: Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling,
    International Conference on Learning Representations (ICLR), 2017 https://github.com/tkipf/gcn

    Notes:
      - The inputs are tensors with a batch dimension:
        Keras requires this batch dimension, and for full-batch methods
        we only have a single "batch".

      - There are three inputs required, the node features: 
         this is a 3 dimensional array, batch size, sequence length, and number of nodes,
        the output indices (the nodes that are to be selected in the final layer),
        and the normalized graph Laplacian matrix

      - This class assumes that a simple unweighted or wegited adjacency matrix is passed to it,
          the normalized Laplacian matrix is calculated within the class.

      - The output indices are used when ``final_layer=True`` and the returned outputs
        are the final-layer features for the nodes indexed by output indices.

      - If ``final_layer=False`` all the node features are output in the same ordering as
        given by the adjacency matrix.

    Args:
        units (int): dimensionality of output feature vectors
        activation (str or func): nonlinear activation applied to layer's output to obtain output features
        use_bias (bool): toggles an optional bias
        final_layer (bool): If False the layer returns output for all nodes,
                            if True it returns the subset specified by the indices passed to it.
        kernel_initializer (str or func, optional): The initialiser to use for the weights.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights.
        kernel_constraint (str or func, optional): The constraint to use for the weights.
        bias_initializer (str or func, optional): The initialiser to use for the bias.
        bias_regularizer (str or func, optional): The regulariser to use for the bias.
        bias_constraint (str or func, optional): The constraint to use for the bias.
    """

    def __init__(
        self,
        units,
        A,
        activation=None,
        use_bias=True,
        input_dim=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs
    ):
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        self.units = units,
        self.adj = self.calculate_laplacian(A)

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        
        self.final_layer = "False"

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        super().__init__(**kwargs)

    def get_config(self):
        """
        Gets class configuration for Keras serialization.
        Used by keras model serialization.

        Returns:
            A dictionary that contains the config of the layer
        """

        config = {
            "units": self.units,
            "use_bias": self.use_bias,
            "final_layer": self.final_layer,
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def calculate_laplacian(self, adj):
        D = np.diag(np.ravel(adj.sum(axis=0)) ** (-0.5))
        adj = np.dot(D, np.dot(adj, D))
        return adj

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.
        Assumes the following inputs:

        Args:
            input_shapes (tuple of ints)
                Shape tuples can include None for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """
        feature_shape, As_shape = input_shapes

        return feature_shape[0], feature_shape[1], self.units

    def set_A(self, A):
        """
        Sets the adjacnency matrix provided as input.
        
        """
        K.set_value(self.A, A)

    def build(self, input_shapes):
        """
        Builds the layer

        Args:
            input_shapes (list of int): shapes of the layer's inputs (node features and adjacency matrix)

        """
        n_nodes = input_shapes[-1]
        t_steps = input_shapes[-2]
        
        self.A = self.add_weight(
            name="A",
            shape=(n_nodes, n_nodes),
            trainable=False,
            initializer=initializers.constant(self.adj),
        )

        # K.set_value(self.A, self.adj)

        self.kernel = self.add_weight(
            shape=(t_steps, t_steps),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(n_nodes,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, features):
        """
        Applies the layer.

        Args:
            inputs (ndarray): node features (size B x T x N),
                where B is the batch size,
                      T is the sequence length, and
                      N is the number of nodes in the graph.

        Returns:
            Keras Tensor that represents the output of the layer.
        """

        # Calculate the layer operation of GCN

        h_graph = K.dot(features, self.A)
        output = tf.transpose(
            K.dot(tf.transpose(h_graph, [0, 2, 1]), self.kernel), [0, 2, 1]
        )

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias

        output = self.activation(output)

        return output


class Graph_Convolution_LSTM(Model):
    
    """
    A stack of 2 Graph Convolutional layers followed by an LSTM, Dropout and,  Dense layer.
    
    This architecture is inspired by:  

    The model minimally requires specification of the layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer,
    activation functions for each hidden layers, and a generator object.

    To use this class as a Keras model, the features and pre-processed adjacency matrix
    should be supplied using either the :class:`FullBatchNodeGenerator` class for node inference
    or the :class:`FullBatchLinkGenerator` class for link inference.

    To have the appropriate pre-processing the generator object should be instanciated
    with the `method='gcn'` argument.

    Note that currently the GCN class is compatible with both sparse and dense adjacency
    matrices and the :class:`FullBatchNodeGenerator` will default to sparse.

    For more details, please see the GCN demo notebook:
    demos/node-classification/gat/gcn-cora-node-classification-example.ipynb

    Example:
        Creating a GCN node classification model from an existing :class:`StellarGraph`
        object ``G``::

            generator = FullBatchNodeGenerator(G, method="gcn")
            gcn = GCN(
                    layer_sizes=[32, 4],
                    activations=["elu","softmax"],
                    generator=generator,
                    dropout=0.5
                )
            x_inp, predictions = gcn.build()

    Notes:
      - The inputs are tensors with a batch dimension of 1. These are provided by the \
        :class:`FullBatchNodeGenerator` object.

      - This assumes that the normalized Lapalacian matrix is provided as input to
        Keras methods. When using the :class:`FullBatchNodeGenerator` specify the
        ``method='gcn'`` argument to do this pre-processing.

      - The nodes provided to the :class:`FullBatchNodeGenerator.flow` method are
        used by the final layer to select the predictions for those nodes in order.
        However, the intermediate layers before the final layer order the nodes
        in the same way as the adjacency matrix.

    Args:
        layer_sizes (list of int): Output sizes of GCN layers in the stack.
        generator (FullBatchNodeGenerator): The generator instance.
        bias (bool): If True, a bias vector is learnt for each layer in the GCN model.
        dropout (float): Dropout rate applied to input features of each GCN layer.
        activations (list of str or func): Activations applied to each layer's output;
            defaults to ['relu', ..., 'relu'].
        kernel_initializer (str or func, optional): The initialiser to use for the weights of each layer.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights of each layer.
        kernel_constraint (str or func, optional): The constraint to use for the weights of each layer.
        bias_initializer (str or func, optional): The initialiser to use for the bias of each layer.
        bias_regularizer (str or func, optional): The regulariser to use for the bias of each layer.
        bias_constraint (str or func, optional): The constraint to use for the bias of each layer.
    """
    
    def __init__(
        self,
        outputs,
        adj,
        layer_sizes=[10,200],
        bias=True,
        dropout=0.5,
        activations=["relu","relu","sigmoid"],
        kernel_initializer=None,
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer=None,
        bias_regularizer=None,
        bias_constraint=None,
    ):

        super(Graph_Convolution_LSTM, self).__init__()

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.bias = bias
        self.dropout = dropout
        self.outputs = outputs

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.gcn_1 = GraphConvolution(units = layer_sizes[0], A=adj, activation=self.activations[0])
        self.gcn_2 = GraphConvolution(units = layer_sizes[0], A=adj, activation=self.activations[1])
        self.lstm = LSTM(self.layer_sizes[1], return_sequences=False)
        self.dense = Dense(self.outputs, activation=self.activations[2])
        self.dropout = Dropout(self.dropout)

    def call(self, inputs):
        layer = self.gcn_1(inputs)
        layer = self.gcn_2(layer)
        layer = self.lstm(layer)
        layer = self.dropout(layer)
        layer = self.dense(layer)
        return layer
