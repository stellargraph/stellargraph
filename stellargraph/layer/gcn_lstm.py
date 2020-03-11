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
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers, Sequential, Model
from tensorflow.keras.layers import Input, Layer, Lambda, Dropout, Reshape, LSTM, Dense

from ..mapper import FullBatchGenerator
from .misc import SqueezedSparseConversion
from .preprocessing_layer import GraphPreProcessingLayer


class GraphConvolution1(Layer):

    """
    Graph Convolution (GCN) Keras layer.
    The implementation is based on the keras-gcn github repo https://github.com/tkipf/keras-gcn.

    Original paper: Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling,
    International Conference on Learning Representations (ICLR), 2017 https://github.com/tkipf/gcn

    Notes:
      - The inputs are tensors with a batch dimension of 1:
        Keras requires this batch dimension, and for full-batch methods
        we only have a single "batch".

      - There are three inputs required, the node features, the output
        indices (the nodes that are to be selected in the final layer)
        and the normalized graph Laplacian matrix

      - This class assumes that the normalized Laplacian matrix is passed as
        input to the Keras methods.

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

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        
        self.adj = A

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
            name="A", shape=(n_nodes, n_nodes), 
            trainable=False, initializer=initializers.constant(self.adj)
        )
        
        #K.set_value(self.A, self.adj)
        
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
            inputs (list): a list of 3 input tensors that includes
                node features (size B x T x N),
                where N is the number of nodes in the graph, and

        Returns:
            Keras Tensor that represents the output of the layer.
        """
        
        # Calculate the layer operation of GCN
        # output = self.kernel * K.dot(features, self.A)
        
        h_graph = K.dot(features, self.A)
        output = tf.transpose(
                K.dot(tf.transpose(h_graph, [0,2,1]), self.kernel),
                [0,2,1]
                )
        
        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias
            
        output = self.activation(output)

        return output




class Graph_Convolution_LSTM(Model):
    
    def __init__(self, outputs, adj):
        super(Graph_Convolution_LSTM, self).__init__()
        self.gcn1 = GraphConvolution1(A = adj, activation = 'relu' )
        self.gcn2 = GraphConvolution1(A = adj, activation = 'relu' )
        self.lstm = LSTM(200, return_sequences = False)
        self.dense = Dense(outputs, activation='sigmoid')
        self.dropout = Dropout(0.5)

    def call(self, inputs):
        x = self.gcn1(inputs)
        x = self.gcn2(x)
        x = self.lstm(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x 

class GCN_LSTM:
    """
    """
    def __init__(
        self,
        train_shape,
        adj,
        layer_sizes = 200,
        bias=True,
        dropout=0.0,
        activations=None,
        kernel_initializer=None,
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer=None,
        bias_regularizer=None,
        bias_constraint=None,
    ):
        

      
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.bias = bias
        self.dropout = dropout
        self.adj = adj

        # Copy required information from generator
       # self.method = generator.method
        self.seq_len = train_shape[0]
        self.n_nodes = train_shape[1]
        self.activations = activations

        # Initialize a stack of graph convolution and LSTM layers
       
        self.gcn_l = GraphConvolution1(activation = 'relu')
        self.lstm_l = LSTM(layer_sizes, return_sequences = False)
        self.dropout_l = Dropout(self.dropout)
        self.dense_l = Dense(self.n_nodes, activation = 'sigmoid')
        
        
    def __call__(self, x):
        """
        Apply a stack of GCN layers to the inputs.
        The input tensors are expected to be a list of the following:
        [
            Node features shape (1, N, F),
            Adjacency indices (1, E, 2),
            Adjacency values (1, E),
            Output indices (1, O)
        ]
        where N is the number of nodes, F the number of input features,
              E is the number of edges, O the number of output nodes.

        Args:
            x (Tensor): input tensors

        Returns:
            Output tensor
        """
        x_features, x_adj = x

        # Currently we require the batch dimension to be one for full-batch methods
        seq_dim, n_nodes, _ = K.int_shape(x_features)
       
       
        h_layer = self.gcn_l(input_shape = (x_features.shape[1], x_features.shape[2],))
        h_layer = self.lstm_l(h_layer)
        h_layer = self.dropout_l(h_layer)
        h_layer  = self.dense_l(h_layer)
        
        return h_layer
                
                
   