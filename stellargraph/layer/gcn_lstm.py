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

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Layer, Dropout, LSTM, Dense, Permute, Reshape
from ..mapper import SlidingFeaturesNodeGenerator
from ..core.experimental import experimental
from ..core.utils import calculate_laplacian


class FixedAdjacencyGraphConvolution(Layer):

    """
    Graph Convolution (GCN) Keras layer.
    The implementation is based on the keras-gcn github repo https://github.com/tkipf/keras-gcn.

    Original paper: Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling,
    International Conference on Learning Representations (ICLR), 2017 https://github.com/tkipf/gcn

    Notes:
      - The inputs are 3 dimensional tensors: batch size, sequence length, and number of nodes.
      - This class assumes that a simple unweighted or weighted adjacency matrix is passed to it,
        the normalized Laplacian matrix is calculated within the class.

    Args:
        units (int): dimensionality of output feature vectors
        A (N x N): weighted/unweighted adjacency matrix
        activation (str or func): nonlinear activation applied to layer's output to obtain output features
        use_bias (bool): toggles an optional bias
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
        **kwargs,
    ):
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        self.units = units
        self.adj = calculate_laplacian(A)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

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
        feature_shape = input_shapes

        return feature_shape[0], feature_shape[1], self.units

    def build(self, input_shapes):
        """
        Builds the layer

        Args:
            input_shapes (list of int): shapes of the layer's inputs (the batches of node features)

        """
        _batch_dim, n_nodes, features = input_shapes

        self.A = self.add_weight(
            name="A",
            shape=(n_nodes, n_nodes),
            trainable=False,
            initializer=initializers.constant(self.adj),
        )
        self.kernel = self.add_weight(
            shape=(features, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                # ensure the per-node bias can be broadcast across each feature
                shape=(n_nodes, 1),
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
            features (ndarray): node features (size B x N x F), where B is the batch size, F = TV is
                the feature size (consisting of the sequence length and the number of variates), and
                N is the number of nodes in the graph.

        Returns:
            Keras Tensor that represents the output of the layer.
        """

        # Calculate the layer operation of GCN
        # shape = B x F x N
        nodes_last = tf.transpose(features, [0, 2, 1])
        neighbours = K.dot(nodes_last, self.A)

        # shape = B x N x F
        h_graph = tf.transpose(neighbours, [0, 2, 1])
        # shape = B x N x units
        output = K.dot(h_graph, self.kernel)

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias

        output = self.activation(output)

        return output


@experimental(
    reason="Lack of unit tests and code refinement", issues=[1132, 1526, 1564]
)
class GCN_LSTM:

    """
    GCN_LSTM is a univariate timeseries forecasting method. The architecture  comprises of a stack of N1 Graph Convolutional layers followed by N2 LSTM layers, a Dropout layer, and  a Dense layer.
    This main components of GNN architecture is inspired by: T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction (https://arxiv.org/abs/1811.05320).
    The implementation of the above paper is based on one graph convolution layer stacked with a GRU layer.

    The StellarGraph implementation is built as a stack of the following set of layers:

    1. User specified no. of Graph Convolutional layers
    2. User specified no. of LSTM layers
    3. 1 Dense layer
    4. 1 Dropout layer.

    The last two layers consistently showed better performance and regularization experimentally.

    Args:
       seq_len: No. of LSTM cells
       adj: unweighted/weighted adjacency matrix of [no.of nodes by no. of nodes dimension
       gc_layer_sizes (list of int): Output sizes of Graph Convolution  layers in the stack.
       lstm_layer_sizes (list of int): Output sizes of LSTM layers in the stack.
       generator (SlidingFeaturesNodeGenerator): A generator instance.
       bias (bool): If True, a bias vector is learnt for each layer in the GCN model.
       dropout (float): Dropout rate applied to input features of each GCN layer.
       gc_activations (list of str or func): Activations applied to each layer's output; defaults to ['relu', ..., 'relu'].
       lstm_activations (list of str or func): Activations applied to each layer's output; sdefaults to ['tanh', ..., 'tanh'].
       kernel_initializer (str or func, optional): The initialiser to use for the weights of each layer.
       kernel_regularizer (str or func, optional): The regulariser to use for the weights of each layer.
       kernel_constraint (str or func, optional): The constraint to use for the weights of each layer.
       bias_initializer (str or func, optional): The initialiser to use for the bias of each layer.
       bias_regularizer (str or func, optional): The regulariser to use for the bias of each layer.
       bias_constraint (str or func, optional): The constraint to use for the bias of each layer.
     """

    def __init__(
        self,
        seq_len,
        adj,
        gc_layer_sizes,
        lstm_layer_sizes,
        gc_activations=None,
        generator=None,
        lstm_activations=None,
        bias=True,
        dropout=0.5,
        kernel_initializer=None,
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer=None,
        bias_regularizer=None,
        bias_constraint=None,
    ):
        if generator is not None:
            if not isinstance(generator, SlidingFeaturesNodeGenerator):
                raise ValueError(
                    f"generator: expected a SlidingFeaturesNodeGenerator, found {type(generator).__name__}"
                )

            if seq_len is not None or adj is not None:
                raise ValueError(
                    "expected only one of generator and (seq_len, adj) to be specified, found multiple"
                )

            adj = generator.graph.to_adjacency_matrix(weighted=True).todense()
            seq_len = generator.window_size
            variates = generator.variates
        else:
            variates = None

        super(GCN_LSTM, self).__init__()

        n_gc_layers = len(gc_layer_sizes)
        n_lstm_layers = len(lstm_layer_sizes)

        self.lstm_layer_sizes = lstm_layer_sizes
        self.gc_layer_sizes = gc_layer_sizes
        self.bias = bias
        self.dropout = dropout
        self.adj = adj
        self.n_nodes = adj.shape[0]
        self.n_features = seq_len
        self.seq_len = seq_len
        self.multivariate_input = variates is not None
        self.variates = variates if self.multivariate_input else 1
        self.outputs = self.n_nodes * self.variates

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        # Activation function for each gcn layer
        if gc_activations is None:
            gc_activations = ["relu"] * n_gc_layers
        elif len(gc_activations) != n_gc_layers:
            raise ValueError(
                "Invalid number of activations; require one function per graph convolution layer"
            )
        self.gc_activations = gc_activations

        # Activation function for each lstm layer
        if lstm_activations is None:
            lstm_activations = ["tanh"] * n_lstm_layers
        elif len(lstm_activations) != n_lstm_layers:
            padding_size = n_lstm_layers - len(lstm_activations)
            if padding_size > 0:
                lstm_activations = lstm_activations + ["tanh"] * padding_size
            else:
                raise ValueError(
                    "Invalid number of activations; require one function per lstm layer"
                )
        self.lstm_activations = lstm_activations

        self._gc_layers = [
            FixedAdjacencyGraphConvolution(
                units=self.variates * layer_size,
                A=self.adj,
                activation=activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint,
            )
            for layer_size, activation in zip(self.gc_layer_sizes, self.gc_activations)
        ]
        self._lstm_layers = [
            LSTM(layer_size, activation=activation, return_sequences=True)
            for layer_size, activation in zip(
                self.lstm_layer_sizes[:-1], self.lstm_activations
            )
        ]
        self._lstm_layers.append(
            LSTM(
                self.lstm_layer_sizes[-1],
                activation=self.lstm_activations[-1],
                return_sequences=False,
            )
        )
        self._decoder_layer = Dense(self.outputs, activation="sigmoid")

    def __call__(self, x):

        x_in, out_indices = x

        h_layer = x_in
        if not self.multivariate_input:
            # normalize to always have a final variate dimension, with V = 1 if it doesn't exist
            # shape = B x N x T x 1
            h_layer = tf.expand_dims(h_layer, axis=-1)

        # flatten variates into sequences, for convolution
        # shape B x N x (TV)
        h_layer = Reshape((self.n_nodes, self.seq_len * self.variates))(h_layer)

        for layer in self._gc_layers:
            h_layer = layer(h_layer)

        # return the layer to its natural multivariate tensor form
        # shape B x N x T' x V (where T' is the sequence length of the last GC)
        h_layer = Reshape((self.n_nodes, -1, self.variates))(h_layer)
        # put time dimension first for LSTM layers
        # shape B x T' x N x V
        h_layer = Permute((2, 1, 3))(h_layer)
        # flatten the variates across all nodes, shape B x T' x (N V)
        h_layer = Reshape((-1, self.n_nodes * self.variates))(h_layer)

        for layer in self._lstm_layers:
            h_layer = layer(h_layer)

        h_layer = Dropout(self.dropout)(h_layer)
        h_layer = self._decoder_layer(h_layer)

        if self.multivariate_input:
            # flatten things out to the multivariate shape
            # shape B x N x V
            h_layer = Reshape((self.n_nodes, self.variates))(h_layer)

        return h_layer

    def in_out_tensors(self):
        """
        Builds a GCN model for node  feature prediction

        Returns:
            tuple: `(x_inp, x_out)`, where `x_inp` is a list of Keras/TensorFlow
            input tensors for the GCN model and `x_out` is a tensor of the GCN model output.
        """
        # Inputs for features
        if self.multivariate_input:
            shape = (None, self.n_nodes, self.n_features, self.variates)
        else:
            shape = (None, self.n_nodes, self.n_features)

        x_t = Input(batch_shape=shape)

        # Indices to gather for model output
        out_indices_t = Input(batch_shape=(None, self.n_nodes), dtype="int32")

        x_inp = [x_t, out_indices_t]
        x_out = self(x_inp)

        return x_inp[0], x_out
