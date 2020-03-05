# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
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

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import (
    Input,
    Layer,
    Lambda,
    Dropout,
    Reshape,
    Masking,
    GlobalAveragePooling1D,
)

from ..mapper import GraphGenerator
from .cluster_gcn import ClusterGraphConvolution
from ..core.experimental import experimental


@experimental(reason="Missing unit tests")
class GraphClassificationConvolution(ClusterGraphConvolution):

    """
    A graph convolutional Keras layer. A stack of such layers can be used to create a model for supervised graph
    classification.

    The implementation is based on the GCN Keras layer of keras-gcn github
    repo https://github.com/tkipf/keras-gcn

    Notes:
      - The inputs are tensors with a batch dimension.

      - There are 2 inputs required, the node features and the normalized graph adjacency matrix.

      - This class assumes that the normalized graph adjacency matrix is passed as
        input to the Keras methods.

    Args:
        units (int): dimensionality of output feature vectors
        activation (str): nonlinear activation applied to layer's output to obtain output features
        use_bias (bool): toggles an optional bias
        kernel_initializer (str): name of layer bias f the initializer for kernel parameters (weights)
        bias_initializer (str): name of the initializer for bias
        kernel_regularizer (str): name of regularizer to be applied to layer kernel. Must be a Keras regularizer.
        bias_regularizer (str): name of regularizer to be applied to layer bias. Must be a Keras regularizer.
        activity_regularizer (str): not used in the current implementation
        kernel_constraint (str): constraint applied to layer's kernel
        bias_constraint (str): constraint applied to layer's bias
    """

    def call(self, inputs):
        """
        Applies the layer.

        Args:
            inputs (list or tuple): a list or tuple of 2 input tensors that includes
                node features (size batch_size x N x F),
                graph adjacency matrix (batch_size x N x N),
                where N is the number of nodes in the graph, and
                F is the dimensionality of node features.

        Returns:
            Keras Tensor that represents the output of the layer.
        """
        features, A = inputs

        h_graph = K.batch_dot(A, features)
        output = K.dot(h_graph, self.kernel)

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias
        output = self.activation(output)

        return output


@experimental(reason="Missing unit tests")
class GraphClassification:
    """
    A stack of :class:`GraphClassificationConvolution` layers together with a `GlobalAveragePooling1D` layer
    that implement a supervised graph classification network using the GCN convolution operator
    (https://arxiv.org/abs/1609.02907).

    The model minimally requires specification of the GCN layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer,
    activation functions for each hidden layers, and a generator object.

    To use this class as a Keras model, the features and pre-processed adjacency matrix
    should be supplied using the :class:`GraphGenerator` class.

    Notes:
      - The inputs are tensors provided by the :class:`GraphGenerator` object.

    Examples:
        Creating a graph classification model from a list of :class:`StellarGraph`
        object ``graphs``. We also add a fully connected dense layer and a binary classification
        layer with `softmax` activation::

            generator = GraphGenerator(graphs)
            model = GraphClassification(
                             layer_sizes=[32, 32],
                             activations=["elu","elu"],
                             generator=generator,
                             dropout=0.5
                )
            x_inp, x_out = model.build()
            predictions = Dense(units=8, activation='relu')(x_out)
            predictions = Dense(units=2, activation='softmax')(predictions)

    Args:
        layer_sizes (list of int): list of output sizes of the graph convolutional layers in the stack.
        activations (list of str): list of activations applied to each layer's output.
        generator (GraphGenerator): an instance of GraphGenerator class constructed on the graphs used for training.
        bias (bool): toggles an optional bias in graph convolutional layers.
        dropout (float): dropout rate applied to input features of each graph convolutional layer.
        kernel_regularizer (str): normalization applied to the kernels of graph convolutional layers.
    """

    def __init__(
        self, layer_sizes, activations, generator, bias=True, dropout=0.0, **kwargs
    ):
        if not isinstance(generator, GraphGenerator):
            raise TypeError("Generator should be a instance of GraphGenerator")

        if len(layer_sizes) != len(activations):
            raise AssertionError(
                "The number of given layers should be the same as the number of activations."
                "However given len(layer_sizes): {} vs len(activations): {}".format(
                    len(layer_sizes), len(activations)
                )
            )

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.bias = bias
        self.dropout = dropout
        self.generator = generator

        # Optional regulariser, etc. for weights and biases
        self._get_regularisers_from_keywords(kwargs)

        # Initialize a stack of GraphClassificationConvolution layers
        n_layers = len(self.layer_sizes)
        self._layers = []
        for ii in range(n_layers):
            l = self.layer_sizes[ii]
            a = self.activations[ii]
            self._layers.append(Dropout(self.dropout))
            self._layers.append(
                GraphClassificationConvolution(
                    l, activation=a, use_bias=self.bias, **self._regularisers
                )
            )

    def _get_regularisers_from_keywords(self, kwargs):
        regularisers = {}
        for param_name in [
            "kernel_initializer",
            "kernel_regularizer",
            "kernel_constraint",
            "bias_initializer",
            "bias_regularizer",
            "bias_constraint",
        ]:
            param_value = kwargs.pop(param_name, None)
            if param_value is not None:
                regularisers[param_name] = param_value
        self._regularisers = regularisers

    def __call__(self, x):
        """
        Apply a stack of GraphClassificationConvolution to the inputs.
        The input tensors are expected to be a list of the following:
        [
            Node features shape (batch size, N, F),
            Mask (batch size, N ),
            Adjacency matrices (batch size, N, N),
        ]
        where N is the number of nodes and F the number of input features

        Args:
            x (Tensor): input tensors

        Returns:
            Output tensor
        """
        x_in, mask, As = x
        h_layer = x_in

        for layer in self._layers:
            if isinstance(layer, GraphClassificationConvolution):
                h_layer = layer([h_layer, As])
            else:
                # For other (non-graph) layers only supply the input tensor
                h_layer = layer(h_layer)

        # add mean pooling layer with mask in order to ignore the padded values
        h_layer = GlobalAveragePooling1D(data_format="channels_last")(
            h_layer, mask=mask
        )

        return h_layer

    def build(self):
        """
        Builds a Graph Classification model for node prediction.

        Returns:
            tuple: `(x_inp, x_out)`, where `x_inp` is a list of two input tensors for the
            Graph Classification model (containing node features and normalized adjacency matrix),
            and `x_out` is a tensor for the Graph Classification model output.
        """
        x_t = Input(shape=(None, self.generator.node_features_size))
        mask = Input(shape=(None,), dtype=tf.bool)
        A_m = Input(shape=(None, None))

        x_inp = [x_t, mask, A_m]
        x_out = self(x_inp)

        return x_inp, x_out
