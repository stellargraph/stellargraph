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


from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
from keras import backend as K
from keras import Input
from keras.layers import Lambda, Dropout, Reshape
from ..mapper.node_mappers import FullBatchNodeGenerator

from typing import List, Tuple, Callable, AnyStr


class GraphConvolution(Layer):

    """
    Graph Convolution (GCN) Keras layer.
    The implementation is based on the keras-gcn github repo https://github.com/tkipf/keras-gcn.

    Original paper: Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling,
    International Conference on Learning Representations (ICLR), 2017 https://github.com/tkipf/gcn

    Args:
        units: dimensionality of output feature vectors
        support: number of support weights
        activation: nonlinear activation applied to layer's output to obtain output features
        use_bias: toggles an optional bias
        kernel_initializer (str): name of layer bias f the initializer for kernel parameters (weights)
        bias_initializer (str): name of the initializer for bias
        attn_kernel_initializer (str): name of the initializer for attention kernel
        kernel_regularizer (str): name of regularizer to be applied to layer kernel. Must be a Keras regularizer.
        bias_regularizer (str): name of regularizer to be applied to layer bias. Must be a Keras regularizer.
        activity_regularizer (str): not used in the current implementation
        kernel_constraint (str): constraint applied to layer's kernel
        bias_constraint (str): constraint applied to layer's bias
        **kwargs:
    """

    def __init__(
        self,
        units,
        support=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.get("input_dim"),)

        super(GraphConvolution, self).__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.support = support

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.
        Assumes that the layer will be built to match that input shape provided.

        Args:
            input_shape (tuple of ints)
                Shape tuples can include None for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """

        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        """
        Builds the layer

        Args:
            input_shape (list of list of int): Shape of input tensors for self
            and neighbour

        """

        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        input_dim = features_shape[1]

        self.kernel = self.add_weight(
            shape=(input_dim * self.support, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        """
        Applies the layer.

        Args:
            inputs: a list of input tensors that includes 2 items: node features (matrix of size N x F),
                and graph adjacency matrix (size N x N), where N is the number of nodes in the graph,
                F is the dimensionality of node features.
            mask: This mask is only used as an tranmission function. It passes the corresponding mask from the previous layer
                to the next attached layer if the previous layer set a mask.
        """

        features = inputs[0]
        A = inputs[1:]

        supports = list()
        for i in range(self.support):
            supports.append(K.dot(A[i], features))
        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.kernel)

        if self.bias:
            output += self.bias
        return self.activation(output)

    def get_config(self):
        """
        Gets class configuration for Keras serialization
        """

        config = {
            "units": self.units,
            "use_bias": self.use_bias,
            "support": self.support,
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }

        base_config = super().get_config()
        return {**base_config, **config}


class GCN:
    """
    A stack of Graph Convolutional layers to implement the graph convolution network model as in https://arxiv.org/abs/1609.02907

    The model minimally requires specification of the layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer,
    activation functions for each hidden layers, and a generator object.

    Args:
        layer_sizes: list of output sizes of GCN layers in the stack
        activations: list of activations applied to each layer's output
        generator: an instance of FullBatchNodeGenerator class constructed on the graph of interest
        bias: toggles an optional bias in GCN layers
        dropout: dropout rate applied to input features of each GCN layer
        kernel_regularizer: normalization applied to the kernels of GCN layers
        kwargs: additional parameters for chebyshev or localpool filters
    """

    def __init__(
        self,
        layer_sizes,
        activations,
        generator,
        bias=True,
        dropout=0.0,
        kernel_regularizer=None,
        **kwargs
    ):
        if not isinstance(generator, FullBatchNodeGenerator):
            raise TypeError("Generator should be a instance of FullBatchNodeGenerator")

        assert len(layer_sizes) == len(activations)

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.bias = bias
        self.dropout = dropout
        self.kernel_regularizer = kernel_regularizer
        self.generator = generator
        self.support = 1
        self.kwargs = kwargs

        # Initialize a stack of GCN layers
        self._layers = []
        for l, a in zip(self.layer_sizes, self.activations):
            self._layers.append(Dropout(self.dropout))
            self._layers.append(
                GraphConvolution(
                    l,
                    self.support,
                    activation=a,
                    use_bias=self.bias,
                    kernel_regularizer=self.kernel_regularizer,
                )
            )

    def __call__(self, x: List):
        """
        Apply a stack of GCN layers

        Args:
            x (list of Tensor): input features

        Returns:
            Output tensor
        """

        H = x[0]
        suppG = x[1:]

        for layer in self._layers:
            if isinstance(layer, GraphConvolution):
                # It is a GCN layer
                H = layer([H] + suppG)
            else:
                # layer is a Dropout layer
                H = layer(H)

        return H

    def node_model(self):
        """
        Builds a GCN model for node prediction

        Returns:
            tuple: (x_inp, x_out) where `x_inp` is a Keras input tensor
                for the specified GCN model and `x_out` is a Keras tensor for the GCN model output.
        """

        x_in = Input(shape=(self.generator.features.shape[1],))

        filter = self.kwargs.get("filter", "localpool")
        if filter == "chebyshev":
            self.support = self.kwargs.get("max_degree", 2)
            suppG = [
                Input(batch_shape=(None, None), sparse=True)
                for _ in range(self.support)
            ]
        else:
            suppG = [Input(batch_shape=(None, None), sparse=True)]

        x_inp = [x_in] + suppG
        x_out = self(x_inp)
        return x_inp, x_out

    # NOTE: Temporarily remove this function from sphinx doc because it has not been implemented
    def _link_model(self, flatten_output=False):
        """
        Builds a GCN model for link (node pair) prediction

        Args:
            flatten_output:
        Returns:
            NotImplemented
        """
        raise NotImplemented
