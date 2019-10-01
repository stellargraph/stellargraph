# -*- coding: utf-8 -*-
#
# Copyright 2018 Data61, CSIRO
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
attri2vec

"""
__all__ = ["Attri2Vec"]

from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, Reshape, Embedding
import warnings


class Attri2Vec:
    """
    Implementation of the attri2vec algorithm of Zhang et al. with Keras layers.
    see: https://arxiv.org/abs/1901.04095.

    The model minimally requires specification of the layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer and a generator object.

    Args:
        layer_sizes (list): Hidden feature dimensions for each layer.
        generator (Sequence): A NodeSequence or LinkSequence. 
        input_dim (int): The dimensions of the node features used as input to the model.
        node_num (int): The number of nodes in the given graph.
        bias (bool): If True a bias vector is learnt for each layer in the attri2vec model, default to False.
        activation (str): The activation function of each layer in the attri2vec model, which takes values from "linear", "relu" and "sigmoid"(default).
        normalize ("l2" or None): The normalization used after each layer, default to None.

    """

    def __init__(
        self,
        layer_sizes,
        generator=None,
        input_dim=None,
        node_num=None,
        bias=False,
        activation="sigmoid",
        normalize=None,
    ):

        if activation == "linear" or activation == "relu" or activation == "sigmoid":
            self.activation = activation
        else:
            raise ValueError(
                "Activation should be either 'linear', 'relu' or 'sigmoid'; received '{}'".format(
                    activation
                )
            )

        if normalize == "l2":
            self._normalization = Lambda(lambda x: K.l2_normalize(x, axis=-1))

        elif normalize is None:
            self._normalization = Lambda(lambda x: x)

        else:
            raise ValueError(
                "Normalization should be either 'l2' or None; received '{}'".format(
                    normalize
                )
            )

        # Get the input_dim and node_num from the generator if it is given
        self.generator = generator
        if generator is not None:
            feature_sizes = generator.generator.graph.node_feature_sizes()
            if len(feature_sizes) > 1:
                raise RuntimeError(
                    "Attri2Vec called on graph with more than one node type."
                )

            self.input_feature_size = feature_sizes.popitem()[1]
            self.input_node_num = generator.generator.graph.number_of_nodes()

        elif input_dim is not None and node_num is not None:
            self.input_feature_size = input_dim
            self.input_node_num = node_num

        else:
            raise RuntimeError(
                "If generator is not provided, input_dim and node_num must be specified."
            )

        # Model parameters
        self.n_layers = len(layer_sizes)
        self.bias = bias

        # Feature dimensions for each layer
        self.dims = [self.input_feature_size] + layer_sizes

    def __call__(self, xin):
        """
        Construct node representations from node attributes through deep neural network

        Args:
            xin (Keras Tensor): Batch input features

        Returns:
            Output tensor
        """
        # Form Attri2Vec layers iteratively
        h_layer = xin
        for layer in range(0, self.n_layers):
            h_layer = Dense(
                self.dims[layer + 1], activation=self.activation, use_bias=self.bias
            )(h_layer)
            h_layer = self._normalization(h_layer)

        return h_layer

    def node_model(self):
        """
        Builds a Attri2Vec model for node representation prediction.

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a Keras input tensor
            for the Attri2Vec model and ``x_out`` is the Keras tensor
            for the Attri2Vec model output.

        """
        # Create tensor inputs
        x_inp = Input(shape=(self.input_feature_size,))

        # Output from Attri2Vec model
        x_out = self(x_inp)

        return x_inp, x_out

    def link_model(self):
        """
        Builds a Attri2Vec model for context node prediction.

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors for (src, dst) nodes in the node pairs
            and ``x_out`` is a list of output tensors for (src, dst) nodes in the node pairs

        """
        # Expose input and output sockets of the model, for source node:
        x_inp_src, x_out_src = self.node_model()

        # Expose input and out sockets of the model, for target node:
        x_inp_dst = Input(shape=(1,))
        output_embedding = Embedding(
            self.input_node_num,
            self.dims[self.n_layers],
            input_length=1,
            name="output_embedding",
        )
        x_out_dst = output_embedding(x_inp_dst)
        x_out_dst = Reshape((self.dims[self.n_layers],))(x_out_dst)

        x_inp = [x_inp_src, x_inp_dst]
        x_out = [x_out_src, x_out_dst]
        return x_inp, x_out

    def build(self):
        """
        Builds a Attri2Vec model for context node prediction.

        Returns:
            tuple: (x_inp, x_out), where ``x_inp`` contains Keras input tensor(s)
            for the Attri2Vec model to perform context node prediction and ``x_out`` contains
            model output tensor(s) of shape (batch_size, layer_sizes[-1]).

        """
        return self.link_model()

    def default_model(self, flatten_output=True):
        warnings.warn(
            "The .default_model() method will be deprecated in future versions. "
            "Please use .build() method instead.",
            PendingDeprecationWarning,
        )
        return self.build()
