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
attri2vec

"""
__all__ = ["Attri2Vec"]

from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, Reshape, Embedding
import warnings
from .misc import deprecated_model_function
from ..mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator


def _require_without_generator(value, name):
    if value is not None:
        return value
    else:
        raise ValueError(
            f"{name}: expected a value for 'input_dim', 'node_num' and 'multiplicity' when "
            f"'generator' is not provided, found {name}=None."
        )


class Attri2Vec:
    """
    Implementation of the attri2vec algorithm of Zhang et al. with Keras layers.
    see: https://arxiv.org/abs/1901.04095.

    The model minimally requires specification of the layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer and a generator object.

    Args:
        layer_sizes (list): Hidden feature dimensions for each layer.
        generator (Sequence): A NodeSequence or LinkSequence.
        bias (bool): If True a bias vector is learnt for each layer in the attri2vec model, default to False.
        activation (str): The activation function of each layer in the attri2vec model, which takes values from "linear", "relu" and "sigmoid"(default).
        normalize ("l2" or None): The normalization used after each layer, default to None.
        input_dim (int, optional): The dimensions of the node features used as input to the model.
        node_num (int, optional): The number of nodes in the given graph.
        multiplicity (int, optional): The number of nodes to process at a time. This is 1 for a node
            inference and 2 for link inference (currently no others are supported).

    .. note::
        The values for ``input_dim``, ``node_num``, and ``multiplicity`` are obtained from the
        provided ``generator`` by default. The additional keyword arguments for these parameters
        provide an alternative way to specify them if a generator cannot be supplied.

    """

    def __init__(
        self,
        layer_sizes,
        generator=None,
        bias=False,
        activation="sigmoid",
        normalize=None,
        input_dim=None,
        node_num=None,
        multiplicity=None,
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

        # Get the model parameters from the generator or the keyword arguments
        if generator is not None:
            self._get_sizes_from_generator(generator)
        else:
            self.input_node_num = _require_without_generator(node_num, "node_num")
            self.input_feature_size = _require_without_generator(input_dim, "input_dim")
            self.multiplicity = _require_without_generator(multiplicity, "multiplicity")

        # Model parameters
        self.n_layers = len(layer_sizes)
        self.bias = bias

        # Feature dimensions for each layer
        self.dims = [self.input_feature_size] + layer_sizes

        # store the trainable layers
        self._layers = [
            Dense(layer_size, activation=self.activation, use_bias=self.bias)
            for layer_size in layer_sizes
        ]

        if self.multiplicity == 1:
            self._output_embedding = None
        else:
            self._output_embedding = Embedding(
                self.input_node_num,
                layer_sizes[-1],
                input_length=1,
                name="output_embedding",
            )

    def _get_sizes_from_generator(self, generator):
        """
        Sets node_num and input_feature_size from the generator.
        Args:
             generator: The supplied generator.
        """
        if not isinstance(generator, (Attri2VecNodeGenerator, Attri2VecLinkGenerator)):
            raise TypeError(
                "Generator should be an instance of Attri2VecNodeGenerator or Attri2VecLinkGenerator"
            )

        self.multiplicity = generator.multiplicity
        self.input_node_num = generator.graph.number_of_nodes()

        feature_sizes = generator.graph.node_feature_sizes()
        if len(feature_sizes) > 1:
            raise RuntimeError(
                "Attri2Vec called on graph with more than one node type."
            )
        self.input_feature_size = feature_sizes.popitem()[1]

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
        for layer in self._layers:
            h_layer = self._normalization(layer(h_layer))

        return h_layer

    def _node_model(self):
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

    def _link_model(self):
        """
        Builds a Attri2Vec model for context node prediction.

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors for (src, dst) nodes in the node pairs
            and ``x_out`` is a list of output tensors for (src, dst) nodes in the node pairs

        """
        # Expose input and output sockets of the model, for source node:
        x_inp_src, x_out_src = self._node_model()

        # Expose input and out sockets of the model, for target node:
        x_inp_dst = Input(shape=(1,))

        assert isinstance(self._output_embedding, Embedding)
        x_out_dst = self._output_embedding(x_inp_dst)
        x_out_dst = Reshape((self.dims[self.n_layers],))(x_out_dst)

        x_inp = [x_inp_src, x_inp_dst]
        x_out = [x_out_src, x_out_dst]
        return x_inp, x_out

    def in_out_tensors(self, multiplicity=None):
        """
        Builds a Attri2Vec model for node or link/node pair prediction, depending on the generator used to construct
        the model (whether it is a node or link/node pair generator).

        Returns:
            tuple: (x_inp, x_out), where ``x_inp`` is a list of Keras input tensors
            for the specified Attri2Vec model (either node or link/node pair model) and ``x_out`` contains
            model output tensor(s) of shape (batch_size, layer_sizes[-1])

        """
        if multiplicity is None:
            multiplicity = self.multiplicity
        if multiplicity == 1:
            return self._node_model()
        elif multiplicity == 2:
            return self._link_model()
        else:
            raise RuntimeError(
                "Currently only multiplicities of 1 and 2 are supported. Consider using node_model or "
                "link_model method explicitly to build node or link prediction model, respectively."
            )

    def default_model(self, flatten_output=True):
        warnings.warn(
            "The .default_model() method is deprecated. Please use .in_out_tensors() method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.in_out_tensors()

    node_model = deprecated_model_function(_node_model, "node_model")
    link_model = deprecated_model_function(_link_model, "link_model")
    build = deprecated_model_function(in_out_tensors, "build")
