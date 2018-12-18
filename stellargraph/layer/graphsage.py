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
GraphSAGE and compatible aggregator layers

"""
__all__ = [
    "GraphSAGE",
    "MeanAggregator",
    "MaxPoolingAggregator",
    "MeanPoolingAggregator",
]

import numpy as np
from keras.engine.topology import Layer
from keras import Input
from keras import backend as K
from keras.layers import Lambda, Dropout, Reshape
from keras.utils import Sequence
from keras import activations
from typing import List, Tuple, Callable, AnyStr


class MeanAggregator(Layer):
    """
    Mean Aggregator for GraphSAGE implemented with Keras base layer

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional bias
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.

    """

    def __init__(
        self,
        output_dim: int = 0,
        bias: bool = False,
        act: Callable or AnyStr = "relu",
        **kwargs
    ):
        self.output_dim = output_dim
        assert output_dim % 2 == 0
        self.half_output_dim = int(output_dim / 2)
        self.has_bias = bias
        self.act = activations.get(act)
        self.w_neigh = None
        self.w_self = None
        self.bias = None
        self._initializer = "glorot_uniform"
        super().__init__(**kwargs)

    def get_config(self):
        """
        Gets class configuration for Keras serialization

        """
        config = {
            "output_dim": self.output_dim,
            "bias": self.has_bias,
            "act": activations.serialize(self.act),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        """
        Builds layer

        Args:
            input_shape (list of list of int): Shape of input tensors for self
            and neighbour

        """
        self.w_neigh = self.add_weight(
            name="w_neigh",
            shape=(input_shape[1][3], self.half_output_dim),
            initializer=self._initializer,
            trainable=True,
        )
        self.w_self = self.add_weight(
            name="w_self",
            shape=(input_shape[0][2], self.half_output_dim),
            initializer=self._initializer,
            trainable=True,
        )
        if self.has_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.output_dim],
                initializer="zeros",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, x, **kwargs):
        """
        Apply MeanAggregation on input tensors, x

        Args:
          x: Keras Tensor

        Returns:
            Keras Tensor representing the aggregated embeddings in the input.

        """
        neigh_means = K.mean(x[1], axis=2)

        from_self = K.dot(x[0], self.w_self)
        from_neigh = K.dot(neigh_means, self.w_neigh)
        h_out = K.concatenate([from_self, from_neigh], axis=2)

        if self.has_bias:
            h_out = self.act(h_out + self.bias)
        else:
            h_out = self.act(h_out)

        return h_out

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.
        Assumes that the layer will be built to match that input shape provided.

        Args:
            input_shape (tuple of ints)
                Shape tuples can include None for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """
        return input_shape[0][0], input_shape[0][1], self.output_dim


class MaxPoolingAggregator(Layer):
    """
    Max Pooling Aggregator for GraphSAGE implemented with Keras base layer

    Implements the aggregator of Eq. (3) in Hamilton et al. (2017)

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional bias
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.

    """

    def __init__(
        self,
        output_dim: int = 0,
        bias: bool = False,
        act: Callable or AnyStr = "relu",
        **kwargs
    ):
        self.output_dim = output_dim
        assert output_dim % 2 == 0
        self.half_output_dim = int(output_dim / 2)

        # TODO: These should be user parameters
        self.hidden_dim = output_dim
        self.hidden_act = activations.get("relu")

        self.has_bias = bias
        self.act = activations.get(act)
        self.w_neigh = None
        self.w_self = None
        self.w_pool = None
        self.b_pool = None
        self.bias = None
        self._initializer = "glorot_uniform"
        super().__init__(**kwargs)

    def get_config(self):
        """
        Gets class configuration for Keras serialization

        """
        config = {
            "output_dim": self.output_dim,
            "bias": self.has_bias,
            "act": activations.serialize(self.act),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        """
        Builds layer

        Args:
            input_shape (list of list of int): Shape of input tensors for self
            and neighbour

        """
        self.w_pool = self.add_weight(
            name="w_pool",
            shape=(input_shape[1][3], self.hidden_dim),
            initializer=self._initializer,
            trainable=True,
        )
        self.b_pool = self.add_weight(
            name="b_pool",
            shape=(self.hidden_dim,),
            initializer=self._initializer,
            trainable=True,
        )
        self.w_neigh = self.add_weight(
            name="w_neigh",
            shape=(self.hidden_dim, self.half_output_dim),
            initializer=self._initializer,
            trainable=True,
        )
        self.w_self = self.add_weight(
            name="w_self",
            shape=(input_shape[0][2], self.half_output_dim),
            initializer=self._initializer,
            trainable=True,
        )
        if self.has_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.output_dim,),
                initializer="zeros",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, x, **kwargs):
        """
        Apply max pooling aggregation on input tensors, x

        Args:
          x: Keras Tensor

        Returns:
            Keras Tensor representing the aggregated embeddings in the input.

        """
        # x[0]: self vector (batch_size, head size, feature_size)
        # x[1]: neighbour vector (batch_size, head size, neighbours, feature_size)

        # Max-pooling of neighbours:
        # Pass neighbour features through a dense layer with self.w_pool, self.b_pool
        xw_neigh = self.hidden_act(K.dot(x[1], self.w_pool) + self.b_pool)
        # Take max over neighbours
        neigh_agg = K.max(xw_neigh, axis=2)

        # Apply separate self & neighbour weights and concatenate
        from_self = K.dot(x[0], self.w_self)
        from_neigh = K.dot(neigh_agg, self.w_neigh)
        h_out = K.concatenate([from_self, from_neigh], axis=2)

        if self.has_bias:
            h_out = self.act(h_out + self.bias)
        else:
            h_out = self.act(h_out)

        return h_out

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.
        Assumes that the layer will be built to match that input shape provided.

        Args:
            input_shape (tuple of ints)
                Shape tuples can include None for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """
        return input_shape[0][0], input_shape[0][1], self.output_dim


class MeanPoolingAggregator(Layer):
    """
    Mean Pooling Aggregator for GraphSAGE implemented with Keras base layer

    Implements the aggregator of Eq. (3) in Hamilton et al. (2017), with max pooling replaced with mean pooling

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional bias
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.

    """

    def __init__(
        self,
        output_dim: int = 0,
        bias: bool = False,
        act: Callable or AnyStr = "relu",
        **kwargs
    ):
        self.output_dim = output_dim
        assert output_dim % 2 == 0
        self.half_output_dim = int(output_dim / 2)

        # TODO: These should be user parameters
        self.hidden_dim = output_dim
        self.hidden_act = activations.get("relu")

        self.has_bias = bias
        self.act = activations.get(act)
        self.w_neigh = None
        self.w_self = None
        self.w_pool = None
        self.b_pool = None
        self.bias = None
        self._initializer = "glorot_uniform"
        super().__init__(**kwargs)

    def get_config(self):
        """
        Gets class configuration for Keras serialization

        """
        config = {
            "output_dim": self.output_dim,
            "bias": self.has_bias,
            "act": activations.serialize(self.act),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        """
        Builds layer

        Args:
            input_shape (list of list of int): Shape of input tensors for self
            and neighbour

        """
        self.w_pool = self.add_weight(
            name="w_pool",
            shape=(input_shape[1][3], self.hidden_dim),
            initializer=self._initializer,
            trainable=True,
        )
        self.b_pool = self.add_weight(
            name="b_pool",
            shape=(self.hidden_dim,),
            initializer=self._initializer,
            trainable=True,
        )
        self.w_neigh = self.add_weight(
            name="w_neigh",
            shape=(self.hidden_dim, self.half_output_dim),
            initializer=self._initializer,
            trainable=True,
        )
        self.w_self = self.add_weight(
            name="w_self",
            shape=(input_shape[0][2], self.half_output_dim),
            initializer=self._initializer,
            trainable=True,
        )
        if self.has_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.output_dim,),
                initializer="zeros",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, x, **kwargs):
        """
        Apply max pooling aggregation on input tensors, x

        Args:
          x: Keras Tensor

        Returns:
            Keras Tensor representing the aggregated embeddings in the input.

        """
        # x[0]: self vector (batch_size, head size, feature_size)
        # x[1]: neighbour vector (batch_size, head size, neighobours, feature_size)

        # Mean-pooling of neighbours:
        # Pass neighbour features through a dense layer with self.hidden_act activations
        xw_neigh = self.hidden_act(K.dot(x[1], self.w_pool) + self.b_pool)
        # Aggregate over neighbour activations using mean
        neigh_agg = K.mean(xw_neigh, axis=2)

        # Apply separate self & neighbour weights and concatenate
        from_self = K.dot(x[0], self.w_self)
        from_neigh = K.dot(neigh_agg, self.w_neigh)
        h_out = K.concatenate([from_self, from_neigh], axis=2)

        if self.has_bias:
            h_out = self.act(h_out + self.bias)
        else:
            h_out = self.act(h_out)

        return h_out

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.
        Assumes that the layer will be built to match that input shape provided.

        Args:
            input_shape (tuple of ints)
                Shape tuples can include None for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """
        return input_shape[0][0], input_shape[0][1], self.output_dim


class GraphSAGE:
    """
    Implementation of the GraphSAGE algorithm with Keras layers.

    Args:
        layer_sizes (list): Hidden feature dimensions for each layer
        generator (Sequence): A NodeSequence or LinkSequence. If specified the n_samples
            and input_dim will be taken from this object.
        n_samples (list): (Optional: needs to be specified if no mapper
            is provided.) The number of samples per layer in the model.
        input_dim (int): The dimensions of the node features used as input to the model.
        aggregator (class): The GraphSAGE aggregator to use. Defaults to the `MeanAggregator`.
        bias (bool): If True a bias vector is learnt for each layer in the GraphSAGE model
        dropout (float): The dropout supplied to each layer in the GraphSAGE model.
        normalize (str): The normalization used after each layer, defaults to L2 normalization.

    """

    def __init__(
        self,
        layer_sizes,
        generator=None,
        n_samples=None,
        input_dim=None,
        aggregator=None,
        bias=True,
        dropout=0.0,
        normalize="l2",
    ):
        # Set the aggregator layer used in the model
        if aggregator is None:
            self._aggregator = MeanAggregator
        elif issubclass(aggregator, Layer):
            self._aggregator = aggregator
        else:
            raise TypeError("Aggregator should be a subclass of Keras Layer")

        # Set the normalization layer used in the model
        if normalize == "l2":
            self._normalization = Lambda(lambda x: K.l2_normalize(x, axis=2))

        elif normalize is None or normalize == "none":
            self._normalization = Lambda(lambda x: x)

        else:
            raise ValueError("Normalization should be either 'l2' or 'none'")

        # Get the input_dim and num_samples from the mapper if it is given
        # Use both the schema and head node type from the mapper
        # TODO: Refactor the horror of generator.generator.graph...
        if generator is not None:
            self.n_samples = generator.generator.num_samples
            feature_sizes = generator.generator.graph.node_feature_sizes()
            if len(feature_sizes) > 1:
                raise RuntimeError(
                    "GraphSAGE called on graph with more than one node type."
                )

            self.input_feature_size = feature_sizes.popitem()[1]

        elif n_samples is not None and input_dim is not None:
            self.n_samples = n_samples
            self.input_feature_size = input_dim

        else:
            raise RuntimeError(
                "If mapper is not provided, n_samples and input_dim must be specified."
            )

        # Model parameters
        self.n_layers = len(self.n_samples)
        self.bias = bias
        self.dropout = dropout

        # Feature dimensions for each layer
        self.dims = [self.input_feature_size] + layer_sizes

        # Aggregator functions for each layer
        self._aggs = [
            self._aggregator(
                output_dim=self.dims[layer + 1],
                bias=self.bias,
                act="relu" if layer < self.n_layers - 1 else "linear",
            )
            for layer in range(self.n_layers)
        ]

        # Sizes of the neighbours for each layer
        self._neigh_reshape = [
            [
                Reshape((-1, max(1, self.n_samples[i]), self.dims[layer]))
                for i in range(self.n_layers - layer)
            ]
            for layer in range(self.n_layers)
        ]

    def __call__(self, x: List):
        """
        Apply aggregator layers

        Args:
            x (list of Tensor): Batch input features

        Returns:
            Output tensor
        """

        def compose_layers(_x: List, layer: int):
            """
            Function to recursively compose aggregation layers. When current layer is at final layer, then length of _x
            should be 1, and compose_layers(_x, layer) returns _x[0].

            Args:
                _x:       List of feature matrix tensors
                layer:   Current layer index

            Returns:
                _x computed from current layer to output layer
            """

            def x_next(agg):
                """
                Compute the list of tensors for the next layer

                Args:
                    agg (Layer): Aggregator layer to apply

                Returns:
                    Output list of tensors of applying the aggregator to inputs

                """
                return [
                    agg(
                        [
                            Dropout(self.dropout)(_x[i]),
                            Dropout(self.dropout)(
                                self._neigh_reshape[layer][i](_x[i + 1])
                            ),
                        ]
                    )
                    for i in range(self.n_layers - layer)
                ]

            return (
                compose_layers(x_next(self._aggs[layer]), layer + 1)
                if layer < self.n_layers
                else _x[0]
            )

        assert isinstance(x, list), "Input features must be a list"
        assert (
            len(x) == self.n_layers + 1 > 1
        ), "Length of input features should match the number of GraphSAGE layers"

        return self._normalization(compose_layers(x, 0))

    def _input_shapes(self) -> List[Tuple[int, int]]:
        """
        Returns the input shapes for the tensors at each layer

        Returns:
            A list of tuples giving the shape (number of nodes, feature size) for
            the corresponding layer

        """

        def shape_at(i: int) -> Tuple[int, int]:
            return (
                max(1, np.product(self.n_samples[:i], dtype=int)),
                self.input_feature_size,
            )

        input_shapes = [shape_at(i) for i in range(self.n_layers + 1)]
        return input_shapes

    def default_model(self, flatten_output=False):
        """
        Return model with default inputs

        Args:
            flatten_output: The GraphSAGE model will return an output tensor
                of form (batch_size, 1, feature_size). If this flag
                is true, the output will be of size
                (batch_size, 1*feature_size)

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors
            for the specified GraphSAGE model and ``x_out`` is tne Keras tensor
            for the GraphSAGE model output.

        """
        # Create tensor inputs
        x_inp = [Input(shape=s) for s in self._input_shapes()]

        # Output from GraphSAGE model
        x_out = self(x_inp)

        if flatten_output:
            x_out = Reshape((-1,))(x_out)

        return x_inp, x_out
