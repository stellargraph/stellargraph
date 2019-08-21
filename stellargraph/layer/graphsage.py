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
    "AttentionalAggregator",
    "DirectedGraphSAGE",
]

import numpy as np
from keras.engine.topology import Layer
from keras import Input
from keras import backend as K
from keras.layers import Lambda, Dropout, Reshape, LeakyReLU
from keras.utils import Sequence
from keras import activations
from typing import List, Tuple, Callable, AnyStr
import warnings


class GraphSAGEAggregator(Layer):
    """
    Base class for GraphSAGE aggregators

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional flag indicating whether (True) or not (False; default)
            a bias term should be included.
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.
        neigh_dim (int): Optional value indicating the maximum number of multi-dimensional
            neighbourhoods; defaults to 1.
    """

    def __init__(
        self,
        output_dim: int = 0,
        bias: bool = False,
        act: Callable or AnyStr = "relu",
        neigh_dim: int = 1,
        **kwargs
    ):
        self.neigh_dim = neigh_dim
        self.output_dim = output_dim
        self.other_output_dim = output_dim // (neigh_dim + 1)
        self.self_output_dim = output_dim - neigh_dim * self.other_output_dim
        self.has_bias = bias
        self.act = activations.get(act)
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

    def weight_output_size(self):
        """
        Calculates the output size, according to
        whether the model is building a MLP and
        the method (concat or sum).

        Returns:
            int: size of the weight outputs.

        """
        if self._build_mlp_only:
            weight_dim = self.output_dim
        else:
            weight_dim = self.self_output_dim

        return weight_dim

    def build(self, input_shape):
        """
        Builds layer

        Args:
            input_shape (list of list of int): Shape of input tensors for self
                and neighbour features

        """
        # Build a MLP model if zero neighbours
        self._build_mlp_only = input_shape[1][2] == 0

        self.w_self = self.add_weight(
            name="w_self",
            shape=(input_shape[0][2], self.weight_output_size()),
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

    def apply_mlp(self, x, **kwargs):
        """
        Create MLP on input self tensor, x

        Args:
          x (List[Tensor]): Tensor giving the node features
                shape: (batch_size, head size, feature_size)

        Returns:
            Keras Tensor representing the aggregated embeddings in the input.

        """
        # Weight maxtrix multiplied by self features
        h_out = K.dot(x, self.w_self)
        # Optionally add bias
        if self.has_bias:
            h_out = h_out + self.bias
        # Finally, apply activation
        return self.act(h_out)

    def aggregate_neighbours(self, x_neigh, neigh_idx: int = 0):
        """
        Override with a method to aggregate tensors over neighbourhood.

        Args:
            x_neigh: The input tensor representing the sampled neighbour nodes.
            neigh_idx: Optional neighbourhood index used for multi-dimensional hops.

        Returns:
            A tensor aggregation of the input nodes features.
        """
        raise NotImplementedError(
            "The GraphSAGEAggregator base class should not be directly instantiated"
        )

    def call(self, x, **kwargs):
        """
        Apply aggregator on input tensors, x

        Args:
          x: Keras Tensor

        Returns:
            Keras Tensor representing the aggregated embeddings in the input.

        """
        # x[0]: self vector (batch_size, head size, feature_size)
        # x[1...n]: optional neighbour vectors (batch_size, head size, neighbours, feature_size)
        x_self = x[0]

        if self._build_mlp_only:
            return self.apply_mlp(x_self, **kwargs)

        # Weight maxtrix multiplied by self features
        from_self = K.dot(x_self, self.w_self)

        # If there are neighbours aggregate over them
        if len(x) > 1:
            sources = [from_self]
            for i in range(1, len(x)):
                sources.append(self.aggregate_neighbours(x[i], neigh_idx=i - 1))
            h_out = K.concatenate(sources, axis=2)
        else:
            h_out = from_self

        # Optionally add bias
        if self.has_bias:
            h_out = h_out + self.bias

        # Finally, apply activation
        return self.act(h_out)

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


class MeanAggregator(GraphSAGEAggregator):
    """
    Mean Aggregator for GraphSAGE implemented with Keras base layer

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional bias
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.

    """

    def build(self, input_shape):
        """
        Builds layer

        Args:
            input_shape (list of list of int): Shape of input tensors for self
                and neighbour features

        """
        super().build(input_shape)

        if self._build_mlp_only:
            self.w_neigh = None

        else:
            in_size = input_shape[1][3]
            out_size = self.other_output_dim
            self.w_neigh = []
            for i in range(self.neigh_dim):
                self.w_neigh.append(
                    self.add_weight(
                        name="w_neigh" + str(i),
                        shape=(in_size, out_size),
                        initializer=self._initializer,
                        trainable=True,
                    )
                )

    def aggregate_neighbours(self, x_neigh, neigh_idx=0):
        return K.dot(K.mean(x_neigh, axis=2), self.w_neigh[neigh_idx])


class MaxPoolingAggregator(GraphSAGEAggregator):
    """
    Max Pooling Aggregator for GraphSAGE implemented with Keras base layer

    Implements the aggregator of Eq. (3) in Hamilton et al. (2017)

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional bias
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: These should be user parameters
        self.hidden_dim = self.output_dim
        self.hidden_act = activations.get("relu")

    def build(self, input_shape):
        """
        Builds layer

        Args:
            input_shape (list of list of int): Shape of input tensors for self
                and neighbour features

        """
        super().build(input_shape)

        if self._build_mlp_only:
            self.w_neigh = None
            self.w_pool = None
            self.b_pool = None

        else:
            self.w_neigh = self.add_weight(
                name="w_neigh",
                shape=(self.hidden_dim, self.weight_output_size()),
                initializer=self._initializer,
                trainable=True,
            )
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

    def aggregate_neighbours(self, x_neigh, **kwargs):
        """
        Aggregates the neighbour tensors by max-pooling of neighbours

        Args:
            x_neigh (Tensor): Neighbour tensor of shape (n_batch, n_head, n_neighbour, n_feat)

        Returns:
            Aggregated neighbour tensor of shape (n_batch, n_head, n_feat)
        """
        # Pass neighbour features through a dense layer with self.w_pool, self.b_pool
        xw_neigh = self.hidden_act(K.dot(x_neigh, self.w_pool) + self.b_pool)

        # Take max of this tensor over neighbour dimension
        neigh_agg = K.max(xw_neigh, axis=2)

        # Final output is a dense layer over the aggregated tensor
        from_neigh = K.dot(neigh_agg, self.w_neigh)
        return from_neigh


class MeanPoolingAggregator(GraphSAGEAggregator):
    """
    Mean Pooling Aggregator for GraphSAGE implemented with Keras base layer

    Implements the aggregator of Eq. (3) in Hamilton et al. (2017), with max pooling replaced with mean pooling

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional bias
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: These should be user parameters
        self.hidden_dim = self.output_dim
        self.hidden_act = activations.get("relu")

    def build(self, input_shape):
        """
        Builds layer

        Args:
            input_shape (list of list of int): Shape of input tensors for self
                and neighbour features

        """
        super().build(input_shape)

        if self._build_mlp_only:
            self.w_neigh = None
            self.w_pool = None
            self.b_pool = None

        else:
            self.w_neigh = self.add_weight(
                name="w_neigh",
                shape=(self.hidden_dim, self.weight_output_size()),
                initializer=self._initializer,
                trainable=True,
            )
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

    def aggregate_neighbours(self, x_neigh, **kwargs):
        """
        Aggregates the neighbour tensors by mean-pooling of neighbours

        Args:
            x_neigh (Tensor): Neighbour tensor of shape (n_batch, n_head, n_neighbour, n_feat)

        Returns:
            Aggregated neighbour tensor of shape (n_batch, n_head, n_feat)
        """
        # Pass neighbour features through a dense layer with self.hidden_act activations
        xw_neigh = self.hidden_act(K.dot(x_neigh, self.w_pool) + self.b_pool)

        # Aggregate over neighbour activations using mean
        neigh_agg = K.mean(xw_neigh, axis=2)

        # Final output is a dense layer over the aggregated tensor
        from_neigh = K.dot(neigh_agg, self.w_neigh)
        return from_neigh


class AttentionalAggregator(GraphSAGEAggregator):
    """
    Attentional Aggregator for GraphSAGE implemented with Keras base layer

    Implements the aggregator of Veličković et al. "Graph Attention Networks" ICLR 2018

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional bias
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # How can we expose these options to the user?
        self.attn_act = LeakyReLU(0.2)

    def weight_output_size(self):
        return self.output_dim

    def build(self, input_shape):
        # Build the full model if non-zero neighbours
        super().build(input_shape)

        self.a_self = self.add_weight(
            name="a_self",
            shape=(self.output_dim, 1),
            initializer=self._initializer,
            trainable=True,
        )

        if self._build_mlp_only:
            self.a_neigh = None
        else:
            self.a_neigh = self.add_weight(
                name="a_neigh",
                shape=(self.output_dim, 1),
                initializer=self._initializer,
                trainable=True,
            )

    def call(self, x, **kwargs):
        """
        Apply aggregator on input tensors, x

        Args:
          x (List[Tensor]): Tensors giving self and neighbour features
                x[0]: self Tensor (batch_size, head size, feature_size)
                x[1]: neighbour Tensor (batch_size, head size, neighbours, feature_size)

        Returns:
            Keras Tensor representing the aggregated embeddings in the input.

        """
        if self._build_mlp_only:
            return self.apply_mlp(x[0], **kwargs)

        # Calculate features for self & neighbours
        xw_self = K.expand_dims(K.dot(x[0], self.w_self), axis=2)
        xw_neigh = K.dot(x[1], self.w_self)

        # Concatenate self vector to neighbour vectors
        # Shape is (n_b, n_h, n_neigh+1, n_feat)
        xw_all = K.concatenate([xw_self, xw_neigh], axis=2)

        # Calculate attention
        attn_self = K.dot(xw_self, self.a_self)  # (n_b, n_h, 1)
        attn_neigh = K.dot(xw_all, self.a_neigh)  # (n_b, n_h, n_neigh+1, 1)

        # Add self and neighbour attn and apply activation
        # Note: This broadcasts to (n_b, n_h, n_neigh + 1, 1)
        attn_u = self.attn_act(attn_self + attn_neigh)

        # Attn coefficients, softmax over the neighbours
        attn = K.softmax(attn_u, axis=2)

        # Multiply attn coefficients by neighbours (and self) and aggregate
        h_out = K.sum(attn * xw_all, axis=2)

        if self.has_bias:
            h_out = self.act(h_out + self.bias)
        else:
            h_out = self.act(h_out)

        return h_out


class GraphSAGE:
    """
    Implementation of the GraphSAGE algorithm of Hamilton et al. with Keras layers.
    see: http://snap.stanford.edu/graphsage/

    The model minimally requires specification of the layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer and a generator object.

    Different neighbour node aggregators can also be specified with the ``aggregator``
    argument, which should be the aggregator class,
    either :class:`MeanAggregator`, :class:`MeanPoolingAggregator`,
    :class:`MaxPoolingAggregator`, or :class:`AttentionalAggregator`.

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
        normalize (str or None): The normalization used after each layer, defaults to L2 normalization.

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
            self._normalization = Lambda(lambda x: K.l2_normalize(x, axis=-1))

        elif normalize is None or normalize == "none" or normalize == "None":
            self._normalization = Lambda(lambda x: x)

        else:
            raise ValueError(
                "Normalization should be either 'l2' or 'none'; received '{}'".format(
                    normalize
                )
            )

        # Get the input_dim and num_samples from the generator if it is given
        # Use both the schema and head node type from the generator
        # TODO: Refactor the horror of generator.generator.graph...
        self.generator = generator
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
                "If generator is not provided, n_samples and input_dim must be specified."
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

    def __call__(self, xin: List):
        """
        Apply aggregator layers

        Args:
            x (list of Tensor): Batch input features

        Returns:
            Output tensor
        """

        def apply_layer(x: List, layer: int):
            """
            Compute the list of output tensors for a single GraphSAGE layer

            Args:
                x (List[Tensor]): Inputs to the layer
                layer (int): Layer index to construct

            Returns:
                Outputs of applying the aggregators as a list of Tensors

            """
            layer_out = []
            for i in range(self.n_layers - layer):
                head_shape = K.int_shape(x[i])[1]

                # Reshape neighbours per node per layer
                neigh_in = Dropout(self.dropout)(
                    Reshape((head_shape, self.n_samples[i], self.dims[layer]))(x[i + 1])
                )

                # Apply aggregator to head node and neighbour nodes
                layer_out.append(
                    self._aggs[layer]([Dropout(self.dropout)(x[i]), neigh_in])
                )

            return layer_out

        if not isinstance(xin, list):
            raise TypeError("Input features to GraphSAGE must be a list")

        if len(xin) != self.n_layers + 1:
            raise ValueError(
                "Length of input features should equal the number of GraphSAGE layers plus one"
            )

        # Form GraphSAGE layers iteratively
        self.layer_tensors = []
        h_layer = xin
        for layer in range(0, self.n_layers):
            h_layer = apply_layer(h_layer, layer)
            self.layer_tensors.append(h_layer)

        # Remove neighbourhood dimension from output tensors of the stack
        # note that at this point h_layer contains the output tensor of the top (last applied) layer of the stack
        h_layer = [
            Reshape(K.int_shape(x)[2:])(x) for x in h_layer if K.int_shape(x)[1] == 1
        ]

        return (
            self._normalization(h_layer[0])
            if len(h_layer) == 1
            else [self._normalization(xi) for xi in h_layer]
        )

    def _input_shapes(self) -> List[Tuple[int, int]]:
        """
        Returns the input shapes for the tensors at each layer

        Returns:
            A list of tuples giving the shape (number of nodes, feature size) for
            the corresponding layer

        """

        def shape_at(i: int) -> Tuple[int, int]:
            return (np.product(self.n_samples[:i], dtype=int), self.input_feature_size)

        input_shapes = [shape_at(i) for i in range(self.n_layers + 1)]
        return input_shapes

    def node_model(self):
        """
        Builds a GraphSAGE model for node prediction

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors
            for the specified GraphSAGE model and ``x_out`` is the Keras tensor
            for the GraphSAGE model output.

        """
        # Create tensor inputs
        x_inp = [Input(shape=s) for s in self._input_shapes()]

        # Output from GraphSAGE model
        x_out = self(x_inp)

        return x_inp, x_out

    def link_model(self):
        """
        Builds a GraphSAGE model for link or node pair prediction

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors for (src, dst) node pairs
            (where (src, dst) node inputs alternate),
            and ``x_out`` is a list of output tensors for (src, dst) nodes in the node pairs

        """
        # Expose input and output sockets of the model, for source and destination nodes:
        x_inp_src, x_out_src = self.node_model()
        x_inp_dst, x_out_dst = self.node_model()
        # re-pack into a list where (source, target) inputs alternate, for link inputs:
        x_inp = [x for ab in zip(x_inp_src, x_inp_dst) for x in ab]
        # same for outputs:
        x_out = [x_out_src, x_out_dst]
        return x_inp, x_out

    def build(self):
        """
        Builds a GraphSAGE model for node or link/node pair prediction, depending on the generator used to construct
        the model (whether it is a node or link/node pair generator).

        Returns:
            tuple: (x_inp, x_out), where ``x_inp`` is a list of Keras input tensors
            for the specified GraphSAGE model (either node or link/node pair model) and ``x_out`` contains
            model output tensor(s) of shape (batch_size, layer_sizes[-1])

        """
        if self.generator is not None and hasattr(self.generator, "_sampling_schema"):
            if len(self.generator._sampling_schema) == 1:
                return self.node_model()
            elif len(self.generator._sampling_schema) == 2:
                return self.link_model()
            else:
                raise RuntimeError(
                    "The generator used for model creation is neither a node nor a link generator, "
                    "unable to figure out how to build the model. Consider using node_model or "
                    "link_model method explicitly to build node or link prediction model, respectively."
                )
        else:
            raise RuntimeError(
                "Suitable generator is not provided at model creation time, unable to figure out how to build the model. "
                "Consider either providing a generator, or using node_model or link_model method explicitly to build node or "
                "link prediction model, respectively."
            )

    def default_model(self, flatten_output=True):
        warnings.warn(
            "The .default_model() method will be deprecated in future versions. "
            "Please use .build() method instead.",
            PendingDeprecationWarning,
        )
        return self.build()


class DirectedGraphSAGE:
    """
    Implementation of a directed version of the GraphSAGE algorithm of Hamilton et al. with Keras layers.
    see: http://snap.stanford.edu/graphsage/

    The model minimally requires specification of the layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer and a generator object.

    Different neighbour node aggregators can also be specified with the ``aggregator``
    argument, which should be the aggregator class,
    either :class:`MeanAggregator`, :class:`MeanPoolingAggregator`,
    :class:`MaxPoolingAggregator`, or :class:`AttentionalAggregator`.

    Args:
        layer_sizes (list): Hidden feature dimensions for each layer

        Either:
            generator (Sequence): A NodeSequence or LinkSequence.
        Or:
            in_samples (list): The number of in-node samples per layer in the model.
            out_samples (list): The number of out-node samples per layer in the model.
            input_dim (int): The dimensions of the node features used as input to the model.
        aggregator (class): The GraphSAGE aggregator to use. Defaults to the `MeanAggregator`.
        bias (bool): If True a bias vector is learnt for each layer in the GraphSAGE model
        dropout (float): The dropout supplied to each layer in the GraphSAGE model.
        normalize (str or None): The normalization used after each layer, defaults to L2 normalization.
    """

    def __init__(
        self,
        layer_sizes,
        generator=None,
        in_samples=None,
        out_samples=None,
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
            self._normalization = Lambda(lambda x: K.l2_normalize(x, axis=-1))
        elif normalize is None or normalize == "none" or normalize == "None":
            self._normalization = Lambda(lambda x: x)
        else:
            raise ValueError(
                "Normalization should be either 'l2' or 'none'; received '{}'".format(
                    normalize
                )
            )

        # Get the input_dim and num_samples from the generator if it is given
        # Use both the schema and head node type from the generator
        # TODO: Refactor the horror of generator.generator.graph...
        self.generator = generator
        if generator is not None:
            self.in_samples = generator.generator.in_samples
            self.out_samples = generator.generator.out_samples
            feature_sizes = generator.generator.graph.node_feature_sizes()
            if len(feature_sizes) > 1:
                raise RuntimeError(
                    "GraphSAGE called on graph with more than one node type."
                )
            self.input_feature_size = feature_sizes.popitem()[1]

        elif (
            in_samples is not None and out_samples is not None and input_dim is not None
        ):
            self.in_samples = in_samples
            self.out_samples = out_samples
            self.input_feature_size = input_dim

        else:
            raise RuntimeError(
                "If generator is not provided, n_samples and input_dim must be specified."
            )

        self.max_hops = max_hops = len(layer_sizes)
        if len(self.in_samples) != max_hops:
            raise ValueError(
                "Mismatched lengths: in-node sample sizes {} versus layer sizes {}".format(
                    self.in_samples, layer_sizes
                )
            )
        if len(self.out_samples) != max_hops:
            raise ValueError(
                "Mismatched lengths: out-node sample sizes {} versus layer sizes {}".format(
                    self.out_samples, layer_sizes
                )
            )

        # Model parameters
        self.max_slots = 2 ** (max_hops + 1) - 1
        self.bias = bias
        self.dropout = dropout

        # Feature dimensions for each layer
        self.layer_sizes = layer_sizes
        self.dims = [self.input_feature_size] + layer_sizes

        # Aggregator functions for each layer
        self._aggs = [
            self._aggregator(
                output_dim=layer_sizes[i],
                bias=self.bias,
                act="relu" if i < max_hops - 1 else "linear",
                neigh_dim=2,
            )
            for i in range(max_hops)
        ]

    def __call__(self, xin: List):
        """
        Apply aggregator layers

        Args:
            x (list of Tensor): Batch input features

        Returns:
            Output tensor
        """

        def aggregate_neighbours(tree: List, stage: int):
            # compute the number of slots with children in the binary tree
            num_slots = (len(tree) - 1) // 2
            new_tree = [None] * num_slots
            for slot in range(num_slots):
                # get parent nodes
                num_head_nodes = K.int_shape(tree[slot])[1]
                parent = Dropout(self.dropout)(tree[slot])
                # find in-nodes
                child_slot = 2 * slot + 1
                size = (
                    self.neighbourhood_sizes[child_slot] // num_head_nodes
                    if num_head_nodes > 0
                    else 0
                )
                in_child = Dropout(self.dropout)(
                    Reshape((num_head_nodes, size, self.dims[stage]))(tree[child_slot])
                )
                # find out-nodes
                child_slot = child_slot + 1
                size = (
                    self.neighbourhood_sizes[child_slot] // num_head_nodes
                    if num_head_nodes > 0
                    else 0
                )
                out_child = Dropout(self.dropout)(
                    Reshape((num_head_nodes, size, self.dims[stage]))(tree[child_slot])
                )
                # aggregate neighbourhoods
                new_tree[slot] = self._aggs[stage]([parent, in_child, out_child])
            return new_tree

        if not isinstance(xin, list):
            raise TypeError("Input features to GraphSAGE must be a list")

        if len(xin) != self.max_slots:
            raise ValueError(
                "Number of input tensors does not match number of GraphSAGE layers"
            )

        # Combine GraphSAGE layers in stages
        stage_tree = xin
        for stage in range(self.max_hops):
            stage_tree = aggregate_neighbours(stage_tree, stage)
        out_layer = stage_tree[0]

        # Remove neighbourhood dimension from output tensors of the stack
        out_layer = Reshape(K.int_shape(out_layer)[2:])(out_layer)
        return self._normalization(out_layer)

    def _compute_input_sizes(self) -> List[int]:
        # Each hop has to sample separately from both the in-nodes
        # and the out-nodes. This gives rise to a binary tree of 'slots'.
        # Storage for the total (cumulative product) number of nodes sampled
        # at the corresponding neighbourhood for each slot:
        num_nodes = [0] * self.max_slots
        num_nodes[0] = 1
        # Storage for the number of hops to reach
        # the corresponding neighbourhood for each slot:
        num_hops = [0] * self.max_slots
        for slot in range(1, self.max_slots):
            parent_slot = (slot + 1) // 2 - 1
            i = num_hops[parent_slot]
            num_hops[slot] = i + 1
            num_nodes[slot] = (
                self.in_samples[i] if slot % 2 == 1 else self.out_samples[i]
            ) * num_nodes[parent_slot]
        return num_nodes

    def node_model(self):
        """
        Builds a GraphSAGE model for node prediction

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors
            for the specified GraphSAGE model and ``x_out`` is the Keras tensor
            for the GraphSAGE model output.

        """
        # Create tensor inputs for neighbourhood sampling;
        x_inp = [
            Input(shape=(s, self.input_feature_size)) for s in self.neighbourhood_sizes
        ]

        # Output from GraphSAGE model
        x_out = self(x_inp)

        return x_inp, x_out

    def link_model(self):
        """
        Builds a GraphSAGE model for link or node pair prediction

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors for (src, dst) node pairs
            (where (src, dst) node inputs alternate),
            and ``x_out`` is a list of output tensors for (src, dst) nodes in the node pairs

        """
        # Expose input and output sockets of the model, for source and destination nodes:
        x_inp_src, x_out_src = self.node_model()
        x_inp_dst, x_out_dst = self.node_model()
        # re-pack into a list where (source, target) inputs alternate, for link inputs:
        x_inp = [x for ab in zip(x_inp_src, x_inp_dst) for x in ab]
        # same for outputs:
        x_out = [x_out_src, x_out_dst]
        return x_inp, x_out

    def build(self):
        """
        Builds a GraphSAGE model for node or link/node pair prediction, depending on the generator used to construct
        the model (whether it is a node or link/node pair generator).

        Returns:
            tuple: (x_inp, x_out), where ``x_inp`` is a list of Keras input tensors
            for the specified GraphSAGE model (either node or link/node pair model) and ``x_out`` contains
            model output tensor(s) of shape (batch_size, layer_sizes[-1])

        """
        self.neighbourhood_sizes = self._compute_input_sizes()

        if self.generator is not None and hasattr(self.generator, "_sampling_schema"):
            if len(self.generator._sampling_schema) == 1:
                return self.node_model()
            elif len(self.generator._sampling_schema) == 2:
                return self.link_model()
            else:
                raise RuntimeError(
                    "The generator used for model creation is neither a node nor a link generator, "
                    "unable to figure out how to build the model. Consider using node_model or "
                    "link_model method explicitly to build node or link prediction model, respectively."
                )
        else:
            raise RuntimeError(
                "Suitable generator is not provided at model creation time, unable to figure out how to build the model. "
                "Consider either providing a generator, or using node_model or link_model method explicitly to build node or "
                "link prediction model, respectively."
            )

    def default_model(self, flatten_output=True):
        warnings.warn(
            "The .default_model() method will be deprecated in future versions. "
            "Please use .build() method instead.",
            PendingDeprecationWarning,
        )
        return self.build()
