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

"""
Definition of Graph Attention Network (GAT) layer, and GAT class that is a stack of GAT layers
"""
__all__ = ["GraphAttention", "GAT"]

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Input, Layer, Dropout, LeakyReLU, Lambda, Reshape
import numpy as np
import tensorflow as tf
from stellargraph.mapper import FullBatchNodeGenerator
import warnings

warnings.simplefilter("default")


class GraphAttention(Layer):
    """
    Graph Attention (GAT) layer, base implementation taken from https://github.com/danielegrattarola/keras-gat,
    some modifications added for ease of use.

    Based on the original paper: Graph Attention Networks. P. Velickovic et al. ICLR 2018 https://arxiv.org/abs/1803.07294

    Args:
            F_out (int): dimensionality of output feature vectors
            attn_heads (int or list of int): number of attention heads
            attn_heads_reduction (str): reduction applied to output features of each attention head, 'concat' or 'average'.
                'Average' should be applied in the final prediction layer of the model (Eq. 6 of the paper).
            in_dropout_rate (float): dropout rate applied to features
            attn_dropout_rate (float): dropout rate applied to attention coefficients
            activation (str): nonlinear activation applied to layer's output to obtain output features (eq. 4 of the GAT paper)
            use_bias (bool): toggles an optional bias
            kernel_initializer (str): name of layer bias f the initializer for kernel parameters (weights)
            bias_initializer (str): name of the initializer for bias
            attn_kernel_initializer (str): name of the initializer for attention kernel
            kernel_regularizer (str): name of regularizer to be applied to layer kernel. Must be a Keras regularizer.
            bias_regularizer (str): name of regularizer to be applied to layer bias. Must be a Keras regularizer.
            attn_kernel_regularizer (str): name of regularizer to be applied to attention kernel. Must be a Keras regularizer.
            activity_regularizer (str): not used in the current implementation
            kernel_constraint (str): constraint applied to layer's kernel. Must be a Keras constraint https://keras.io/constraints/
            bias_constraint (str): constraint applied to layer's bias. Must be a Keras constraint https://keras.io/constraints/
            attn_kernel_constraint (str): constraint applied to attention kernel. Must be a Keras constraint https://keras.io/constraints/
            **kwargs: optional keyword arguments

    """

    def __init__(
        self,
        F_out,
        attn_heads=1,
        attn_heads_reduction="concat",  # {'concat', 'average'}
        in_dropout_rate=0.0,
        attn_dropout_rate=0.0,
        activation="relu",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        attn_kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_kernel_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        attn_kernel_constraint=None,
        **kwargs
    ):

        if attn_heads_reduction not in {"concat", "average"}:
            raise ValueError(
                "{}: Possible heads reduction methods: concat, average; received {}".format(
                    type(self).__name__, attn_heads_reduction
                )
            )

        self.F_out = F_out  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.in_dropout_rate = in_dropout_rate  # dropout rate for node features
        self.attn_dropout_rate = attn_dropout_rate  # dropout rate for attention coefs
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

        # Populated by build()
        self.kernels = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == "concat":
            # Output will have shape (..., K * F')
            self.output_dim = self.F_out * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_out

        super(GraphAttention, self).__init__(**kwargs)

    def get_config(self):
        """
        Gets class configuration for Keras serialization

        """
        config = {
            "F_out": self.F_out,
            "attn_heads": self.attn_heads,
            "attn_heads_reduction": self.attn_heads_reduction,
            "in_dropout_rate": self.in_dropout_rate,
            "attn_dropout_rate": self.attn_dropout_rate,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "attn_kernel_initializer": initializers.serialize(
                self.attn_kernel_initializer
            ),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "attn_kernel_regularizer": regularizers.serialize(
                self.attn_kernel_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "attn_kernel_constraint": constraints.serialize(
                self.attn_kernel_constraint
            ),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        """
        Builds the layer

        Args:
            input_shape (list of list of int): shapes of the layer's input(s)

        """
        assert len(input_shape) >= 2
        F_in = int(input_shape[0][-1])

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(
                shape=(F_in, self.F_out),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name="kernel_{}".format(head),
            )
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(
                    shape=(self.F_out,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name="bias_{}".format(head),
                )
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(
                shape=(self.F_out, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
                name="attn_kernel_self_{}".format(head),
            )
            attn_kernel_neighs = self.add_weight(
                shape=(self.F_out, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
                name="attn_kernel_neigh_{}".format(head),
            )
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Applies the layer.

        Args:
            inputs (list): list of inputs with 2 items: node features (matrix of size N x F),
                and graph adjacency matrix (size N x N), where N is the number of nodes in the graph,
                F is the dimensionality of node features

        """
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)
        # Convert A to dense tensor - needed for the mask to work
        # TODO: replace this dense implementation of GraphAttention layer with a sparse implementation
        if K.is_sparse(A):
            A = tf.sparse_tensor_to_dense(A, validate_indices=False)

        # For the GAT model to match that in the paper, we need to ensure that the graph has self-loops,
        # since the neighbourhood of node i in eq. (4) includes node i itself.
        # Adding self-loops to A via setting the diagonal elements of A to 1.0:
        if kwargs.get("add_self_loops", False):
            # get the number of nodes from inputs[1] directly
            N = K.int_shape(inputs[1])[-1]
            if N is not None:
                # create self-loops
                A = tf.linalg.set_diag(A, K.cast(np.ones((N,)), dtype="float"))
            else:
                raise ValueError(
                    "{}: need to know number of nodes to add self-loops; obtained None instead".format(
                        type(self).__name__
                    )
                )

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[
                head
            ]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(
                features, attention_kernel[0]
            )  # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(
                features, attention_kernel[1]
            )  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + K.transpose(
                attn_for_neighs
            )  # (N x N) via broadcasting

            # Add nonlinearity
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            # YT: this only works for 'binary' A, not for 'weighted' A!
            # YT: if A does not have self-loops, the node itself will be masked, so A should have self-loops
            # YT: this is ensured by setting the diagonal elements of A tensor to 1 above
            mask = -10e9 * (1.0 - A)
            dense += mask

            # Apply softmax to get attention coefficients
            dense = K.softmax(dense)  # (N x N), Eq. 3 of the paper

            # Apply dropout to features and attention coefficients
            dropout_feat = Dropout(self.in_dropout_rate)(features)  # (N x F')
            dropout_attn = Dropout(self.attn_dropout_rate)(dense)  # (N x N)

            # Linear combination with neighbors' features [YT: see Eq. 4]
            node_features = K.dot(dropout_attn, dropout_feat)  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == "concat":
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape


class GAT:
    """
    A stack of Graph Attention (GAT) layers with aggregation of multiple attention heads, Eqs 5-6 of the GAT paper https://arxiv.org/abs/1803.07294

    Args:
            layer_sizes (list of int): list of output sizes of GAT layers in the stack. The length of this list defines
                the number of GraphAttention layers in the stack.
            attn_heads (int or list of int): number of attention heads in GraphAttention layers. The options are:

                - a single integer: the passed value of `attn_heads` will be applied to all GraphAttention layers in the stack, except the last layer (for which the number of attn_heads will be set to 1).
                - a list of integers: elements of the list define the number of attention heads in the corresponding layers in the stack.

            attn_heads_reduction (list of str or None): reductions applied to output features of each attention head,
                for all layers in the stack. Valid entries in the list are {'concat', 'average'}.
                If None is passed, the default reductions are applied: 'concat' reduction to all layers in the stack
                except the final layer, 'average' reduction to the last layer (Eqs. 5-6 of the GAT paper).
            activations (list of str): list of activations applied to each layer's output
            bias (bool): toggles an optional bias in GAT layers
            in_dropout (float): dropout rate applied to input features of each GAT layer
            attn_dropout (float): dropout rate applied to attention maps
            normalize (str or None): normalization applied to the final output features of the GAT layers stack
            generator (FullBatchNodeGenerator): an instance of FullBatchNodeGenerator class constructed on the graph of interest
    """

    def __init__(
        self,
        layer_sizes,
        activations,
        attn_heads=1,
        attn_heads_reduction=None,
        bias=True,
        in_dropout=0.0,
        attn_dropout=0.0,
        normalize="l2",
        generator=None,
    ):
        self._gat_layer = GraphAttention
        self.bias = bias
        self.in_dropout = in_dropout
        self.attn_dropout = attn_dropout
        self.generator = generator

        # Check layer_sizes (must be list of int):
        # check type:
        if not isinstance(layer_sizes, list):
            raise TypeError(
                "{}: layer_sizes should be a list of integers; received type {} instead.".format(
                    type(self).__name__, type(layer_sizes).__name__
                )
            )
        # check that values are valid:
        elif not all([isinstance(s, int) and s > 0 for s in layer_sizes]):
            raise ValueError(
                "{}: all elements in layer_sizes should be positive integers!".format(
                    type(self).__name__
                )
            )

        # Check attn_heads (must be int or list of int):
        if isinstance(attn_heads, list):
            # check the length
            if not len(attn_heads) == len(layer_sizes):
                raise ValueError(
                    "{}: length of attn_heads list ({}) should match the number of GAT layers ({})".format(
                        type(self).__name__, len(attn_heads), len(layer_sizes)
                    )
                )
            # check that values in the list are valid
            if not all([isinstance(a, int) and a > 0 for a in attn_heads]):
                raise ValueError(
                    "{}: all elements in attn_heads should be positive integers!".format(
                        type(self).__name__
                    )
                )

            self.attn_heads = attn_heads  # (list of int as passed by the user)
        elif isinstance(attn_heads, int):
            self.attn_heads = list()
            for l, _ in enumerate(layer_sizes):
                # number of attention heads for layer l: attn_heads (int) for all but the last layer (for which it's set to 1)
                self.attn_heads.append(attn_heads if l < len(layer_sizes) - 1 else 1)
        else:
            raise TypeError(
                "{}: attn_heads should be an integer or a list of integers!".format(
                    type(self).__name__
                )
            )

        # Check attn_heads_reduction (list of str, or None):
        if attn_heads_reduction is None:
            # set default head reductions, see eqs 5-6 of the GAT paper
            self.attn_heads_reduction = ["concat"] * (len(layer_sizes) - 1) + [
                "average"
            ]
        else:
            # user-specified list of head reductions (valid entries are 'concat' and 'average')
            # check type (must be a list of str):
            if not isinstance(attn_heads_reduction, list):
                raise TypeError(
                    "{}: attn_heads_reduction should be a string; received type {} instead.".format(
                        type(self).__name__, type(attn_heads_reduction).__name__
                    )
                )

            # check length of attn_heads_reduction list:
            if not len(attn_heads_reduction) == len(layer_sizes):
                raise ValueError(
                    "{}: length of attn_heads_reduction list ({}) should match the number of GAT layers ({})".format(
                        type(self).__name__, len(attn_heads_reduction), len(layer_sizes)
                    )
                )

            # check that list elements are valid:
            if all(
                [ahr.lower() in {"concat", "average"} for ahr in attn_heads_reduction]
            ):
                self.attn_heads_reduction = attn_heads_reduction
            else:
                raise ValueError(
                    "{}: elements of attn_heads_reduction list should be either 'concat' or 'average'!".format(
                        type(self).__name__
                    )
                )

        # Check activations (list of str):
        # check type:
        if not isinstance(activations, list):
            raise TypeError(
                "{}: activations should be a list of strings; received {} instead".format(
                    type(self).__name__, type(activations)
                )
            )
        # check length:
        if not len(activations) == len(layer_sizes):
            raise ValueError(
                "{}: length of activations list ({}) should match the number of GAT layers ({})".format(
                    type(self).__name__, len(activations), len(layer_sizes)
                )
            )
        self.activations = activations

        # check generator:
        if generator is not None:
            if not isinstance(generator, FullBatchNodeGenerator):
                raise ValueError(
                    "{}: generator must be of type FullBatchNodeGenerator or None; received object of type {} instead".format(
                        type(self).__name__, type(generator).__name__
                    )
                )

        # Set the normalization layer used in the model
        if normalize == "l2":
            self._normalization = Lambda(lambda x: K.l2_normalize(x, axis=1))

        elif normalize is None or str(normalize).lower() in {"none", "linear"}:
            self._normalization = Lambda(lambda x: x)

        else:
            raise ValueError(
                "Normalization should be either 'l2' or None (also allowed as 'none'); received '{}'".format(
                    normalize
                )
            )

        # Initialize a stack of GAT layers
        self._layers = []
        for l, F_out in enumerate(layer_sizes):
            # Dropout on input node features before each GAT layer
            self._layers.append(Dropout(self.in_dropout))
            # GraphAttention layer
            self._layers.append(
                self._gat_layer(
                    F_out=F_out,
                    attn_heads=self.attn_heads[l],
                    attn_heads_reduction=self.attn_heads_reduction[l],
                    in_dropout_rate=self.in_dropout,
                    attn_dropout_rate=self.attn_dropout,
                    activation=self.activations[l],
                    use_bias=self.bias,
                )
            )

    def __call__(self, x_inp, **kwargs):
        """
        Apply a stack of GAT layers to the input x_inp

        Args:
            x_inp (Tensor): input of the 1st GAT layer in the stack

        Returns: Output tensor of the GAT layers stack

        """

        assert isinstance(x_inp, list), "input must be a list, got {} instead".format(
            type(x_inp)
        )

        x = x_inp[0]
        A = x_inp[1]

        for layer in self._layers:
            if isinstance(layer, self._gat_layer):  # layer is a GAT layer
                x = layer([x, A], add_self_loops=kwargs.get("add_self_loops"))
            else:  # layer is a Dropout layer
                x = layer(x)

        return self._normalization(x)

    def node_model(self, num_nodes=None, feature_size=None, add_self_loops=True):
        """
        Builds a GAT model for node prediction

        Args:
            num_nodes (int or None): (optional) number of nodes in the graph (in the full batch). If not provided, this will be taken from self.generator.
            feature_size (int or None): (optional) dimensionality of node attributes. If not provided, this will be taken from self.generator.
            add_self_loops (bool): (default is True) toggles adding self-loops to the graph's adjacency matrix in the GraphAttention layers of the GAT model.

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of two Keras input tensors for the specified GAT model
            (containing node features and graph adjacency matrix), and ``x_out`` is a Keras tensor for the GAT model output.

        """
        # Create input tensor:
        if self.generator is not None:
            N = self.generator.Aadj.shape[0]

            assert self.generator.features.shape[0] == N
            F = self.generator.features.shape[1]
            is_adj_sparse = self.generator.sparse

        elif num_nodes is not None and feature_size is not None:
            N = num_nodes
            F = feature_size
            is_adj_sparse = True
        else:
            raise RuntimeError(
                "node_model: if generator is not provided to object constructor, num_nodes and feature_size must be specified."
            )

        X_in = Input(shape=(F,))
        A_in = Input(shape=(N,), sparse=is_adj_sparse)

        x_inp = [X_in, A_in]

        # Output from GAT model, N x F', where F' is the output size of the last GAT layer in the stack
        x_out = self(x_inp, add_self_loops=add_self_loops)

        return x_inp, x_out

    def link_model(self):
        """
        Builds a GAT model for link (node pair) prediction (implementation pending)

        """
        raise NotImplemented

    def default_model(self, flatten_output=False):
        warnings.warn(
            "The .default_model() method will be deprecated soon. "
            "Please use .node_model() or .link_model() methods instead.",
            PendingDeprecationWarning,
        )
        return self.node_model()
