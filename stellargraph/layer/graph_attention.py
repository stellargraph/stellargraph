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

# TODO:  check license?

"""
Definition of Graph Attention Network (GAT) layer and GAT class that is a stack of GAT layers
"""
__all__ = ["GraphAttention", "GAT"]

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Input, Layer, Dropout, LeakyReLU, Lambda, Reshape
import numpy as np
import tensorflow as tf
import warnings
warnings.simplefilter('default')

class GraphAttention(Layer):
    """
    GAT layer, base implementation taken from https://github.com/danielegrattarola/keras-gat
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
        """

        Args:
            F_out: dimensionality of output feature vectors
            attn_heads: number of attention heads
            attn_heads_reduction: reduction applied to output features of each attention head, 'concat' or 'average'.
                'Average' should be applied in the final prediction layer of the model (Eq. 6 of the paper).
            in_dropout_rate: dropout rate applied to features
            attn_dropout_rate: dropout rate applied to attention coefficients
            activation: nonlinear activation applied to layer's output to obtain output features (eq. 4 of the GAT paper)
            use_bias: toggles an optional bias
            kernel_initializer (str): name of layer bias f the initializer for kernel parameters (weights)
            bias_initializer (str): name of the initializer for bias
            attn_kernel_initializer (str): name of the initializer for attention kernel
            kernel_regularizer (str): name of regularizer to be applied to layer kernel. Must be a Keras regularizer.
            bias_regularizer (str): name of regularizer to be applied to layer bias. Must be a Keras regularizer.
            attn_kernel_regularizer (str): name of regularizer to be applied to attention kernel. Must be a Keras regularizer.
            activity_regularizer (str): not used in the current implementation
            kernel_constraint (str): constraint applied to layer's kernel
            bias_constraint (str): constraint applied to layer's bias
            attn_kernel_constraint (str): constraint applied to attention kernel
            **kwargs:
        """

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
            "output_dim": self.output_dim,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def build(self, input_shape):
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

        Args:
            inputs (list): list of inputs with 2 items: node features (matrix of size N x F),
                and graph adjacency matrix (size N x N), where N is the number of nodes in the graph,
                F is the dimensionality of node features
            **kwargs:

        Returns:

        """
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)
        # Convert A to dense tensor - needed for the mask to work
        if K.is_sparse(A):
            A = tf.sparse_tensor_to_dense(A, validate_indices=False)

        # For the GAT model to match that in the paper, we need to ensure that the graph has self-loops,
        # since the neighbourhood of node i in eq. (4) includes node i itself.
        # Adding self-loops to A via setting the diagonal elements of A to 1.0:
        # N = kwargs.get("num_nodes")
        # get the number of nodes from inputs[1] directly, rather than passing it via kwargs
        N = inputs[1]._keras_shape[-1]
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
    A stack of GAT layers with aggregation of multiple attention heads, Eqs 5-6 of GAT paper
    """

    def __init__(
        self,
        layer_sizes,
        attn_heads=1,
        attn_heads_reduction=None,
        activations=None,
        bias=True,
        in_dropout=0.0,
        attn_dropout=0.0,
        normalize="l2",
        generator=None,
    ):
        """

        Args:
            layer_sizes: list of output sizes of GAT layers in the stack
            attn_heads: number of attention heads
            attn_heads_reduction:
            activations: list of activations applied to each layer's output
            bias: toggles an optional bias in GAT layers
            in_dropout: dropout rate applied to input features of each GAT layer
            attn_dropout: dropout rate applied to attention maps
            normalize: normalization applied to the final output features of the GAT layers stack
            generator: an instance of FullBatchNodeGenerator class constructed on the graph of interest
        """
        self._gat_layer = GraphAttention
        self.attn_heads = attn_heads
        self.bias = bias
        self.in_dropout = in_dropout
        self.attn_dropout = attn_dropout
        self.generator = generator

        if attn_heads_reduction is None:
            # default head reductions, see eqs 5-6 of the GAT paper
            self.attn_heads_reduction = ["concat"] * (len(layer_sizes) - 1) + [
                "average"
            ]
        else:
            # user-specified head reductions
            self.attn_heads_reduction = attn_heads_reduction

        assert isinstance(
            activations, list
        ), "Activations should be a list; received {} instead".format(type(activations))
        assert len(activations) == len(
            self.attn_heads_reduction
        ), "Length of activations list ({}) should match the number of GAT layers ({})".format(
            len(activations), len(layer_sizes)
        )
        self.activations = activations

        # Set the normalization layer used in the model
        if normalize == "l2":
            self._normalization = Lambda(lambda x: K.l2_normalize(x, axis=1))

        elif (
            normalize is None
            or normalize == "none"
            or normalize == "None"
            or normalize == "linear"
        ):
            self._normalization = Lambda(lambda x: x)

        else:
            raise ValueError(
                "Normalization should be either 'l2' or 'none'; received '{}'".format(
                    normalize
                )
            )

        # Initialize a stack of GAT layers
        self._layers = []
        for l, F_out in enumerate(layer_sizes):
            # number of attention heads for layer l:
            attn_heads = self.attn_heads if l < len(layer_sizes) - 1 else 1
            # Dropout on input node features before each GAT layer
            self._layers.append(Dropout(self.in_dropout))
            # GAT layer
            self._layers.append(
                self._gat_layer(
                    F_out=F_out,
                    attn_heads=attn_heads,
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
                x = layer([x, A], num_nodes=kwargs.get("num_nodes"))
            else:  # layer is a Dropout layer
                x = layer(x)

        return self._normalization(x)

    def node_model(self, num_nodes=None, feature_size=None):
        """
        Builds a GAT model for node prediction

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a Keras input tensor
            for the specified GAT model and ``x_out`` is a Keras tensor for the GAT model output.

        """
        # Create input tensor:
        if self.generator is not None:
            try:
                N = self.generator.Aadj.shape[0]
            except:
                if num_nodes is not None:
                    N = num_nodes
                else:
                    raise RuntimeError(
                        "node_model: unable to get number of nodes from either generator or the num_nodes argument; stopping."
                    )
            assert self.generator.features.shape[0] == N
            try:
                F = self.generator.features.shape[1]
            except:
                if feature_size is not None:
                    F = feature_size
                else:
                    raise RuntimeError(
                        "node_model: unable to get input feature size from either generator or the feature_size argument; stopping."
                    )
        elif num_nodes is not None and feature_size is not None:
            N = num_nodes
            F = feature_size
        else:
            raise RuntimeError(
                "node_model: if generator is not provided to object constructor, num_nodes and feature_size must be specified."
            )

        X_in = Input(shape=(F,))
        # sparse=True makes model.fit_generator() method work:
        A_in = Input(shape=(N,), sparse=True)

        x_inp = [X_in, A_in]

        # Output from GAT model, N x F', where F' is the output size of the last GAT layer in the stack
        x_out = self(x_inp)

        return x_inp, x_out

    def link_model(self, flatten_output=False):
        """
        Builds a GAT model for link (node pair) prediction
        Args:
            flatten_output:

        Returns:

        """
        raise NotImplemented

    def default_model(self, flatten_output=False):
        warnings.warn(
            "The .default_model() method will be deprecated soon. "
            "Please use .node_model() or .link_model() methods instead.",
            PendingDeprecationWarning
        )
        return self.node_model()
