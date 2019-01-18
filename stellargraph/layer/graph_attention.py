# TODO: license?

"""
Definition of Graph Attention Network (GAT) layer and GAT class that is a stack of GAT layers
"""
__all__ = ["GraphAttention", "GAT"]

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Input, Layer, Dropout, LeakyReLU, Lambda, Reshape
import numpy as np
import tensorflow as tf


class GraphAttention(Layer):
    """
    GAT layer, implementation taken from https://github.com/danielegrattarola/keras-gat
    """

    def __init__(
        self,
        F_,
        attn_heads=1,
        attn_heads_reduction="concat",  # {'concat', 'average'}
        dropout_rate=0.5,
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
            F_: dimensionality of output feature vectors
            attn_heads: number of attention heads
            attn_heads_reduction: reduction applied to output features of each attention head, 'concat' or 'average'.
                'Average' should be applied in the final prediction layer of the model (Eq. 6 of the paper).
            dropout_rate: dropout rate applied to both features and attention coefficients
            activation: nonlinear activation applied to layer's output to obtain output features (eq. 4 of the GAT paper)
            use_bias: toggles an optional bias
            kernel_initializer (str): name of the initializer for kernel parameters (weights)
            bias_initializer (str): name of the initializer for bias
            attn_kernel_initializer (str): name of the initializer for attention kernel
            kernel_regularizer:
            bias_regularizer:
            attn_kernel_regularizer:
            activity_regularizer:
            kernel_constraint:
            bias_constraint:
            attn_kernel_constraint:
            **kwargs:
        """

        if attn_heads_reduction not in {"concat", "average"}:
            raise ValueError("Possbile reduction methods: concat, average")

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
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
        self.supports_masking = False

        # Populated by build()
        self.kernels = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == "concat":
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = int(input_shape[0][-1])

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(
                shape=(F, self.F_),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name="kernel_{}".format(head),
            )
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(
                    shape=(self.F_,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name="bias_{}".format(head),
                )
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(
                shape=(self.F_, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
                name="attn_kernel_self_{}".format(head),
            )
            attn_kernel_neighs = self.add_weight(
                shape=(self.F_, 1),
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

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])  # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + K.transpose(attn_for_neighs)  # (N x N) via broadcasting

            # Add nonlinearity
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            # YT: this only works for 'binary' A, not for 'weighted' A!
            mask = -10e9 * (1.0 - A)
            dense += mask

            # Apply softmax to get attention coefficients
            dense = K.softmax(dense)  # (N x N), Eq. 3 of the paper

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')

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
        dropout=0.,
        normalize="l2",
        generator=None,
    ):
        self._gat_layer = GraphAttention
        self.attn_heads = attn_heads
        self.bias = bias
        self.dropout = dropout
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
        for l, F_ in enumerate(layer_sizes):
            # number of attention heads for layer l:
            attn_heads = self.attn_heads if l < len(layer_sizes) - 1 else 1
            self._layers.append(Dropout(self.dropout))
            self._layers.append(
                self._gat_layer(
                    F_=F_,
                    attn_heads=attn_heads,
                    attn_heads_reduction=self.attn_heads_reduction[l],
                    dropout_rate=self.dropout,
                    activation=self.activations[l],
                    use_bias=self.bias,
                )
            )

    def __call__(self, x_inp):
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
                x = layer([x, A])
            else: # layer is a Dropout layer
                x = layer(x)

        return self._normalization(x)

    def node_model(self):
        """
        Builds a GAT model for node prediction

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a Keras input tensor
            for the specified GAT model and ``x_out`` is a Keras tensor for the GAT model output.

        """
        # Create input tensor:
        N = self.generator.A.shape[0]
        assert self.generator.features.shape[0] == N
        F = self.generator.features.shape[1]

        X_in = Input(shape=(F,))
        A_in = Input(shape=(N,), sparse=True)  # , sparse=True) makes model.fit_generator() method work
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
