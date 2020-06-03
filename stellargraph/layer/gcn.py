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

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Layer, Lambda, Dropout, Reshape

from ..mapper import FullBatchGenerator, ClusterNodeGenerator
from .misc import SqueezedSparseConversion, deprecated_model_function, GatherIndices
from .preprocessing_layer import GraphPreProcessingLayer


class GraphConvolution(Layer):

    """
    Graph Convolution (GCN) Keras layer.
    The implementation is based on the keras-gcn github repo https://github.com/tkipf/keras-gcn.

    Original paper: Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling,
    International Conference on Learning Representations (ICLR), 2017 https://github.com/tkipf/gcn

    Notes:
      - The batch axis represents independent graphs to be convolved with this GCN kernel (for
        instance, for full-batch node prediction on a single graph, its dimension should be 1).

      - If the adjancency matrix is dense, both it and the features should have a batch axis, with
        equal batch dimension.

      - If the adjancency matrix is sparse, it should not have a batch axis, and the batch
        dimension of the features must be 1.

      - There are two inputs required, the node features,
        and the normalized graph Laplacian matrix

      - This class assumes that the normalized Laplacian matrix is passed as
        input to the Keras methods.

    Args:
        units (int): dimensionality of output feature vectors
        activation (str or func): nonlinear activation applied to layer's output to obtain output features
        use_bias (bool): toggles an optional bias
        final_layer (bool): Deprecated, use ``tf.gather`` or :class:`GatherIndices`
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
        activation=None,
        use_bias=True,
        final_layer=None,
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
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        if final_layer is not None:
            raise ValueError(
                "'final_layer' is not longer supported, use 'tf.gather' or 'GatherIndices' separately"
            )

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
        feature_shape, *As_shapes = input_shapes

        batch_dim = feature_shape[0]
        out_dim = feature_shape[1]

        return batch_dim, out_dim, self.units

    def build(self, input_shapes):
        """
        Builds the layer

        Args:
            input_shapes (list of int): shapes of the layer's inputs (node features and adjacency matrix)

        """
        feat_shape = input_shapes[0]
        input_dim = int(feat_shape[-1])

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
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

    def call(self, inputs):
        """
        Applies the layer.

        Args:
            inputs (list): a list of 3 input tensors that includes
                node features (size 1 x N x F),
                graph adjacency matrix (size N x N),
                where N is the number of nodes in the graph, and
                F is the dimensionality of node features.

        Returns:
            Keras Tensor that represents the output of the layer.
        """
        features, *As = inputs

        # Calculate the layer operation of GCN
        A = As[0]
        if K.is_sparse(A):
            # FIXME(#1222): batch_dot doesn't support sparse tensors, so we special case them to
            # only work with a single batch element (and the adjacency matrix without a batch
            # dimension)
            if features.shape[0] != 1:
                raise ValueError(
                    f"features: expected batch dimension = 1 when using sparse adjacency matrix in GraphConvolution, found features batch dimension {features.shape[0]}"
                )
            if len(A.shape) != 2:
                raise ValueError(
                    f"adjacency: expected a single adjacency matrix when using sparse adjacency matrix in GraphConvolution (tensor of rank 2), found adjacency tensor of rank {len(A.shape)}"
                )

            features_sq = K.squeeze(features, axis=0)
            h_graph = K.dot(A, features_sq)
            h_graph = K.expand_dims(h_graph, axis=0)
        else:
            h_graph = K.batch_dot(A, features)
        output = K.dot(h_graph, self.kernel)

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias
        output = self.activation(output)

        return output


class GCN:
    """
    A stack of Graph Convolutional layers that implement a graph convolution network model
    as in https://arxiv.org/abs/1609.02907

    The model minimally requires specification of the layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer,
    activation functions for each hidden layers, and a generator object.

    To use this class as a Keras model, the features and pre-processed adjacency matrix
    should be supplied using either the :class:`FullBatchNodeGenerator` class for node inference
    or the :class:`FullBatchLinkGenerator` class for link inference.

    To have the appropriate pre-processing the generator object should be instanciated
    with the `method='gcn'` argument.

    Note that currently the GCN class is compatible with both sparse and dense adjacency
    matrices and the :class:`FullBatchNodeGenerator` will default to sparse.

    For more details, please see `the GCN demo notebook <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/gcn-node-classification.html>`_

    Example:
        Creating a GCN node classification model from an existing :class:`StellarGraph`
        object ``G``::

            generator = FullBatchNodeGenerator(G, method="gcn")
            gcn = GCN(
                    layer_sizes=[32, 4],
                    activations=["elu","softmax"],
                    generator=generator,
                    dropout=0.5
                )
            x_inp, predictions = gcn.in_out_tensors()

    Notes:
      - The inputs are tensors with a batch dimension of 1. These are provided by the \
        :class:`FullBatchNodeGenerator` object.

      - This assumes that the normalized Lapalacian matrix is provided as input to
        Keras methods. When using the :class:`FullBatchNodeGenerator` specify the
        ``method='gcn'`` argument to do this pre-processing.

      - The nodes provided to the :class:`FullBatchNodeGenerator.flow` method are
        used by the final layer to select the predictions for those nodes in order.
        However, the intermediate layers before the final layer order the nodes
        in the same way as the adjacency matrix.

    Args:
        layer_sizes (list of int): Output sizes of GCN layers in the stack.
        generator (FullBatchNodeGenerator): The generator instance.
        bias (bool): If True, a bias vector is learnt for each layer in the GCN model.
        dropout (float): Dropout rate applied to input features of each GCN layer.
        activations (list of str or func): Activations applied to each layer's output;
            defaults to ['relu', ..., 'relu'].
        kernel_initializer (str or func, optional): The initialiser to use for the weights of each layer.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights of each layer.
        kernel_constraint (str or func, optional): The constraint to use for the weights of each layer.
        bias_initializer (str or func, optional): The initialiser to use for the bias of each layer.
        bias_regularizer (str or func, optional): The regulariser to use for the bias of each layer.
        bias_constraint (str or func, optional): The constraint to use for the bias of each layer.
    """

    def __init__(
        self,
        layer_sizes,
        generator,
        bias=True,
        dropout=0.0,
        activations=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
    ):
        if not isinstance(generator, (FullBatchGenerator, ClusterNodeGenerator)):
            raise TypeError(
                f"Generator should be a instance of FullBatchNodeGenerator, "
                f"FullBatchLinkGenerator or ClusterNodeGenerator"
            )

        n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.bias = bias
        self.dropout = dropout

        # Copy required information from generator
        self.method = generator.method
        self.multiplicity = generator.multiplicity
        self.n_features = generator.features.shape[1]
        self.use_sparse = generator.use_sparse
        if isinstance(generator, FullBatchGenerator):
            self.n_nodes = generator.features.shape[0]
        else:
            self.n_nodes = None

        if self.method == "none":
            self.graph_norm_layer = GraphPreProcessingLayer(num_of_nodes=self.n_nodes)

        # Activation function for each layer
        if activations is None:
            activations = ["relu"] * n_layers
        elif len(activations) != n_layers:
            raise ValueError(
                "Invalid number of activations; require one function per layer"
            )
        self.activations = activations

        # Initialize a stack of GCN layers
        self._layers = []
        for ii in range(n_layers):
            self._layers.append(Dropout(self.dropout))
            self._layers.append(
                GraphConvolution(
                    self.layer_sizes[ii],
                    activation=self.activations[ii],
                    use_bias=self.bias,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_initializer=bias_initializer,
                    bias_regularizer=bias_regularizer,
                    bias_constraint=bias_constraint,
                )
            )

    def __call__(self, x):
        """
        Apply a stack of GCN layers to the inputs.
        The input tensors are expected to be a list of the following:
        [
            Node features shape (1, N, F),
            Adjacency indices (1, E, 2),
            Adjacency values (1, E),
            Output indices (1, O)
        ]
        where N is the number of nodes, F the number of input features,
              E is the number of edges, O the number of output nodes.

        Args:
            x (Tensor): input tensors

        Returns:
            Output tensor
        """
        x_in, out_indices, *As = x

        # Currently we require the batch dimension to be one for full-batch methods
        batch_dim, n_nodes, _ = K.int_shape(x_in)
        if batch_dim != 1:
            raise ValueError(
                "Currently full-batch methods only support a batch dimension of one"
            )

        # Convert input indices & values to a sparse matrix
        if self.use_sparse:
            A_indices, A_values = As
            Ainput = [
                SqueezedSparseConversion(
                    shape=(n_nodes, n_nodes), dtype=A_values.dtype
                )([A_indices, A_values])
            ]
        else:
            Ainput = As

        # TODO: Support multiple matrices?
        if len(Ainput) != 1:
            raise NotImplementedError(
                "The GCN method currently only accepts a single matrix"
            )

        h_layer = x_in
        if self.method == "none":
            # For GCN, if no preprocessing has been done, we apply the preprocessing layer to perform that.
            Ainput = [self.graph_norm_layer(Ainput[0])]
        for layer in self._layers:
            if isinstance(layer, GraphConvolution):
                # For a GCN layer add the matrix
                h_layer = layer([h_layer] + Ainput)
            else:
                # For other (non-graph) layers only supply the input tensor
                h_layer = layer(h_layer)

        # only return data for the requested nodes
        h_layer = GatherIndices(batch_dims=1)([h_layer, out_indices])

        return h_layer

    def in_out_tensors(self, multiplicity=None):
        """
        Builds a GCN model for node or link prediction

        Returns:
            tuple: `(x_inp, x_out)`, where `x_inp` is a list of Keras/TensorFlow
            input tensors for the GCN model and `x_out` is a tensor of the GCN model output.
        """
        # Inputs for features
        x_t = Input(batch_shape=(1, self.n_nodes, self.n_features))

        # If not specified use multiplicity from instanciation
        if multiplicity is None:
            multiplicity = self.multiplicity

        # Indices to gather for model output
        if multiplicity == 1:
            out_indices_t = Input(batch_shape=(1, None), dtype="int32")
        else:
            out_indices_t = Input(batch_shape=(1, None, multiplicity), dtype="int32")

        # Create inputs for sparse or dense matrices
        if self.use_sparse:
            # Placeholders for the sparse adjacency matrix
            A_indices_t = Input(batch_shape=(1, None, 2), dtype="int64")
            A_values_t = Input(batch_shape=(1, None))
            A_placeholders = [A_indices_t, A_values_t]

        else:
            # Placeholders for the dense adjacency matrix
            A_m = Input(batch_shape=(1, self.n_nodes, self.n_nodes))
            A_placeholders = [A_m]

        # TODO: Support multiple matrices

        x_inp = [x_t, out_indices_t] + A_placeholders
        x_out = self(x_inp)

        # Flatten output by removing singleton batch dimension
        if x_out.shape[0] == 1:
            self.x_out_flat = Lambda(lambda x: K.squeeze(x, 0))(x_out)
        else:
            self.x_out_flat = x_out

        return x_inp, x_out

    def _link_model(self):
        if self.multiplicity != 2:
            warnings.warn(
                "Link model requested but a generator not supporting links was supplied."
            )
        return self.in_out_tensors(multiplicity=2)

    def _node_model(self):
        if self.multiplicity != 1:
            warnings.warn(
                "Node model requested but a generator not supporting nodes was supplied."
            )
        return self.in_out_tensors(multiplicity=1)

    node_model = deprecated_model_function(_node_model, "node_model")
    link_model = deprecated_model_function(_link_model, "link_model")
    build = deprecated_model_function(in_out_tensors, "build")
