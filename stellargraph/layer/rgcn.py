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

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Lambda, Dropout, Input
from tensorflow.keras import activations, initializers, constraints, regularizers
from .misc import SqueezedSparseConversion, deprecated_model_function, GatherIndices
from ..mapper.full_batch_generators import RelationalFullBatchNodeGenerator


class RelationalGraphConvolution(Layer):
    """
        Relational Graph Convolution (RGCN) Keras layer.

        Original paper: Modeling Relational Data with Graph Convolutional Networks.
        Thomas N. Kipf, Michael Schlichtkrull (2017). https://arxiv.org/pdf/1703.06103.pdf

        Notes:
          - The inputs are tensors with a batch dimension of 1:
            Keras requires this batch dimension, and for full-batch methods
            we only have a single "batch".

          - There are 1 + R inputs required (where R is the number of relationships): the node features,
            and a normalized adjacency matrix for each relationship

        Args:
            units (int): dimensionality of output feature vectors
            num_relationships (int): the number of relationships in the graph
            num_bases (int): the number of basis matrices to use for parameterizing the weight matrices as described in
                the paper; defaults to 0. num_bases < 0 triggers the default behaviour of num_bases = 0
            activation (str or func): nonlinear activation applied to layer's output to obtain output features
            use_bias (bool): toggles an optional bias
            final_layer (bool): Deprecated, use ``tf.gather`` or :class:`GatherIndices`
            kernel_initializer (str or func): The initialiser to use for the self kernel and also relational kernels if num_bases=0.
            kernel_regularizer (str or func): The regulariser to use for the self kernel and also relational kernels if num_bases=0.
            kernel_constraint (str or func): The constraint to use for the self kernel and also relational kernels if num_bases=0.
            basis_initializer (str or func): The initialiser to use for the basis matrices.
            basis_regularizer (str or func): The regulariser to use for the basis matrices.
            basis_constraint (str or func): The constraint to use for the basis matrices.
            coefficient_initializer (str or func): The initialiser to use for the coefficients.
            coefficient_regularizer (str or func): The regulariser to use for the coefficients.
            coefficient_constraint (str or func): The constraint to use for the coefficients.
            bias_initializer (str or func): The initialiser to use for the bias.
            bias_regularizer (str or func): The regulariser to use for the bias.
            bias_constraint (str or func): The constraint to use for the bias.
            input_dim (int, optional): the size of the input shape, if known.
            kwargs: any additional arguments to pass to :class:`tensorflow.keras.layers.Layer`
        """

    def __init__(
        self,
        units,
        num_relationships,
        num_bases=0,
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
        basis_initializer="glorot_uniform",
        basis_regularizer=None,
        basis_constraint=None,
        coefficient_initializer="glorot_uniform",
        coefficient_regularizer=None,
        coefficient_constraint=None,
        **kwargs
    ):
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        super().__init__(**kwargs)

        if not isinstance(num_bases, int):
            raise TypeError("num_bases should be an int")

        if not isinstance(units, int):
            raise TypeError("units should be an int")

        if units <= 0:
            raise ValueError("units should be positive")

        if not isinstance(num_relationships, int):
            raise TypeError("num_relationships should be an int")

        if num_relationships <= 0:
            raise ValueError("num_relationships should be positive")

        self.units = units
        self.num_relationships = num_relationships
        self.num_bases = num_bases
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.basis_initializer = initializers.get(basis_initializer)
        self.basis_regularizer = regularizers.get(basis_regularizer)
        self.basis_constraint = constraints.get(basis_constraint)
        self.coefficient_initializer = initializers.get(coefficient_initializer)
        self.coefficient_regularizer = regularizers.get(coefficient_regularizer)
        self.coefficient_constraint = constraints.get(coefficient_constraint)

        if final_layer is not None:
            raise ValueError(
                "'final_layer' is not longer supported, use 'tf.gather' or 'GatherIndices' separately"
            )

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
            "basis_initializer": initializers.serialize(self.basis_initializer),
            "coefficient_initializer": initializers.serialize(
                self.coefficient_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "basis_regularizer": regularizers.serialize(self.basis_regularizer),
            "coefficient_regularizer": regularizers.serialize(
                self.coefficient_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "basis_constraint": constraints.serialize(self.basis_constraint),
            "coefficient_constraint": constraints.serialize(
                self.coefficient_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "num_relationships": self.num_relationships,
            "num_bases": self.num_bases,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.

        Args:
            input_shapes (tuple of ints)
                Shape tuples can include None for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """
        feature_shape, A_shape = input_shapes

        batch_dim = feature_shape[0]
        out_dim = feature_shape[1]

        return batch_dim, out_dim, self.units

    def build(self, input_shapes):
        """
        Builds the layer

        Args:
            input_shapes (list of int): shapes of the layer's inputs
            (node features, node_indices, and adjacency matrices)

        """
        feat_shape = input_shapes[0]
        input_dim = int(feat_shape[-1])

        if self.num_bases > 0:

            # creates a kernel for each edge type/relationship in the graph
            # each kernel is a linear combination of basis matrices
            # the basis matrices are shared for all edge types/relationships
            # each edge type has a different set of learnable coefficients

            # initialize the shared basis matrices
            self.bases = self.add_weight(
                shape=(input_dim, self.units, self.num_bases),
                initializer=self.basis_initializer,
                name="bases",
                regularizer=self.basis_regularizer,
                constraint=self.basis_constraint,
            )

            # initialize the coefficients for each edge type/relationship
            self.coefficients = [
                self.add_weight(
                    shape=(self.num_bases,),
                    initializer=self.coefficient_initializer,
                    name="coeff",
                    regularizer=self.coefficient_regularizer,
                    constraint=self.coefficient_constraint,
                )
                for _ in range(self.num_relationships)
            ]

            # To support eager TF the relational_kernels need to be explicitly calculated
            # each time the layer is called
            self.relational_kernels = None

        else:
            self.bases = None
            self.coefficients = None

            self.relational_kernels = [
                self.add_weight(
                    shape=(input_dim, self.units),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                )
                for _ in range(self.num_relationships)
            ]

        self.self_kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
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
            inputs (list): a list of 2 + R input tensors that includes
                node features (size 1 x N x F),
                and a graph adjacency matrix (size N x N) for each relationship.
                R is the number of relationships in the graph (edge type),
                N is the number of nodes in the graph, and
                F is the dimensionality of node features.

        Returns:
            Keras Tensor that represents the output of the layer.
        """
        features, *As = inputs
        batch_dim, n_nodes, _ = K.int_shape(features)
        if batch_dim != 1:
            raise ValueError(
                "Currently full-batch methods only support a batch dimension of one"
            )

        # Remove singleton batch dimension
        features = K.squeeze(features, 0)

        # Calculate the layer operation of RGCN
        output = K.dot(features, self.self_kernel)

        if self.relational_kernels is None:
            # explicitly calculate the relational kernels if basis matrices are used
            relational_kernels = [
                tf.einsum("ijk,k->ij", self.bases, coeff) for coeff in self.coefficients
            ]
        else:
            relational_kernels = self.relational_kernels

        for i in range(self.num_relationships):
            h_graph = K.dot(As[i], features)
            output += K.dot(h_graph, relational_kernels[i])

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias
        output = self.activation(output)

        # Add batch dimension back if we removed it
        if batch_dim == 1:
            output = K.expand_dims(output, 0)

        return output


class RGCN:
    """
    A stack of Relational Graph Convolutional layers that implement a relational graph
    convolution neural network model as in https://arxiv.org/pdf/1703.06103.pdf

    The model minimally requires specification of the layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer,
    activation functions for each hidden layers, and a generator object.

    To use this class as a Keras model, the features and pre-processed adjacency matrix
    should be supplied using the :class:`RelationalFullBatchNodeGenerator` class.
    The generator object should be instantiated as follows::

        generator = RelationalFullBatchNodeGenerator(G)

    Note that currently the RGCN class is compatible with both sparse and dense adjacency
    matrices and the :class:`RelationalFullBatchNodeGenerator` will default to sparse.

    For more details, please see `the RGCN demo notebook <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/rgcn-node-classification.html>`_

    Notes:
      - The inputs are tensors with a batch dimension of 1. These are provided by the \
        :class:`RelationalFullBatchNodeGenerator` object.

      - The nodes provided to the :class:`RelationalFullBatchNodeGenerator.flow` method are
        used by the final layer to select the predictions for those nodes in order.
        However, the intermediate layers before the final layer order the nodes
        in the same way as the adjacency matrix.

    Examples:
        Creating a RGCN node classification model from an existing :class:`StellarGraph`
        object ``G``::

            generator = RelationalFullBatchNodeGenerator(G)
            rgcn = RGCN(
                    layer_sizes=[32, 4],
                    activations=["elu","softmax"],
                    bases=10,
                    generator=generator,
                    dropout=0.5
                )
            x_inp, predictions = rgcn.in_out_tensors()

    Args:
        layer_sizes (list of int): Output sizes of RGCN layers in the stack.
        generator (RelationalFullBatchNodeGenerator): The generator instance.
        num_bases (int): Specifies number of basis matrices to use for the weight matrics of the RGCN layer
            as in the paper. Defaults to 0 which specifies that no basis decomposition is used.
        bias (bool): If True, a bias vector is learnt for each layer in the RGCN model.
        dropout (float): Dropout rate applied to input features of each RGCN layer.
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
        num_bases=0,
        dropout=0.0,
        activations=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
    ):
        if not isinstance(generator, RelationalFullBatchNodeGenerator):
            raise TypeError(
                "Generator should be a instance of RelationalFullBatchNodeGenerator"
            )

        n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.bias = bias
        self.num_bases = num_bases
        self.dropout = dropout

        # Copy required information from generator
        self.multiplicity = generator.multiplicity
        self.n_nodes = generator.features.shape[0]
        self.n_features = generator.features.shape[1]
        self.n_edge_types = len(generator.As)

        # Check if the generator is producing a sparse matrix
        self.use_sparse = generator.use_sparse

        # Activation function for each layer
        if activations is None:
            activations = ["relu"] * n_layers

        elif len(activations) != n_layers:
            raise ValueError(
                "Invalid number of activations; require one function per layer"
            )

        self.activations = activations
        self.num_bases = num_bases

        # Initialize a stack of RGCN layers
        self._layers = []
        for ii in range(n_layers):
            self._layers.append(Dropout(self.dropout))
            self._layers.append(
                RelationalGraphConvolution(
                    self.layer_sizes[ii],
                    num_relationships=len(generator.As),
                    num_bases=self.num_bases,
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
        Apply a stack of RGCN layers to the inputs.
        The input tensors are expected to be a list of the following:
        [Node features shape (1, N, F), Output indices (1, Z)] +
        [Adjacency indices for each relationship (1, E, 2) for _ in range(R)]
        [Adjacency values for each relationshiop (1, E) for _ in range(R)]


        where N is the number of nodes, F the number of input features,
              E is the number of edges, Z the number of output nodes,
              R is the number of relationships in the graph (edge types).

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

        # Convert input indices & values to sparse matrices
        if self.use_sparse:

            As_indices = As[: self.n_edge_types]
            As_values = As[self.n_edge_types :]

            Ainput = [
                SqueezedSparseConversion(
                    shape=(n_nodes, n_nodes), dtype=As_values[i].dtype
                )([As_indices[i], As_values[i]])
                for i in range(self.n_edge_types)
            ]

        # Otherwise, create dense matrices from input tensor
        else:
            Ainput = [Lambda(lambda A: K.squeeze(A, 0))(A_) for A_ in As]

        h_layer = x_in

        for layer in self._layers:
            if isinstance(layer, RelationalGraphConvolution):
                # For an RGCN layer add the adjacency matrices
                h_layer = layer([h_layer] + Ainput)
            else:
                # For other (non-graph) layers only supply the input tensor
                h_layer = layer(h_layer)

        # only return data for the requested nodes
        h_layer = GatherIndices(batch_dims=1)([h_layer, out_indices])

        return h_layer

    def _node_model(self):
        """
        Builds a RGCN model for node prediction

        Returns:
            tuple: `(x_inp, x_out)`, where
            `x_inp` is a list of Keras input tensors for the RGCN model (containing node features,
             node indices, and the indices and values for the sparse adjacency matrices for each relationship),
            and `x_out` is a Keras tensor for the RGCN model output.
        """

        # Inputs for features & target indices
        x_t = Input(batch_shape=(1, self.n_nodes, self.n_features))
        out_indices_t = Input(batch_shape=(1, None), dtype="int32")

        # Create inputs for sparse or dense matrices
        if self.use_sparse:
            # Placeholders for the sparse adjacency matrix
            A_indices_t = [
                Input(batch_shape=(1, None, 2), dtype="int64")
                for i in range(self.n_edge_types)
            ]
            A_values_t = [
                Input(batch_shape=(1, None)) for i in range(self.n_edge_types)
            ]
            A_placeholders = A_indices_t + A_values_t

        else:
            # Placeholders for the dense adjacency matrix
            A_placeholders = [
                Input(batch_shape=(1, self.n_nodes, self.n_nodes))
                for i in range(self.n_edge_types)
            ]

        x_inp = [x_t, out_indices_t] + A_placeholders
        x_out = self(x_inp)

        # Flatten output by removing singleton batch dimension
        if x_out.shape[0] == 1:
            self.x_out_flat = Lambda(lambda x: K.squeeze(x, 0))(x_out)
        else:
            self.x_out_flat = x_out

        return x_inp, x_out

    def in_out_tensors(self, multiplicity=None):
        """
        Builds a RGCN model for node prediction. Link/node pair prediction will added in the future.

        Returns:
            tuple: (x_inp, x_out), where ``x_inp`` is a list of Keras input tensors
            for the specified RGCN model and ``x_out`` contains
            model output tensor(s) of shape (batch_size, layer_sizes[-1])

        """
        if multiplicity is None:
            multiplicity = self.multiplicity

        if multiplicity == 1:
            return self._node_model()

        else:
            raise NotImplementedError(
                "Currently only node prediction if supported for RGCN."
            )

    build = deprecated_model_function(in_out_tensors, "build")
