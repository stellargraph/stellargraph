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

import warnings
from tensorflow.keras.layers import Dense, Lambda, Dropout, Input, Layer
import tensorflow.keras.backend as K

from ..mapper import FullBatchGenerator
from .preprocessing_layer import GraphPreProcessingLayer
from .misc import SqueezedSparseConversion


class APPNPPropagationLayer(Layer):

    """
    Implementation of Approximate Personalized Propagation of Neural Predictions (PPNP)
    as in https://arxiv.org/abs/1810.05997.

    Notes:
      - The inputs are tensors with a batch dimension of 1:
        Keras requires this batch dimension, and for full-batch methods
        we only have a single "batch".

      - There are three inputs required, the node features, the output
        indices (the nodes that are to be selected in the final layer)
        and the normalized graph Laplacian matrix

      - This class assumes that the normalized Laplacian matrix is passed as
        input to the Keras methods.

      - The output indices are used when ``final_layer=True`` and the returned outputs
        are the final-layer features for the nodes indexed by output indices.

      - If ``final_layer=False`` all the node features are output in the same ordering as
        given by the adjacency matrix.

    Args:
        units (int): dimensionality of output feature vectors
        final_layer (bool): If False the layer returns output for all nodes,
                            if True it returns the subset specified by the indices passed to it.
        teleport_probability: "probability" of returning to the starting node in the propogation step as desribed  in
        the paper (alpha in the paper)
    """

    def __init__(self, units, teleport_probability=0.1, final_layer=False, **kwargs):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.get("input_dim"),)

        super().__init__(**kwargs)

        self.units = units
        self.teleport_probability = teleport_probability
        self.final_layer = final_layer

    def get_config(self):
        """
        Gets class configuration for Keras serialization.
        Used by keras model serialization.

        Returns:
            A dictionary that contains the config of the layer
        """

        config = {
            "units": self.units,
            "final_layer": self.final_layer,
            "teleport_probability": self.teleport_probability,
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
        feature_shape, out_shape, *As_shapes = input_shapes

        batch_dim = feature_shape[0]
        if self.final_layer:
            out_dim = out_shape[1]
        else:
            out_dim = feature_shape[1]

        return batch_dim, out_dim, self.units

    def build(self, input_shapes):
        """
        Builds the layer

        Args:
            input_shapes (list of int): shapes of the layer's inputs (node features and adjacency matrix)
        """
        self.built = True

    def call(self, inputs):
        """
        Applies the layer.

        Args:
            inputs (list): a list of 3 input tensors that includes
                propagated node features (size 1 x N x F),
                node features (size 1 x N x F),
                output indices (size 1 x M)
                graph adjacency matrix (size N x N),
                where N is the number of nodes in the graph, and
                F is the dimensionality of node features.

        Returns:
            Keras Tensor that represents the output of the layer.
        """
        propagated_features, features, out_indices, *As = inputs
        batch_dim, n_nodes, _ = K.int_shape(features)
        if batch_dim != 1:
            raise ValueError(
                "Currently full-batch methods only support a batch dimension of one"
            )

        # Remove singleton batch dimension
        features = K.squeeze(features, 0)
        propagated_features = K.squeeze(propagated_features, 0)
        out_indices = K.squeeze(out_indices, 0)

        # Propagate the node features
        A = As[0]
        output = (1 - self.teleport_probability) * K.dot(
            A, propagated_features
        ) + self.teleport_probability * features

        # On the final layer we gather the nodes referenced by the indices
        if self.final_layer:
            output = K.gather(output, out_indices)

        # Add batch dimension back if we removed it
        if batch_dim == 1:
            output = K.expand_dims(output, 0)

        return output


class APPNP:
    """
    Implementation of Approximate Personalized Propagation of Neural Predictions (APPNP)
    as in https://arxiv.org/abs/1810.05997.

    The model minimally requires specification of the fully connected layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer,
    activation functions for each hidden layers, and a generator object.

    To use this class as a Keras model, the features and pre-processed adjacency matrix
    should be supplied using either the :class:`FullBatchNodeGenerator` class for node inference
    or the :class:`FullBatchLinkGenerator` class for link inference.

    To have the appropriate pre-processing the generator object should be instanciated
    with the `method='gcn'` argument.

    Example:
        Building an APPNP node model::

            generator = FullBatchNodeGenerator(G, method="gcn")
            ppnp = APPNP(
                layer_sizes=[64, 64, 1], 
                activations=['relu', 'relu', 'relu'], 
                generator=generator,
                dropout=0.5
            )
            x_in, x_out = ppnp.build()

    Notes:
      - The inputs are tensors with a batch dimension of 1. These are provided by the \
        :class:`FullBatchNodeGenerator` object.

      - This assumes that the normalized Laplacian matrix is provided as input to
        Keras methods. When using the :class:`FullBatchNodeGenerator` specify the
        ``method='gcn'`` argument to do this pre-processing.

      - The nodes provided to the :class:`FullBatchNodeGenerator.flow` method are
        used by the final layer to select the predictions for those nodes in order.
        However, the intermediate layers before the final layer order the nodes
        in the same way as the adjacency matrix.

      - The size of the final fully connected layer must be equal to the number of classes to predict.

    Args:
        layer_sizes (list of int): list of output sizes of fully connected layers in the stack
        activations (list of str): list of activations applied to each fully connected layer's output
        generator (FullBatchNodeGenerator): an instance of FullBatchNodeGenerator class constructed on the graph of interest
        bias (bool): toggles an optional bias in fully connected layers
        dropout (float): dropout rate applied to input features of each layer
        kernel_regularizer (str): normalization applied to the kernels of fully connetcted layers
        teleport_probability: "probability" of returning to the starting node in the propagation step as desribed in
        the paper (alpha in the paper)
        approx_iter: number of iterations to approximate PPNP as described in the paper (K in the paper)
    """

    def __init__(
        self,
        layer_sizes,
        generator,
        activations,
        bias=True,
        dropout=0.0,
        teleport_probability=0.1,
        kernel_regularizer=None,
        approx_iter=10,
    ):

        if not isinstance(generator, FullBatchGenerator):
            raise TypeError(
                "Generator should be a instance of FullBatchNodeGenerator or FullBatchLinkGenerator"
            )

        if not len(layer_sizes) == len(activations):
            raise ValueError(
                "The number of layers should equal the number of activations"
            )

        if not isinstance(approx_iter, int) or approx_iter <= 0:
            raise ValueError("approx_iter should be a positive integer")

        if (teleport_probability > 1.0) or (teleport_probability < 0.0):
            raise ValueError(
                "teleport_probability should be between 0 and 1 (inclusive)"
            )

        self.layer_sizes = layer_sizes
        self.teleport_probability = teleport_probability
        self.activations = activations
        self.bias = bias
        self.dropout = dropout
        self.kernel_regularizer = kernel_regularizer
        self.support = 1
        self.approx_iter = approx_iter

        # Copy required information from generator
        self.method = generator.method
        self.multiplicity = generator.multiplicity
        self.n_nodes = generator.features.shape[0]
        self.n_features = generator.features.shape[1]

        # Check if the generator is producing a sparse matrix
        self.use_sparse = generator.use_sparse
        if self.method == "none":
            self.graph_norm_layer = GraphPreProcessingLayer(num_of_nodes=self.n_nodes)

        self._layers = []

        # Initialize a stack of fully connected layers
        n_layers = len(self.layer_sizes)
        for ii in range(n_layers):
            l = self.layer_sizes[ii]
            a = self.activations[ii]
            self._layers.append(Dropout(self.dropout))
            self._layers.append(
                Dense(
                    l,
                    activation=a,
                    use_bias=self.bias,
                    kernel_regularizer=self.kernel_regularizer,
                )
            )

        feature_dim = self.layer_sizes[-1]
        for ii in range(approx_iter):
            self._layers.append(Dropout(self.dropout))
            self._layers.append(
                APPNPPropagationLayer(
                    feature_dim,
                    teleport_probability=self.teleport_probability,
                    final_layer=(ii == (self.approx_iter - 1)),
                )
            )

    def __call__(self, x):
        """
        Apply APPNP to the inputs.
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

        # Otherwise, create dense matrix from input tensor
        else:
            Ainput = [Lambda(lambda A: K.squeeze(A, 0))(A) for A in As]

        # TODO: Support multiple matrices?
        if len(Ainput) != 1:
            raise NotImplementedError(
                "The APPNP method currently only accepts a single matrix"
            )

        h_layer = x_in

        for layer in self._layers[: (2 * len(self.layer_sizes))]:
            h_layer = layer(h_layer)

        feature_layer = h_layer

        for layer in self._layers[(2 * len(self.layer_sizes)) :]:
            if isinstance(layer, APPNPPropagationLayer):
                h_layer = layer([h_layer, feature_layer, out_indices] + Ainput)
            else:
                # For other (non-graph) layers only supply the input tensor
                h_layer = layer(h_layer)

        return h_layer

    def build(self, multiplicity=None):
        """
        Builds an APPNP model for node or link prediction

        Returns:
            tuple: `(x_inp, x_out)`, where `x_inp` is a list of Keras/TensorFlow
            input tensors for the model and `x_out` is a tensor of the model output.
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

    def link_model(self):
        if self.multiplicity != 2:
            warnings.warn(
                "Link model requested but a generator not supporting links was supplied."
            )
        return self.build(multiplicity=2)

    def node_model(self):
        if self.multiplicity != 1:
            warnings.warn(
                "Node model requested but a generator not supporting nodes was supplied."
            )
        return self.build(multiplicity=1)

    def propagate_model(self, base_model):
        """
        Propagates a trained model using personalised PageRank.
        Args:
            base_model (keras Model): trained model with node features as input, predicted classes as output

        returns:
            tuple: `(x_inp, x_out)`, where `x_inp` is a list of two Keras input tensors
            for the APPNP model (containing node features and graph adjacency),
            and `x_out` is a Keras tensor for the APPNP model output.
        """
        if self.multiplicity != 1:
            raise RuntimeError(
                "APPNP does not currently support propagating a link model"
            )

        out_indices_t = Input(batch_shape=(1, None), dtype="int32")

        if self.use_sparse:
            # Placeholders for the sparse adjacency matrix
            A_indices_t = Input(batch_shape=(1, None, 2), dtype="int64")
            A_values_t = Input(batch_shape=(1, None))
            A_placeholders = [A_indices_t, A_values_t]

        else:
            # Placeholders for the dense adjacency matrix
            A_m = Input(batch_shape=(1, self.n_nodes, self.n_nodes))
            A_placeholders = [A_m]

        if self.use_sparse:
            A_indices, A_values = A_placeholders
            Ainput = [
                SqueezedSparseConversion(
                    shape=(self.n_nodes, self.n_nodes), dtype=A_values.dtype
                )([A_indices, A_values])
            ]

        # Otherwise, create dense matrix from input tensor
        else:
            Ainput = [Lambda(lambda A: K.squeeze(A, 0))(A) for A in A_placeholders]

        # Inputs for features & target indices
        x_t = Input(batch_shape=(1, self.n_nodes, self.n_features))
        x_inp = [x_t, out_indices_t] + A_placeholders

        # pass the node features through the base model
        feature_layer = x_t
        for layer in base_model.layers[1:]:
            feature_layer = layer(feature_layer)

        h_layer = feature_layer
        # iterate through APPNPPropagation layers
        for layer in self._layers[(2 * len(self.layer_sizes)) :]:
            if isinstance(layer, APPNPPropagationLayer):
                h_layer = layer([h_layer, feature_layer, out_indices_t] + Ainput)
            else:
                h_layer = layer(h_layer)

        x_out = h_layer
        return x_inp, x_out
