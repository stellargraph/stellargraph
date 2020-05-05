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

from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Layer, Lambda, Dropout, Reshape
from .misc import deprecated_model_function, GatherIndices
from ..mapper import ClusterNodeGenerator
from .gcn import GraphConvolution

import warnings


class ClusterGraphConvolution(GraphConvolution):
    """
    Deprecated: use :class:`GraphConvolution`.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ClusterGraphConvolution has been replaced by GraphConvolution without functionality change",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class ClusterGCN:
    """
    A stack of Cluster Graph Convolutional layers that implement a cluster graph convolution network
    model as in https://arxiv.org/abs/1905.07953

    The model minimally requires specification of the layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer,
    activation functions for each hidden layers, and a generator object.

    To use this class as a Keras model, the features and pre-processed adjacency matrix
    should be supplied using the :class:`ClusterNodeGenerator` class.

    For more details, please see `the Cluster-GCN demo notebook
    <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/cluster-gcn-node-classification.html>`_

    Notes:
      - The inputs are tensors with a batch dimension of 1. These are provided by the \
        :class:`ClusterNodeGenerator` object.

      - The nodes provided to the :class:`ClusterNodeGenerator.flow` method are
        used by the final layer to select the predictions for those nodes in order.
        However, the intermediate layers before the final layer order the nodes
        in the same way as the adjacency matrix.

    Examples:
        Creating a Cluster-GCN node classification model from an existing :class:`StellarGraph`
        object ``G``::

            generator = ClusterNodeGenerator(G, clusters=10, q=2)
            cluster_gcn = ClusterGCN(
                             layer_sizes=[32, 4],
                             activations=["elu","softmax"],
                             generator=generator,
                             dropout=0.5
                )
            x_inp, predictions = cluster_gcn.in_out_tensors()

    Args:
        layer_sizes (list of int): list of output sizes of the graph convolutional layers in the stack
        activations (list of str): list of activations applied to each layer's output
        generator (ClusterNodeGenerator): an instance of ClusterNodeGenerator class constructed on the graph of interest
        bias (bool): toggles an optional bias in graph convolutional layers
        dropout (float): dropout rate applied to input features of each graph convolutional layer
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
        activations,
        generator,
        bias=True,
        dropout=0.0,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
    ):
        if not isinstance(generator, ClusterNodeGenerator):
            raise TypeError("Generator should be a instance of ClusterNodeGenerator")

        if len(layer_sizes) != len(activations):
            raise AssertionError(
                "The number of given layers should be the same as the number of activations."
                "However given len(layer_sizes): {} vs len(activations): {}".format(
                    len(layer_sizes), len(activations)
                )
            )

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.bias = bias
        self.dropout = dropout
        self.generator = generator
        self.support = 1

        # Initialize a stack of Cluster GCN layers
        n_layers = len(self.layer_sizes)
        self._layers = []
        for ii in range(n_layers):
            l = self.layer_sizes[ii]
            a = self.activations[ii]
            self._layers.append(Dropout(self.dropout))
            self._layers.append(
                GraphConvolution(
                    l,
                    activation=a,
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
        Apply a stack of Cluster GCN-layers to the inputs.
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

        h_layer = x_in

        for layer in self._layers:
            if isinstance(layer, GraphConvolution):
                # For a GCN layer add the matrix
                h_layer = layer([h_layer] + As)
            else:
                # For other (non-graph) layers only supply the input tensor
                h_layer = layer(h_layer)

        # only return data for the requested nodes
        h_layer = GatherIndices(batch_dims=1)([h_layer, out_indices])

        return h_layer

    def in_out_tensors(self):
        """
        Builds a Cluster-GCN model for node prediction.

        Returns:
            tuple: `(x_inp, x_out)`, where `x_inp` is a list of two input tensors for the
            Cluster-GCN model (containing node features and normalized adjacency matrix),
            and `x_out` is a tensor for the Cluster-GCN model output.
        """
        # Placeholder for node features
        N_feat = self.generator.features.shape[1]

        # Inputs for features & target indices
        x_t = Input(batch_shape=(1, None, N_feat))
        out_indices_t = Input(batch_shape=(1, None), dtype="int32")

        # Placeholders for the dense adjacency matrix
        A_m = Input(batch_shape=(1, None, None))
        A_placeholders = [A_m]

        x_inp = [x_t, out_indices_t] + A_placeholders
        x_out = self(x_inp)

        return x_inp, x_out

    build = deprecated_model_function(in_out_tensors, "build")
