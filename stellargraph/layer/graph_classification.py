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
from .misc import deprecated_model_function
from ..mapper import PaddedGraphGenerator
from .cluster_gcn import ClusterGraphConvolution
from .sort_pooling import SortPooling

from tensorflow.keras.layers import Input, Dropout, GlobalAveragePooling1D
from ..core.experimental import experimental


class GraphClassificationConvolution(ClusterGraphConvolution):

    """
    A graph convolutional Keras layer. A stack of such layers can be used to create a model for supervised graph
    classification.

    The implementation is based on the GCN Keras layer of keras-gcn github
    repo https://github.com/tkipf/keras-gcn

    Notes:
      - The inputs are tensors with a batch dimension.

      - There are 2 inputs required, the node features and the normalized graph adjacency matrices for each batch entry.

      - This class assumes that the normalized graph adjacency matrices are passed as input to the Keras methods.

    Args:
        units (int): dimensionality of output feature vectors
        activation (str): nonlinear activation applied to layer's output to obtain output features
        use_bias (bool): toggles an optional bias

        kernel_initializer (str or func, optional): The initialiser to use for the weights of each graph
            convolutional layer.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights of each graph
            convolutional layer.
        kernel_constraint (str or func, optional): The constraint to use for the weights of each layer graph
            convolutional.
        bias_initializer (str or func, optional): The initialiser to use for the bias of each layer graph
            convolutional.
        bias_regularizer (str or func, optional): The regulariser to use for the bias of each layer graph
            convolutional.
        bias_constraint (str or func, optional): The constraint to use for the bias of each layer graph
            convolutional.
     """

    def call(self, inputs):
        """
        Applies the layer.

        Args:
            inputs (list or tuple): a list or tuple of 2 input tensors that includes
                node features (size batch_size x N x F),
                graph adjacency matrix (batch_size x N x N),
                where N is the number of nodes in the graph, and F is the dimensionality of node features.

        Returns:
            Keras Tensor that represents the output of the layer.
        """
        features, A = inputs

        h_graph = K.batch_dot(A, features)
        output = K.dot(h_graph, self.kernel)

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias
        output = self.activation(output)

        return output


class GCNSupervisedGraphClassification:
    """
    A stack of :class:`GraphClassificationConvolution` layers together with a Keras `GlobalAveragePooling1D` layer
    that implement a supervised graph classification network using the GCN convolution operator
    (https://arxiv.org/abs/1609.02907).

    The model minimally requires specification of the GCN layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer,
    activation functions for each hidden layers, and a generator object.

    To use this class as a Keras model, the features and pre-processed adjacency matrix
    should be supplied using the :class:`PaddedGraphGenerator` class.

    Examples:
        Creating a graph classification model from a list of :class:`StellarGraph`
        objects (``graphs``). We also add two fully connected dense layers using the last one for binary classification
        with `softmax` activation::

            generator = PaddedGraphGenerator(graphs)
            model = GCNSupervisedGraphClassification(
                             layer_sizes=[32, 32],
                             activations=["elu","elu"],
                             generator=generator,
                             dropout=0.5
                )
            x_inp, x_out = model.in_out_tensors()
            predictions = Dense(units=8, activation='relu')(x_out)
            predictions = Dense(units=2, activation='softmax')(predictions)

    Args:
        layer_sizes (list of int): list of output sizes of the graph GCN layers in the stack.
        activations (list of str): list of activations applied to each GCN layer's output.
        generator (PaddedGraphGenerator): an instance of :class:`PaddedGraphGenerator` class constructed on the graphs used for
            training.
        bias (bool, optional): toggles an optional bias in graph convolutional layers.
        dropout (float, optional): dropout rate applied to input features of each GCN layer.
        kernel_initializer (str or func, optional): The initialiser to use for the weights of each graph
            convolutional layer.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights of each graph
            convolutional layer.
        kernel_constraint (str or func, optional): The constraint to use for the weights of each layer graph
            convolutional.
        bias_initializer (str or func, optional): The initialiser to use for the bias of each layer graph
            convolutional.
        bias_regularizer (str or func, optional): The regulariser to use for the bias of each layer graph
            convolutional.
        bias_constraint (str or func, optional): The constraint to use for the bias of each layer graph
            convolutional.

    """

    def __init__(
        self,
        layer_sizes,
        activations,
        generator,
        bias=True,
        dropout=0.0,
        kernel_initializer=None,
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer=None,
        bias_regularizer=None,
        bias_constraint=None,
    ):
        if not isinstance(generator, PaddedGraphGenerator):
            raise TypeError(
                f"generator: expected instance of PaddedGraphGenerator, found {type(generator).__name__}"
            )

        if len(layer_sizes) != len(activations):
            raise ValueError(
                "expected the number of layers to be the same as the number of activations,"
                f"found {len(layer_sizes)} layer sizes vs {len(activations)} activations"
            )

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.bias = bias
        self.dropout = dropout
        self.generator = generator

        # Initialize a stack of GraphClassificationConvolution layers
        n_layers = len(self.layer_sizes)
        self._layers = []
        for ii in range(n_layers):
            l = self.layer_sizes[ii]
            a = self.activations[ii]
            self._layers.append(Dropout(self.dropout))
            self._layers.append(
                GraphClassificationConvolution(
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
        Apply a stack of :class:`GraphClassificationConvolution` layers to the inputs.
        The input tensors are expected to be a list of the following:
        [
            Node features shape (batch size, N, F),
            Mask (batch size, N ),
            Adjacency matrices (batch size, N, N),
        ]
        where N is the number of nodes and F the number of input features

        Args:
            x (Tensor): input tensors

        Returns:
            Output tensor
        """
        x_in, mask, As = x
        h_layer = x_in

        for layer in self._layers:
            if isinstance(layer, GraphClassificationConvolution):
                h_layer = layer([h_layer, As])
            else:
                # For other (non-graph) layers only supply the input tensor
                h_layer = layer(h_layer)

        # add mean pooling layer with mask in order to ignore the padded values
        h_layer = GlobalAveragePooling1D(data_format="channels_last")(
            h_layer, mask=mask
        )

        return h_layer

    def in_out_tensors(self):
        """
        Builds a Graph Classification model.

        Returns:
            tuple: `(x_inp, x_out)`, where `x_inp` is a list of two input tensors for the
            Graph Classification model (containing node features and normalized adjacency matrix),
            and `x_out` is a tensor for the Graph Classification model output.
        """
        x_t = Input(shape=(None, self.generator.node_features_size))
        mask = Input(shape=(None,), dtype=tf.bool)
        A_m = Input(shape=(None, None))

        x_inp = [x_t, mask, A_m]
        x_out = self(x_inp)

        return x_inp, x_out

    build = deprecated_model_function(in_out_tensors, "build")


@experimental(reason="Missing unit tests and generally untested.")
class DeepGraphConvolutionalNeuralNetwork(GCNSupervisedGraphClassification):
    """
    A stack of :class:`GraphClassificationConvolution` layers together with a `SortPooling` layer
    that implement a supervised graph classification network (DGCNN) using the GCN convolution operator
    (https://arxiv.org/abs/1609.02907).

    The DGCNN model was introduced in the paper, "An End-to-End Deep Learning Architecture for Graph Classification" by
    M. Zhang, Z. Cui, M. Neumann, and Y. Chen, AAAI 2018, https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf

    The model minimally requires specification of the GCN layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer,
    activation functions for each hidden layers, and a generator object.

    To use this class as a Keras model, the features and pre-processed adjacency matrix
    should be supplied using the :class:`GraphGenerator` class.

    Args:
        layer_sizes (list of int): list of output sizes of the graph GCN layers in the stack.
        activations (list of str): list of activations applied to each GCN layer's output.
        k (int): size (number of rows) of output tensor.
        generator (GraphGenerator): an instance of :class:`GraphGenerator` class constructed on the graphs used for
            training.
        bias (bool, optional): toggles an optional bias in graph convolutional layers.
        dropout (float, optional): dropout rate applied to input features of each GCN layer.
        kernel_initializer (str or func, optional): The initialiser to use for the weights of each graph
            convolutional layer.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights of each graph
            convolutional layer.
        kernel_constraint (str or func, optional): The constraint to use for the weights of each layer graph
            convolutional.
        bias_initializer (str or func, optional): The initialiser to use for the bias of each layer graph
            convolutional.
        bias_regularizer (str or func, optional): The regulariser to use for the bias of each layer graph
            convolutional.
        bias_constraint (str or func, optional): The constraint to use for the bias of each layer graph
            convolutional.

    """

    def __init__(
        self,
        layer_sizes,
        activations,
        k,
        generator,
        bias=True,
        dropout=0.0,
        kernel_initializer=None,
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer=None,
        bias_regularizer=None,
        bias_constraint=None,
    ):
        super().__init__(
            layer_sizes=layer_sizes,
            activations=activations,
            generator=generator,
            bias=bias,
            dropout=dropout,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
        )

        if not isinstance(k, int):
            raise TypeError(
                f"k: expected k to be integer type, found {type(k).__name__}."
            )

        if k <= 0:
            raise ValueError(f"k: expected k to be strictly positive, found {k}")

        self.k = k

        # Add the SortPooling layer
        self._layers.append(SortPooling(k=self.k, flatten_output=True))

    def __call__(self, x):
        """
        Apply a stack of :class:`GraphClassificationConvolution` layers to the inputs followed by a single
        SortPooling layer.
        The input tensors are expected to be a list of the following:
        [
            Node features shape (batch size, N, F),
            Mask (batch size, N ),
            Adjacency matrices (batch size, N, N),
        ]
        where N is the number of nodes and F the number of input features

        Args:
            x (Tensor): input tensors

        Returns:
            Output tensor
        """
        gcn_layers = []

        x_in, mask, As = x
        h_layer = x_in

        for layer in self._layers:
            if isinstance(layer, GraphClassificationConvolution):
                h_layer = layer([h_layer, As])
                gcn_layers.append(h_layer)
            elif isinstance(layer, SortPooling):
                # concatenate the GCN output tensors and use as input to the SortPooling layer
                h_layer = tf.concat(gcn_layers, axis=-1)
                h_layer = layer([h_layer, mask])
            else:
                # For other (non-graph) layers only supply the input tensor
                h_layer = layer(h_layer)

        return h_layer
