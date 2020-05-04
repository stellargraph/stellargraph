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
from .gcn import GraphConvolution
from .sort_pooling import SortPooling
from tensorflow.keras.layers import Input, Dropout, GlobalAveragePooling1D


class GCNSupervisedGraphClassification:
    """
    A stack of :class:`GraphConvolution` layers together with a Keras `GlobalAveragePooling1D` layer (by default)
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

        pooling (callable, optional): a Keras layer or function that takes two arguments and returns
            a tensor representing the embeddings for each graph in the batch. Arguments:

            - embeddings tensor argument with shape ``batch size × nodes × output size``, where
              ``nodes`` is the maximum number of nodes of a graph in the batch and ``output size``
              is the size of the final graph convolutional layer, or, if ``pool_all_layers``, the
              sum of the sizes of each graph convolutional layers.
            - ``mask`` tensor named argument of booleans with shape ``batch size × nodes``. ``True``
              values indicate which rows of the embeddings argument are valid, and all other rows
              (corresponding to ``mask == False``) must be ignored.

            The returned tensor can have any shape ``batch size``, ``batch size × N1``, ``batch size
            × N1 × N2``, ..., as long as the ``N1``, ``N2``, ... are constant across all graphs:
            they must not depend on the ``nodes`` dimension or on the number of ``True`` values in
            ``mask``. ``pooling`` defaults to mean pooling via ``GlobalAveragePooling1D``.

        pool_all_layers (bool, optional): which layers to pass to the pooling method: if ``True``,
            pass the concatenation of the output of every GCN layer, otherwise pass only the output
            of the last GCN layer.
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
        pooling=None,
        pool_all_layers=False,
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

        if pooling is not None:
            self.pooling = pooling
        else:
            self.pooling = GlobalAveragePooling1D(data_format="channels_last")

        self.pool_all_layers = pool_all_layers

        # Initialize a stack of GraphConvolution layers
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
        Apply a stack of :class:`GraphConvolution` layers to the inputs.
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

        gcn_layers = []

        for layer in self._layers:
            if isinstance(layer, GraphConvolution):
                h_layer = layer([h_layer, As])
                gcn_layers.append(h_layer)
            else:
                # For other (non-graph) layers only supply the input tensor
                h_layer = layer(h_layer)

        if self.pool_all_layers:
            h_layer = tf.concat(gcn_layers, axis=-1)

        # mask to ignore the padded values
        h_layer = self.pooling(h_layer, mask=mask)

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


class DeepGraphCNN(GCNSupervisedGraphClassification):
    """
    A stack of :class:`GraphConvolution` layers together with a `SortPooling` layer
    that implement a supervised graph classification network (DGCNN) using the GCN convolution operator
    (https://arxiv.org/abs/1609.02907).

    The DGCNN model was introduced in the paper, "An End-to-End Deep Learning Architecture for Graph Classification" by
    M. Zhang, Z. Cui, M. Neumann, and Y. Chen, AAAI 2018, https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf

    The model minimally requires specification of the GCN layer sizes as a list of ints corresponding to the feature
    dimensions for each hidden layer, activation functions for each hidden layer, a generator object, and the number of
    output nodes for the class:`SortPooling` layer.

    To use this class as a Keras model, the features and pre-processed adjacency matrix should be supplied using the
    :class:`PaddedGraphGenerator` class.

    Examples:
        Creating a graph classification model from a list of :class:`StellarGraph`
        objects (``graphs``). We also add two one-dimensional convolutional layers, a max pooling layer, and two fully
        connected dense layers one with dropout one used for binary classification::

            generator = PaddedGraphGenerator(graphs)
            model = DeepGraphCNN(
                             layer_sizes=[32, 32, 32, 1],
                             activations=["tanh","tanh", "tanh", "tanh"],
                             generator=generator,
                             k=30
                )
            x_inp, x_out = model.in_out_tensors()

            x_out = Conv1D(filters=16, kernel_size=97, strides=97)(x_out)
            x_out = MaxPool1D(pool_size=2)(x_out)
            x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)
            x_out = Flatten()(x_out)
            x_out = Dense(units=128, activation="relu")(x_out)
            x_out = Dropout(rate=0.5)(x_out)
            predictions = Dense(units=1, activation="sigmoid")(x_out)

            model = Model(inputs=x_inp, outputs=predictions)


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
            pooling=SortPooling(k=k, flatten_output=True),
            pool_all_layers=True,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
        )
