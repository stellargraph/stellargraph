# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.keras.layers import Layer
from ..core.experimental import experimental


@experimental(reason="Missing unit tests and generally untested.", issues=[1044])
class SortPooling(Layer):

    """
    Sort Pooling Keras layer. A stack of such layers together with Keras convolutional layers can be used to create
    graph classification models.

    Original paper: An End-to-End Deep Learning Atchitecture for Graph Classification, M. Zhang, Z. Cui, M. Neumann, and
    Y. Chen, AAAI-18, https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/17146


    Args:
        k (int): The number of rows of output tensor.

    """

    def __init__(self, k):
        super().__init__()

        self.trainable = False
        self.k = k

    def get_config(self):
        """
        Gets class configuration for Keras serialization.
        Used by keras model serialization.

        Returns:
            A dictionary that contains the config of the layer
        """
        return {"k": self.k}

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
        return input_shapes[0], self.k, input_shapes[2]

    def call(self, inputs, **kwargs):
        """
        Applies the layer.

        Args:
            inputs (list): a list of 3 input tensors that includes
                node features (size B x N x Sum F_i),

                where B is the batch size, N is the number of nodes in the largest graph in the batch, and
                F_i is the dimensionality of node features output from the i-th convolutional layer.

        Returns:
            Keras Tensor that represents the output of the layer.
        """
        outputs = tf.map_fn(
            lambda x: tf.gather(
                x, tf.argsort(x[..., -1], axis=0, direction="DESCENDING")
            ),
            inputs,
        )

        # Truncate or pad to size self.k
        if outputs.shape[1] < self.k:
            outputs = tf.pad(outputs, [[0, 0], [0, self.k - outputs.shape[1]], [0, 0]])
        elif outputs.shape[1] > self.k:
            outputs = outputs[:, : self.k, :]

        return outputs
