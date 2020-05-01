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
from ..core.validation import require_integer_in_range


class SortPooling(Layer):
    """
    Sort Pooling Keras layer.

    Note that sorting is performed using only the last column of the input tensor as stated in [1], "For convenience,
    we set the last graph convolution to have one channel and only used this single channel for sorting."

    [1] An End-to-End Deep Learning Atchitecture for Graph Classification, M. Zhang, Z. Cui, M. Neumann, and
    Y. Chen, AAAI-18, https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/17146

    Args:
        k (int): The number of rows of output tensor.
        flatten_output (bool): If True then the output tensor is reshaped to vector for each element in the batch.
    """

    def __init__(self, k, flatten_output=False):
        super().__init__()

        require_integer_in_range(k, "k", min_val=1)

        self.trainable = False
        self.k = k
        self.flatten_output = flatten_output

    def get_config(self):
        """
        Gets class configuration for Keras serialization. Used by keras model serialization.

        Returns:
            A dictionary that contains the config of the layer
        """
        return {"k": self.k, "flatten_output": self.flatten_output}

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
        if self.flatten_output:
            return input_shapes[0], self.k * input_shapes[2], 1
        else:
            return input_shapes[0], self.k, input_shapes[2]

    def _sort_tensor_with_mask(self, inputs):

        embeddings, mask = inputs[0], inputs[1]

        masked_sorted_embeddings = tf.gather(
            embeddings,
            tf.argsort(
                tf.boolean_mask(embeddings, mask)[..., -1],
                axis=0,
                direction="DESCENDING",
            ),
        )

        embeddings = tf.pad(
            masked_sorted_embeddings,
            [
                [0, (tf.shape(embeddings)[0] - tf.shape(masked_sorted_embeddings)[0])],
                [0, 0],
            ],
        )

        return embeddings

    def call(self, embeddings, mask):
        """
        Applies the layer.

        Args:
            embeddings (tensor): the node features (size B x N x Sum F_i)
                where B is the batch size, N is the number of nodes in the largest graph in the batch, and
                F_i is the dimensionality of node features output from the i-th convolutional layer.
            mask (tensor): a boolean mask (size B x N)
        Returns:
            Keras Tensor that represents the output of the layer.
        """

        outputs = tf.map_fn(
            self._sort_tensor_with_mask, (embeddings, mask), dtype=embeddings.dtype
        )

        # padding or truncation based on the value of self.k and the graph size (number of nodes)
        outputs_shape = tf.shape(outputs)

        outputs = tf.cond(
            tf.math.less(outputs_shape, self.k)[1],
            true_fn=lambda: tf.pad(
                outputs, [[0, 0], [0, (self.k - outputs_shape)[1]], [0, 0]]
            ),
            false_fn=lambda: outputs[:, : self.k, :],
        )

        if self.flatten_output:
            outputs = tf.reshape(
                outputs, [outputs_shape[0], embeddings.shape[-1] * self.k, 1]
            )

        return outputs
