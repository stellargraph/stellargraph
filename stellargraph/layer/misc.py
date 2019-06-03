# -*- coding: utf-8 -*-
#
# Copyright 2018-2019 Data61, CSIRO
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

from keras.engine import Layer
from keras import backend as K

import tensorflow as tf


class SqueezedSparseConversion(Layer):
    """
    Converts Keras tensors containing indices and values to a tensorflow sparse
    tensor. The input tensors are expected to have a batch dimension of 1 which
    will be removed before conversion to a matrix.

    This only works with a tensorflow Keras backend.

    Example:
        ```
        A_indices = Input(batch_shape=(1, None, 2), dtype="int64")
        A_values = Input(batch_shape=(1, None))
        Ainput = TFSparseConversion(shape=(N, N))([A_indices, A_values])
        ```

    Args:
        shape (list of int): The shape of the sparse matrix to create
        dtype (str or tf.dtypes.DType): Data type for the created sparse matrix
    """

    def __init__(self, shape, dtype=None):
        super().__init__()

        self.trainable = False
        self.supports_masking = True
        self.matrix_shape = shape
        self.dtype = dtype

        # Check backend
        if K.backend() != "tensorflow":
            raise RuntimeError(
                "SqueezedSparseConversion only supports the Tensorflow backend"
            )

    def get_config(self):
        config = {"shape": self.matrix_shape, "dtype": self.dtype}
        return config

    def compute_output_shape(self, input_shapes):
        return tuple(self.matrix_shape)

    def call(self, inputs):
        """
        Applies the layer.

        Args:
            inputs (list): Two input tensors contining
                matrix indices (size 1 x E x 2) of type int64, and
                matrix values (size (size 1 x E),
                where E is the number of non-zero entries in the matrix.

        Returns:
            Tensorflow SparseTensor that represents the converted sparse matrix.
        """
        # Here we reduce the batch dimension (if 1)
        indices = K.squeeze(inputs[0], 0)
        values = K.squeeze(inputs[1], 0)

        if not self.dtype:
            dtype = K.dtype(values)

        # Build sparse tensor for the matrix
        output = tf.SparseTensor(
            indices=indices, values=values, dense_shape=self.matrix_shape
        )
        return output
