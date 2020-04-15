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

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import warnings


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

    def __init__(self, shape, axis=0, dtype=None):
        super().__init__(dtype=dtype)

        self.trainable = False
        self.supports_masking = True
        self.matrix_shape = shape
        # self.dtype = dtype
        self.axis = axis

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
        Creates a Tensorflow `SparseTensor` from the inputs

        Args:
            inputs (list): Two input tensors contining
                matrix indices (size 1 x E x 2) of type int64, and
                matrix values (size (size 1 x E),
                where E is the number of non-zero entries in the matrix.

        Returns:
            Tensorflow SparseTensor that represents the converted sparse matrix.
        """
        # Here we squeeze the specified axis
        if self.axis is not None:
            indices = K.squeeze(inputs[0], self.axis)
            values = K.squeeze(inputs[1], self.axis)
        else:
            indices = inputs[0]
            values = inputs[1]

        if self.dtype is not None:
            values = K.cast(values, self.dtype)

        # Import tensorflow here so that the backend check will work without
        # tensorflow installed.
        import tensorflow as tf

        # Build sparse tensor for the matrix
        output = tf.SparseTensor(
            indices=indices, values=values, dense_shape=self.matrix_shape
        )
        return output


class GatherIndices(Layer):
    """
    Gathers slices from a data tensor, based on an indices tensors (``tf.gather`` in Layer form).

    Args:
        axis (int or Tensor): the data axis to gather from.
        batch_dims (int): the number of batch dimensions in the data and indices.
    """

    def __init__(self, axis=None, batch_dims=0, **kwargs):
        super().__init__(**kwargs)
        self._axis = axis
        self._batch_dims = batch_dims

    def get_config(self):
        config = super().get_config()
        config.update(axis=self._axis, batch_dims=self._batch_dims)
        return config

    def compute_output_shape(self, input_shapes):
        data_shape, indices_shape = input_shapes
        axis = self._batch_dims if self._axis is None else self._axis
        # per https://www.tensorflow.org/api_docs/python/tf/gather
        return (
            data_shape[:axis]
            + indices_shape[self._batch_dims :]
            + data_shape[axis + 1 :]
        )

    def call(self, inputs):
        """
        Args:
            inputs (list): a pair of tensors, corresponding to the ``params`` and ``indices``
                parameters to ``tf.gather``.
        """
        data, indices = inputs
        return tf.gather(data, indices, axis=self._axis, batch_dims=self._batch_dims)


def deprecated_model_function(function, old_name):
    def _function_wrapper(*args, **kwargs):
        """Deprecated: use :meth:`in_out_tensors`."""

        warnings.warn(
            f"The '{old_name}' method is deprecated, use 'in_out_tensors' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return function(*args, **kwargs)

    return _function_wrapper
