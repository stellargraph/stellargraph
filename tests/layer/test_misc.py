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


"""
GCN tests

"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.sparse as sps
import pytest
from stellargraph.layer.misc import *


def sparse_matrix_example(N=10, density=0.1):
    A = sps.rand(N, N, density=density, format="coo")
    A_indices = np.hstack((A.row[:, None], A.col[:, None]))
    A_values = A.data
    return A_indices, A_values, A


def test_squeezedsparseconversion():
    N = 10
    x_t = keras.Input(batch_shape=(1, N, 1), dtype="float32")
    A_ind = keras.Input(batch_shape=(1, None, 2), dtype="int64")
    A_val = keras.Input(batch_shape=(1, None), dtype="float32")

    # Test with final_layer=False
    A_mat = SqueezedSparseConversion(shape=(N, N), dtype=A_val.dtype)([A_ind, A_val])

    x_out = keras.layers.Lambda(
        lambda xin: K.expand_dims(K.dot(xin[0], K.squeeze(xin[1], 0)), 0)
    )([A_mat, x_t])

    model = keras.Model(inputs=[x_t, A_ind, A_val], outputs=x_out)

    x = np.random.randn(1, N, 1)
    A_indices, A_values, A = sparse_matrix_example(N)

    z = model.predict([x, np.expand_dims(A_indices, 0), np.expand_dims(A_values, 0)])

    assert np.allclose(z.squeeze(), A.dot(x.squeeze()), atol=1e-7)


def test_squeezedsparseconversion_dtype():
    N = 10
    x_t = keras.Input(batch_shape=(1, N, 1), dtype="float64")
    A_ind = keras.Input(batch_shape=(1, None, 2), dtype="int64")
    A_val = keras.Input(batch_shape=(1, None), dtype="float32")

    # Test with final_layer=False
    A_mat = SqueezedSparseConversion(shape=(N, N), dtype="float64")([A_ind, A_val])

    x_out = keras.layers.Lambda(
        lambda xin: K.expand_dims(K.dot(xin[0], K.squeeze(xin[1], 0)), 0)
    )([A_mat, x_t])

    model = keras.Model(inputs=[x_t, A_ind, A_val], outputs=x_out)

    x = np.random.randn(1, N, 1)
    A_indices, A_values, A = sparse_matrix_example(N)

    z = model.predict([x, np.expand_dims(A_indices, 0), np.expand_dims(A_values, 0)])

    assert A_mat.dtype == tf.dtypes.float64
    assert np.allclose(z.squeeze(), A.dot(x.squeeze()), atol=1e-7)


def test_squeezedsparseconversion_axis():
    N = 10
    A_indices, A_values, A = sparse_matrix_example(N)
    nnz = len(A_indices)

    A_ind = keras.Input(batch_shape=(nnz, 2), dtype="int64")
    A_val = keras.Input(batch_shape=(nnz, 1), dtype="float32")

    # Keras reshapes everything to have ndim at least 2, we need to flatten values
    A_val_1 = keras.layers.Lambda(lambda A: K.reshape(A, (-1,)))(A_val)

    # Test with final_layer=False
    A_mat = SqueezedSparseConversion(shape=(N, N), axis=None, dtype=A_val_1.dtype)(
        [A_ind, A_val_1]
    )

    ones = tf.ones((N, 1))

    x_out = keras.layers.Lambda(lambda xin: K.dot(xin, ones))(A_mat)

    model = keras.Model(inputs=[A_ind, A_val], outputs=x_out)
    z = model.predict([A_indices, A_values])

    assert np.allclose(z, A.sum(axis=1), atol=1e-7)
