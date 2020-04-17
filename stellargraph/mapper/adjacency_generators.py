# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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
import numpy as np
from ..core import StellarGraph
from ..core.validation import require_integer_in_range
from ..core.utils import normalize_adj
from .base import Generator


class AdjacencyPowerGenerator(Generator):
    """
    A data generator for use with the Watch Your Step algorithm [1]. It calculates and returns the first `num_powers`
    of the adjacency matrix row by row.

    Args:
        G (StellarGraph): a machine-learning StellarGraph-type graph
        num_powers (int): the number of adjacency powers to calculate. Defaults
            to `10` as this value was found to perform well by the authors of the paper.

    """

    def __init__(self, G, num_powers=10):

        if not isinstance(G, StellarGraph):
            raise TypeError("G must be a StellarGraph object.")

        require_integer_in_range(num_powers, "num_powers", min_val=1)

        Aadj = G.to_adjacency_matrix().tocoo()
        indices = np.column_stack((Aadj.col, Aadj.row))

        self.Aadj_T = tf.sparse.SparseTensor(
            indices=indices,
            values=Aadj.data.astype(np.float32),
            dense_shape=Aadj.shape,
        )

        self.transition_matrix_T = tf.sparse.SparseTensor(
            indices=indices,
            values=normalize_adj(Aadj, symmetric=False).data.astype(np.float32),
            dense_shape=Aadj.shape,
        )

        self.num_powers = num_powers

    def num_batch_dims(self):
        return 1

    def flow(self, batch_size, num_parallel_calls=1):
        """
        Creates the `tensorflow.data.Dataset` object for training node embeddings from powers of the adjacency matrix.

        Args:
            batch_size (int): the number of rows of the adjacency powers to include in each batch.
            num_parallel_calls (int): the number of threads to use for pre-processing of batches.

        Returns:
            A `tensorflow.data.Dataset` object for training node embeddings from powers of the adjacency matrix.
        """

        require_integer_in_range(batch_size, "batch_size", min_val=1)
        require_integer_in_range(num_parallel_calls, "num_parallel_calls", min_val=1)

        row_dataset = tf.data.Dataset.from_tensor_slices(
            tf.sparse.eye(int(self.Aadj_T.shape[0]))
        )

        adj_powers_dataset = row_dataset.map(
            lambda ohe_rows: _partial_powers(
                ohe_rows, self.transition_matrix_T, num_powers=self.num_powers
            ),
            num_parallel_calls=num_parallel_calls,
        )

        row_index_dataset = tf.data.Dataset.range(self.Aadj_T.shape[0])

        row_index_adj_powers_dataset = tf.data.Dataset.zip(
            (row_index_dataset, adj_powers_dataset)
        )

        batch_adj_dataset = row_dataset.map(
            lambda ohe_rows: _select_row_from_sparse_tensor(ohe_rows, self.Aadj_T),
            num_parallel_calls=num_parallel_calls,
        )

        training_dataset = tf.data.Dataset.zip(
            (row_index_adj_powers_dataset, batch_adj_dataset)
        ).batch(batch_size)

        return training_dataset.repeat()


def _partial_powers(one_hot_encoded_row, Aadj_T, num_powers):
    """
    This function computes the first num_powers powers of the adjacency matrix
    for the row specified in one_hot_encoded_row

    Args:
        one_hot_encoded_row: one-hot-encoded row
        Aadj_T: the transpose of the adjacency matrix
        num_powers (int): the adjacency number of powers to compute

    returns:
        A matrix of the shape (num_powers, Aadj_T.shape[1]) of
        the specified row of the first num_powers of the adjacency matrix.
    """

    # make sure the transpose of the adjacency is used
    # tensorflow requires that the sparse matrix is the first operand

    partial_power = tf.reshape(
        tf.sparse.to_dense(one_hot_encoded_row), shape=(1, Aadj_T.shape[1])
    )
    partial_powers_list = []
    for i in range(num_powers):

        partial_power = K.transpose(K.dot(Aadj_T, K.transpose(partial_power)))
        partial_powers_list.append(partial_power)

    return K.squeeze(tf.stack(partial_powers_list, axis=1), axis=0)


def _select_row_from_sparse_tensor(one_hot_encoded_row, sp_tensor_T):
    """
    This function gathers the row specified in one_hot_encoded_row from the input sparse matrix

    Args:
        one_hot_encoded_row: one-hot-encoded row
        sp_tensor_T: the transpose of the sparse matrix

    returns:
        The specified row from sp_tensor_T.
    """
    one_hot_encoded_row = tf.reshape(
        tf.sparse.to_dense(one_hot_encoded_row), shape=(1, sp_tensor_T.shape[1])
    )
    row_T = K.dot(sp_tensor_T, K.transpose(one_hot_encoded_row))
    row = K.transpose(row_T)
    return row
