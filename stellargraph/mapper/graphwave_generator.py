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
from scipy.sparse.linalg import eigs
from scipy.sparse import diags
from ..core.experimental import experimental


@experimental(
    reason="lacks unit tests, and the time complexity could be reduced using Chebyshev polynomials.",
    issues=[815, 853],
)
class GraphWaveGenerator:
    """
    Implementation of the GraphWave structural embedding algorithm from the paper:
        "Learning Structural Node Embeddings via Diffusion Wavelets" (https://arxiv.org/pdf/1710.10321.pdf)

    This class is minimally with a StellarGraph object. Calling the flow function will return a tensorflow
    DataSet that contains the GraphWave embeddings.

    This implementation differs from the paper by removing the automatic method of calculating scales. This method was
    found to not work well in practice, and replicating the results of the paper requires manually specifying much larger
    scales than those automatically calculated.
    """

    # This code looks for small eigenvalues of the graph Laplacian. The minimum eigenvalue of the graph Laplacian
    # is always 0, however numerical errors can cause small negative number as large as -1e-5.
    # To initialize a search that will reliably include the smallest eigenvalue an initial value
    # of -1e-3 is used
    _INITIAL_EIGS_SIGMA = -1e-3

    def __init__(self, G, scales=(5, 10), deg=20):
        """
        Args:
            G (StellarGraph): the StellarGraph object.
            scales (iterable of floats): the wavelet scales to use. Smaller values embed smaller scale structural
                features, and larger values embed larger structural features.
            num_eigenvecs (int): the number of eigenvectors to use. When set to `None` the number of eigenvectors
                is automatically determined.
            min_delta (float): when `num_eigenvecs=None` this controls the error of the GraphWave approximation.
                A small `min_delta` will result in a better approximation but is more computationally expensive and
                vice verse.
        """

        if not isinstance(G, StellarGraph):
            raise TypeError("G must be a StellarGraph object.")

        node_types = list(G.node_types)
        if len(node_types) > 1:
            raise TypeError(
                "{}: node generator requires graph with single node type; "
                "a graph with multiple node types is passed. Stopping.".format(
                    type(self).__name__
                )
            )

        # Create sparse adjacency matrix:
        adj = G.to_adjacency_matrix().tocoo()

        # Function to map node IDs to indices for quicker node index lookups
        self._node_lookup = G._get_index_for_nodes
        self.deg = deg

        degree_mat = diags(np.asarray(adj.sum(1)).ravel())
        laplacian = degree_mat - adj
        laplacian = laplacian.tocoo()

        self.scales = np.array(scales).astype(np.float32)

        max_eig = eigs(laplacian, k=1, return_eigenvectors=False)
        max_eig = np.real(max_eig).astype(np.float32)[0]
        xs = np.linspace(0, max_eig, 100)

        coeffs = [np.polynomial.chebyshev.chebfit(xs, np.exp(-scale * xs), deg=self.deg) for scale in scales]
        self.coeffs = tf.convert_to_tensor(np.stack(coeffs, axis=0).astype(np.float32))

        self.laplacian = tf.sparse.SparseTensor(
            indices=np.column_stack((laplacian.row, laplacian.col)),
            values=laplacian.data.astype(np.float32),
            dense_shape=laplacian.shape,
        )

    def flow(
        self,
        node_ids,
        sample_points,
        batch_size,
        targets=None,
        repeat=True,
        num_parallel_calls=1,
    ):
        """
        Creates a tensorflow DataSet object of GraphWave embeddings.

        Args:
            node_ids: an iterable of node ids for the nodes of interest
                (e.g., training, validation, or test set nodes)
            sample_points: a 1D array of points at which to sample the characteristic function. This should be of the
                form: `sample_points=np.linspace(0, max_val, number_of_samples)` and is graph dependant.
            batch_size: the number of node embeddings to include in a batch.
            targets: a 1D or 2D array of numeric node targets with shape `(len(node_ids)`
                or (len(node_ids), target_size)`
            repeat (bool): indicates whether iterating through the DataSet will continue infinitely or stop after one
                full pass.
            num_parallel_calls (int): number of threads to use.
        """
        ts = tf.convert_to_tensor(sample_points.astype(np.float32))

        # calculates the columns of U exp(-scale * eigenvalues) U^T on the fly
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.sparse.eye(int(self.laplacian.shape[0]))
        ).map(
            lambda x: _chebyshev(x, self.laplacian, self.coeffs, self.deg),
            num_parallel_calls=num_parallel_calls,
        )

        # empirically calculate the characteristic function for each column of U exp(-scale * eigenvalues) U^T
        dataset = dataset.map(
            lambda x: _empirical_characteristic_function(x, ts),
            num_parallel_calls=num_parallel_calls,
        )

        if not targets is None:

            target_dataset = tf.data.Dataset.from_tensor_slices(targets)

            dataset = tf.data.Dataset.zip((dataset, target_dataset))

        # cache embeddings in memory for performance
        if repeat:
            return dataset.cache().batch(batch_size).repeat()
        else:
            return dataset.cache().batch(batch_size)


def _empirical_characteristic_function(samples, ts):
    """
    This function estimates the characteristic function for the wavelet spread of a single node.

    Args:
        samples (Tensor): a tensor of samples drawn from a wavelet distribution at different scales.
        ts (Tensor): a tensor containing the "time" points to sample the characteristic function at.
    Returns:
        embedding (Tensor): the node embedding for the GraphWave algorithm.
    """
    # (scales, ns) -> (1, scales, ns)
    samples = samples[tf.newaxis, :, :]
    # (nt,) -> (nt, 1, 1)
    ts = ts[:, tf.newaxis, tf.newaxis]

    # (1, scales, ns) * (nt, 1, 1) -> (nt, scales, ns) via broadcasting rules
    t_psi = samples * ts

    # (nt, scales, ns) -> (nt, scales)
    mean_cos_t_psi = tf.math.reduce_mean(tf.math.cos(t_psi), axis=2)

    # (nt, scales, ns) -> (nt, scales)
    mean_sin_t_psi = tf.math.reduce_mean(tf.math.sin(t_psi), axis=2)

    # [(nt, scales), (nt, scales)] -> (2 * nt * scales,)
    embedding = K.flatten(tf.concat([mean_cos_t_psi, mean_sin_t_psi], axis=0))

    return embedding


def _chebyshev(one_hot_encoded_row, laplacian, coeffs, deg):
    """
    This function calculates one column of the Chebyshev approximation of exp(-scale * laplacian) for
    all scales.

    Args:
        one_hot_encoded_row (SparseTensor): a sparse tensor indicating which column to calculate.
        laplacian (SparseTensor): the unormalized graph laplacian
        coeffs: the Chebyshev coefficients for exp(-scale * x) for each scale in the shape (num_scales, deg)
        deg: the degree of the Chebyshev polynomial
    Returns:
        Tensor
    """
    T_0 = tf.reshape(
        tf.sparse.to_dense(one_hot_encoded_row), shape=(laplacian.shape[0], 1)
    )
    T_1 = K.dot(laplacian, T_0)
    cheby_polys = [T_0, T_1]
    for i in range(deg - 1):
        cheby_poly = 2 * K.dot(laplacian, cheby_polys[-1]) - cheby_polys[-2]
        cheby_polys.append(cheby_poly)

    cheby_polys = K.squeeze(tf.stack(cheby_polys, axis=0), axis=-1)
    return tf.matmul(coeffs, cheby_polys)

