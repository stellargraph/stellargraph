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
    issues=[815, 853]
)
class GraphWaveGenerator:
    """
    Implementation of the GraphWave structural embedding algorithm from the paper:
        "Learning Structural Node Embeddings via Diffusion Wavelets" (https://arxiv.org/pdf/1710.10321.pdf)

    This class is minimally initialized with a StellarGraph object. Calling the flow function will return a tensorflow
    DataSet that contains the GraphWave embeddings.
    """

    def __init__(self, G, scales="auto", num_scales=3, num_eigenvecs=-1):
        """
        Args:
            G (StellarGraph): the StellarGraph object.
            scales (str or list of floats): the wavelet scales to use. "auto" will cause the scale values to be
                automatically calculated.
            num_scales (int): the number of scales when scales = "auto".
            num_eigenvecs (int): the number of eigenvectors to use. When set to `-1` the maximum number of eigenvectors
                is calculated.
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
        # Use the node orderings the same as in the graph features
        self.node_list = G.nodes_of_type(node_types[0])
        adj = G.to_adjacency_matrix(self.node_list).tocoo()

        # Function to map node IDs to indices for quicker node index lookups
        # TODO: Move this to the graph class
        node_index_dict = dict(zip(self.node_list, range(len(self.node_list))))
        self._node_lookup = np.vectorize(node_index_dict.get, otypes=[np.int64])

        degree_mat = diags(np.array(adj.sum(1)).flatten())
        laplacian = degree_mat - adj

        if num_eigenvecs == -1:
            num_eigenvecs = laplacian.shape[0] - 2

        # TODO: add in option to compute wavelet transform using Chebysev polynomials
        eigen_vals, eigen_vecs = eigs(laplacian, k=num_eigenvecs)
        eigen_vals = np.real(eigen_vals).astype(np.float32)
        self.eigen_vecs = np.real(eigen_vecs).astype(np.float32)

        if scales == "auto":

            e2 = eigen_vals[eigen_vals > 0].min()
            eN = eigen_vals.max()

            min_scale = -np.log(0.95) / np.sqrt(eN * e2)
            max_scale = -np.log(0.85) / np.sqrt(eN * e2)

            scales = np.linspace(min_scale, max_scale, num_scales)

        self.scales = scales

        # the columns of U exp(-scale * eigenvalues) U^T (U = eigenvectors) are used to calculate the node embeddings
        # (each column corresponds to a node)
        # to avoid computing a dense NxN matrix when only several eigenvalues are specified
        # U exp(-scale * eigenvalues) is computed and stored - which is an N x num_eigenvectors matrix
        # the columns of U exp(-scale * eigenvalues) U^T are then computed on the fly in generator.flow()
        Ues = [
            self.eigen_vecs.dot(np.diag(np.exp(-s * eigen_vals)))[np.newaxis, :] for s in scales
        ]  # a list of [U exp(-scale * eigenvalues) for scale in scales]
        self.Ues = tf.convert_to_tensor(np.concatenate(Ues, axis=0))

    def flow(
        self, node_ids, sample_points, batch_size, targets=None, repeat=True, num_parallel_calls=1
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

        dataset = tf.data.Dataset.from_tensor_slices(self.eigen_vecs[self._node_lookup(node_ids)])

        # calculates the columns of U exp(-scale * eigenvalues) U^T on the fly
        # and empirically the characteristic function for each column of U exp(-scale * eigenvalues) U^T
        dataset = dataset.map(lambda x: _empirical_characteristic_function(tf.linalg.matvec(self.Ues, x), ts))

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
