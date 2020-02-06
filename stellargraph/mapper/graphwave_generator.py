import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from ..core import StellarGraph
from ..core.utils import GCN_Aadj_feats_op
from scipy.sparse.linalg import eigs


class GraphWaveGenerator:
    """
    Implementation of the GraphWave structural embedding algorithm from the paper:
        "Learning Structural Node Embeddings via Diffusion Wavelets" (https://arxiv.org/pdf/1710.10321.pdf)

    This class is minimially initialized with a StellarGraph object. Calling the flow function will return a tensorflow
    DataSet that contains the GraphWave embeddings.
    """

    def __init__(self, G, scales=[1.0,], num_eigenvecs=-1):
        """
        Args:
            G (StellarGraph): the StellarGraph object.
            scales (list of floats): the wavelet scales to use.
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
        self.Aadj = G.to_adjacency_matrix(self.node_list)

        # Function to map node IDs to indices for quicker node index lookups
        # TODO: Move this to the graph class
        node_index_dict = dict(zip(self.node_list, range(len(self.node_list))))
        self._node_lookup = np.vectorize(node_index_dict.get, otypes=[np.int64])

        Aadj = G.to_adjacency_matrix().tocoo()
        _, Aadj = GCN_Aadj_feats_op(None, Aadj)

        if num_eigenvecs == -1:
            num_eigenvecs = Aadj.shape[0] - 2

        self.eigen_vals, self.eigen_vecs = eigs(Aadj, k=num_eigenvecs)

        self.eigen_vals = np.real(self.eigen_vals).astype(np.float32)
        self.eigen_vecs = np.real(self.eigen_vecs).astype(np.float32)

        # TODO: add in option to automatically determine scales

        self.eUs = [
            np.diag(np.exp(s * self.eigen_vals)).dot(self.eigen_vecs.transpose())
            for s in scales
        ]
        self.eUs = tf.convert_to_tensor(np.dstack(self.eUs))

    def flow(self, node_ids, sample_points, batch_size, targets=None, repeat=True):
        """
        Creates a tensorflow DataSet object of GraphWave embeddings.

        Args:
            node_ids: an iterable of node ids for the nodes of interest
                (e.g., training, validation, or test set nodes)
            sample_points: a 1D array of points at which to sample the characteristic function.
            batch_size: the number of node embeddings to include in a batch.
            targets: a 1D or 2D array of numeric node targets with shape `(len(node_ids)`
                or (len(node_ids), target_size)`
            repeat (bool): indicates whether iterating through the DataSet will continue infinitely or stop after one
                full pass.

        """
        ts = tf.convert_to_tensor(sample_points.astype(np.float32))

        dataset = (
            tf.data.Dataset.from_tensor_slices(
                self.eigen_vecs[self._node_lookup(node_ids)]
            )
            .map(lambda x: tf.einsum("ijk,i->jk", self.eUs, x))
            .map(lambda x: _empirical_characteristic_function(x, ts))
        )

        if not targets is None:

            target_dataset = tf.data.Dataset.from_tensor_slices(targets)

            dataset = tf.data.Dataset.zip((dataset, target_dataset))

        if repeat:
            return dataset.batch(batch_size).repeat()
        else:
            return dataset.batch(batch_size)


def _empirical_characteristic_function(samples, ts):
    """
    This function estimates the characteristic function for the wavelet spread of a single node.

    Args:
        samples (Tensor): a tensor of samples drawn from a wavelet distribution at different scales.
        ts (Tensor): a tensor containing the "time" points to sample the characteristic function at.
    Returns:
        embedding (Tensor): the node embedding for the GraphWave algorithm.
    """

    samples = K.expand_dims(samples, 0)  # (ns, scales) -> (1, ns, scales)
    ts = K.expand_dims(K.expand_dims(ts, 1))  # (nt,) -> (nt, 1, 1)

    t_psi = (
        samples * ts
    )  # (1, ns, scales) * (nt, 1, 1) -> (nt, ns, scales) via broadcasting rules

    mean_cos_t_psi = tf.math.reduce_mean(
        tf.math.cos(t_psi), axis=1
    )  # (nt, ns, scales) -> (nt, scales)

    mean_sin_t_psi = tf.math.reduce_mean(
        tf.math.sin(t_psi), axis=1
    )  # (nt, ns, scales) -> (nt, scales)

    # [(nt, scales), (nt, scales)] -> (2 * nt * scales,)
    embedding = K.flatten(tf.concat([mean_cos_t_psi, mean_sin_t_psi], axis=0))

    return embedding
