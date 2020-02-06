import tensorflow as tf
import numpy as np
from ..core import StellarGraph
from ..core.utils import empirical_characteristic_function, GCN_Aadj_feats_op
from scipy.sparse.linalg import eigs


class GraphWaveGenerator:
    '''
    '''

    def __init__(self, G, scales=[1.0,], num_eigenvecs=-1):
        '''
        '''

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

        self.eUs = [np.diag(np.exp(s * self.eigen_vals)).dot(self.eigen_vecs.transpose()) for s in scales]
        self.eUs = tf.convert_to_tensor(np.dstack(self.eUs))

    def flow(self, node_ids, sample_points, batch_size, targets=None, repeat=True):

        ts = tf.convert_to_tensor(sample_points.astype(np.float32))

        dataset = tf.data.Dataset.from_tensor_slices(self.eigen_vecs[self._node_lookup(node_ids)]).map(
            lambda x: tf.einsum('ijk,i->jk', self.eUs, x)
        ).map(
            lambda x: empirical_characteristic_function(x, ts)
        )

        if not targets is None:

            target_dataset = tf.data.Dataset.from_tensor_slices(targets)

            dataset = tf.data.Dataset.zip(
                (dataset, target_dataset)
            )

        if repeat:
            return dataset.batch(batch_size).repeat()
        else:
            return dataset.batch(batch_size)
