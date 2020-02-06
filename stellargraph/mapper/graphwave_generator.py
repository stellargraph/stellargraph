import tensorflow as tf
import numpy as np
from ..core import StellarGraph
from ..core.utils import empirical_characteristic_function, GCN_Aadj_feats_op
from scipy.sparse.linalg import eigs


class GraphWaveGenerator:
    '''
    '''

    def __init__(self, G, filter_func=np.exp, num_eigenvecs=-1):
        '''
        '''

        if not isinstance(G, StellarGraph):
            raise TypeError("G must be a StellarGraph object.")

        Aadj = G.to_adjacency_matrix().tocoo()
        _, Aadj = GCN_Aadj_feats_op(None, Aadj)

        if num_eigenvecs == -1:
            num_eigenvecs = Aadj.shape[0] - 2

        self.eigen_vals, self.eigen_vecs = eigs(Aadj, k=num_eigenvecs)

        self.eigen_vals = np.real(self.eigen_vals).astype(np.float32)
        self.eigen_vecs = np.real(self.eigen_vecs).astype(np.float32)

        self.eU = np.diag(filter_func(self.eigen_vals)).dot(self.eigen_vecs.transpose())
        self.eU = tf.convert_to_tensor(self.eU)

    def flow(self, batch_size, targets=None, embed_dim=64, repeat=True):
        ts = tf.linspace(-10.0, 10.0, (embed_dim // 2))

        dataset = tf.data.Dataset.from_tensor_slices(self.eigen_vecs).batch(1).map(
            lambda x: tf.linalg.matmul(x, self.eU)
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
