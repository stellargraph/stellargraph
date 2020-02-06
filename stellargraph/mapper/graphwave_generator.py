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

        Aadj = G.to_adjacency_matrix().tocoo()
        _, Aadj = GCN_Aadj_feats_op(None, Aadj)

        if num_eigenvecs == -1:
            num_eigenvecs = Aadj.shape[0] - 2

        self.eigen_vals, self.eigen_vecs = eigs(Aadj, k=num_eigenvecs)

        self.eigen_vals = np.real(self.eigen_vals).astype(np.float32)
        self.eigen_vecs = np.real(self.eigen_vecs).astype(np.float32)

        self.eUs = [np.diag(np.exp(s * self.eigen_vals)).dot(self.eigen_vecs.transpose()) for s in scales]
        self.eUs = tf.convert_to_tensor(np.dstack(self.eUs))


    def flow(self, batch_size, targets=None, embed_dim=64, repeat=True):

        ts = tf.linspace(0.0, 10.0, (embed_dim // 2))

        dataset = tf.data.Dataset.from_tensor_slices(self.eigen_vecs).map(
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
