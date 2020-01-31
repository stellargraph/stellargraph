import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from ..core import StellarGraph
from ..core.utils import normalize_adj
from ..core.tf_utils import partial_powers, select_row_from_sparse_tensor
from ..core.experimental import experimental


@experimental(reason="lack of unit tests")
class AdjacencyPowerGenerator:
    """
    A data generator for use with the Watch Your Step algorithm [1]. It calculates and returns the first `num_powers`
    of the adjacency matrix row by row.

    .. warning::

        This class is experimental: it is insufficiently tested.

    Args:
        G (StellarGraph): a machine-learning StellarGraph-type graph
        num_powers (int): the number of adjacency powers to calculate

    """

    def __init__(self, G, num_powers=5):

        if not isinstance(G, StellarGraph):
            raise TypeError("G must be a StellarGraph object.")

        Aadj = G.to_adjacency_matrix()
        indices = np.column_stack((Aadj.col, Aadj.row))

        self.Aadj_T = tf.sparse.SparseTensor(
            indices=indices,
            values=Aadj.data.astype(np.float32),
            dense_shape=(Aadj.shape[0], Aadj.shape[0]),
        )

        self.transition_matrix_T = tf.sparse.SparseTensor(
            indices=indices,
            values=normalize_adj(Aadj, symmetric=False).data.astype(np.float32),
            dense_shape=(Aadj.shape[0], Aadj.shape[0]),
        )

        self.num_powers = num_powers

    def flow(self, batch_size, threads=1):
        """
        Creates the `tensorflow.data.Dataset` object for training node embeddings from powers of the adjacency matrix.

        Args:
            batch_size (int): the number of rows of the adjacency powers to include in each batch.

        Returns:
            A `tensorflow.data.Dataset` object for training node embeddings from powers of the adjacency matrix.
        """
        row_dataset = tf.data.Dataset.from_tensor_slices(
            tf.sparse.eye(int(self.Aadj_T.shape[0]))
        )

        adj_powers_dataset = row_dataset.map(
            lambda ohe_rows: partial_powers(
                ohe_rows, self.transition_matrix_T, num_powers=self.num_powers
            ),
            num_parallel_calls=threads,
        )

        row_index_dataset = tf.data.Dataset.range(self.Aadj_T.shape[0])

        row_index_adj_powers_dataset = tf.data.Dataset.zip(
            (row_index_dataset, adj_powers_dataset)
        )

        batch_adj_dataset = row_dataset.map(
            lambda ohe_rows: select_row_from_sparse_tensor(ohe_rows, self.Aadj_T),
            num_parallel_calls=threads,
        )

        training_dataset = tf.data.Dataset.zip(
            (row_index_adj_powers_dataset, batch_adj_dataset)
        ).batch(batch_size)

        return training_dataset.repeat()
