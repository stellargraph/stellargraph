import tensorflow as tf
from tensorflow.keras import backend as K
from ..core import StellarGraph
from ..core.utils import partial_powers, select_row_from_sparse_tensor


class AdjacencyPowerGenerator:
    '''

    '''

    def __init__(self, G, num_powers=5):
        '''

        :param G:
        :param num_powers:
        '''

        if not isinstance(G, StellarGraph):
            raise TypeError("G must be a StellarGraph object.")

        nodelist = G.nodes()
        node_index = dict(zip(nodelist, range(len(nodelist))))

        edgelist = G.edges()
        indices = [[node_index[n2], node_index[n1]] for n1, n2 in edgelist]

        values = tf.ones(len(edgelist), tf.float32)

        self.Aadj_T = tf.sparse.SparseTensor(
            indices=indices, values=values,
            dense_shape=(len(nodelist), len(nodelist))
        )

        self.num_powers = num_powers

    def flow(self, batch_size):

        row_dataset = tf.data.Dataset.from_tensor_slices(tf.sparse.eye(int(self.Aadj_T.shape[0])))

        adj_powers_dataset = row_dataset.map(
            lambda ohe_rows: partial_powers(ohe_rows, self.Aadj_T),
            num_parallel_calls=10
        )

        row_index_dataset = tf.data.Dataset.range(self.Aadj_T.shape[0])

        row_index_adj_powers_dataset = tf.data.Dataset.zip((row_index_dataset, adj_powers_dataset))

        batch_adj_dataset = row_dataset.map(
            lambda ohe_rows: select_row_from_sparse_tensor(ohe_rows, self.Aadj_T),
            num_parallel_calls=10
        )

        training_dataset = tf.data.Dataset.zip(
            (row_index_adj_powers_dataset, batch_adj_dataset)
        ).batch(batch_size)

        return training_dataset.repeat()



