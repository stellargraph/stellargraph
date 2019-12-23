import tensorflow as tf
from tensorflow.keras import backend as K



def partial_powers(one_hot_encoded_row, Aadj_T, num_powers=5):
    '''
    This function computes the first num_powers powers of the adjacency matrix
    for the row specified in one_hot_encoded_row

    Args:
        one_hot_encoded_row: one-hot-encoded row
        Aadj_T: the transpose of the adjacency matrix
        num_powers (int): the adjacency number of powers to compute

    returns:
        A matrix of the shape (num_powers, Aadj_T.shape[1]) of
        the specified row of the first num_powers of the adjacency matrix.
    '''

    # make sure the transpose of the adjacency is used
    # tensorflow requires that the sparse matrix is the first operand

    batch_adj = select_row_from_sparse_tensor(one_hot_encoded_row, Aadj_T)

    partial_powers_list = [batch_adj]

    for i in range(1, num_powers):

        partial_power = K.dot(Aadj_T, K.transpose(partial_powers_list[-1]))
        partial_power = K.transpose(partial_power)
        partial_powers_list.append(partial_power)

    return K.squeeze(tf.stack(partial_powers_list, axis=1), axis=0)


def select_row_from_sparse_tensor(one_hot_encoded_row, sp_tensor_T):
    '''
    This function gathers the row specified in one_hot_encoded_row from the input sparse matrix

    Args:
        one_hot_encoded_row: one-hot-encoded row
        sp_tensor_T: the transpose of the sparse matrix

    returns:
        The specified row from sp_tensor_T.
    '''
    one_hot_encoded_row = tf.reshape(tf.sparse.to_dense(one_hot_encoded_row), shape=(1, sp_tensor_T.shape[1]))
    row_T = K.dot(sp_tensor_T, K.transpose(one_hot_encoded_row))
    row = K.transpose(row_T)
    return row