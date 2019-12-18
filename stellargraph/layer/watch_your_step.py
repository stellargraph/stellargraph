import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Input, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, initializers, constraints
import numpy as np

from ..mapper.adjacency_power_generators import AdjacencyPowerGenerator

class AttentiveWalk(Layer):

    def __init__(self, walk_length=5, **kwargs):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.get("input_dim"),)

        self.walk_length = walk_length

        self._get_regularisers_from_keywords(kwargs)
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][-1],)

    def _get_regularisers_from_keywords(self, kwargs):
        self.kernel_initializer = initializers.get(
            kwargs.pop("kernel_initializer", "glorot_uniform")
        )

        self.kernel_regularizer = regularizers.get(
            kwargs.pop("kernel_regularizer", None)
        )

        self.kernel_constraint = constraints.get(kwargs.pop("kernel_constraint", None))

    def build(self, input_shapes):
        """
        """

        self.attention_weights = self.add_weight(
            shape=(self.walk_length,),
            initializer=self.kernel_initializer,
            name="attention_weights",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        self.built = True

    def call(self, inputs):
        sigmoids, partial_powers = inputs

        attention = K.softmax(self.attention_weights)
        expected_walk = tf.einsum("ijk,j->ik", partial_powers, attention)

        # batch_adj = K.squeeze(tf.gather(partial_powers, indices=[0], axis=1), axis=1)
        # adj_mask = tf.cast((batch_adj == 0), 'float32')
        #
        # # calculate loss here
        # loss = K.mean(K.abs(-((expected_walk * K.log(sigmoids + 1e-6))) -
        #                     adj_mask * K.log(1 - sigmoids + 1e-6)))
        # loss = tf.expand_dims(loss, 0)

        return expected_walk


class WatchYourStep:

    def __init__(self, generator, num_walks, embedding_dimension):

        if not isinstance(generator, AdjacencyPowerGenerator):
            raise TypeError("generator should be an instance of AdjacencyPowerGenerator.")

        if not isinstance(num_walks, int):
            raise TypeError("num_walks should be an int.")

        if num_walks <= 0:
            raise ValueError("num_walks should be a positive int.")

        self.num_powers = generator.num_powers
        self.n_nodes = int(generator.Aadj_T.shape[0])
        self.embedding_dimension = embedding_dimension


    def build(self):

        input_rows = Input(batch_shape=(None,), name='row_node_ids', dtype='int64')
        input_powers = Input(batch_shape=(None, self.num_powers, self.n_nodes))

        input_cols = Lambda(lambda x: tf.constant(np.arange(int(self.n_nodes)), dtype='int64'))(input_rows)

        left_embedding = Embedding(
            self.n_nodes, self.embedding_dimension,
            input_length=None, name='LEFT_EMBEDDINGS'
        )

        right_embedding = Embedding(
            self.n_nodes, self.embedding_dimension,
            input_length=None, name='RIGHT_EMBEDDINGS'
        )

        vectors_left = Lambda(lambda x: K.transpose(x))(left_embedding(input_rows))
        vectors_right = right_embedding(input_cols)

        dot_product = Lambda(lambda x: K.transpose(K.dot(x[0], x[1])))([vectors_right, vectors_left])

        sigmoids = tf.keras.activations.sigmoid(dot_product)
        expected_walk = AttentiveWalk(walk_length=self.num_powers)([sigmoids, input_powers])

        return [input_rows, input_powers], expected_walk





def full_embedding_model(batch_rows, vector_dim, vocab_size, walk_length):
    input_rows = Input(batch_shape=(1, batch_rows), name='row_node_ids', dtype='int64')
    input_powers = Input(batch_shape=(1, batch_rows, walk_length, int(Aadj_T.shape[0])))

    all_inputs = [input_rows, input_powers]

    rows = K.squeeze(input_rows, axis=0)
    powers = K.squeeze(input_powers, axis=0)

    # connect to input to satisfy keras API - input_cols is actually constant
    input_cols = Lambda(lambda x: tf.constant(np.arange(int(Aadj_T.shape[0])), dtype='int64'))(input_rows)

    left_embedding = Embedding(
        vocab_size, vector_dim,
        input_length=None, name='LEFT_EMBEDDINGS'
    )

    right_embedding = Embedding(
        vocab_size, vector_dim,
        input_length=None, name='RIGHT_EMBEDDINGS'
    )

    vectors_left = Lambda(lambda x: K.transpose(x))(left_embedding(rows))
    vectors_right = right_embedding(input_cols)

    dot_product = Lambda(lambda x: K.transpose(K.dot(x[0], x[1])))([vectors_right, vectors_left])

    sigmoids = tf.keras.activations.sigmoid(dot_product)

    loss = AttentiveWalk(walk_length=walk_length)([sigmoids, powers])

    model = Model(inputs=all_inputs, outputs=loss)

    return model


# def graph_log_liklihood(partial_powers, output):
#     sigmoids, attention = y_pred
#     expected_walk = tf.einsum("bijk,bj->bik", partial_powers, attention)
#
#     batch_adj = K.squeeze(tf.gather(partial_powers, indices=[0], axis=1), axis=1)
#     adj_mask = tf.cast((batch_adj == 0), 'float32')
#
#     # calculate loss here
#     loss = K.mean(K.abs(-((expected_walk * K.log(sigmoids + 1e-6))) -
#                         adj_mask * K.log(1 - sigmoids + 1e-6)))
#     loss = tf.expand_dims(loss, 0)
#
#     return loss


# def identity_loss(y_true, y_pred):
#     return y_true * y_pred