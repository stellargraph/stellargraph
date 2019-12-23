import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Input, Lambda, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, initializers, constraints
import numpy as np

from ..mapper.adjacency_generators import AdjacencyPowerGenerator

class AttentiveWalk(Layer):
    """
    This implements the graph attention as in Watch Your Step: Learning Node Embeddings via Graph Attention
    https://arxiv.org/pdf/1710.09599.pdf.

    Args:
        walk_length (int): the length of the random walks. Equivalent to the number of adjacency powers used.
        attention_initializer (str or func): The initialiser to use for the attention weights;
            defaults to 'glorot_uniform'.
        attention_regularizer (str or func): The regulariser to use for the attention weights;
            defaults to None.
        attention_constraint (str or func): The constraint to use for the attention weights;
            defaults to None.
    """
    def __init__(self, walk_length=5, **kwargs):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.get("input_dim"),)

        self.walk_length = walk_length

        self._get_regularisers_from_keywords(kwargs)
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.
        Assumes the following inputs:

        Args:
            input_shapes (tuple of ints)
                Shape tuples can include None for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """
        return (input_shapes[0][-1],)

    def _get_regularisers_from_keywords(self, kwargs):
        self.attention_initializer = initializers.get(
            kwargs.pop("attention_initializer", "glorot_uniform")
        )

        self.attention_regularizer = regularizers.get(
            kwargs.pop("attention_regularizer", None)
        )

        self.attention_constraint = constraints.get(kwargs.pop("attention_constraint", None))

    def build(self, input_shapes):
        """
        Builds the layer

        Args:
            input_shapes (list of int): shapes of the layer's inputs (node features and adjacency matrix)

        """

        self.attention_weights = self.add_weight(
            shape=(self.walk_length,),
            initializer=self.attention_initializer,
            name="attention_weights",
            regularizer=self.attention_regularizer,
            constraint=self.attention_constraint,
        )

        self.built = True

    def call(self, partial_powers):
        """
        Applies the layer and calculates the expected random walks.

        Args:
            partial_powers: num_rows rows of the first num_powers powers of adjacency matrix with shape
            (num_rows, num_powers, num_nodes)

        Returns:
            Tensor that represents the expected random walks starting from nodes corresponding to the input rows of
            shape (num_rows, num_nodes)
        """

        attention = K.softmax(self.attention_weights)
        expected_walk = tf.einsum("ijk,j->ik", partial_powers, attention)

        return expected_walk


class WatchYourStep:
    """
    Implementation of the node embeddings as in Watch Your Step: Learning Node Embeddings via Graph Attention
    https://arxiv.org/pdf/1710.09599.pdf.

    This model requires specification of the number of random walks starting from each node, and the embedding dimension
    to use for the node embeddings. Note, that the embedding dimension should be an even number or else will be
    rounded down to nearest even number.

    Args:
        generator (AdjacencyPowerGenerator): the generator
        num_walks (int): the number of random walks starting at each node to use when calculating the expected random
        walks.
        embedding dimension (int): the dimension to use for the node embeddings
    """

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
        """
        This function builds the layers for a keras model.

        returns:
            A tuple of (inputs, outputs) to use with a keras model.
        """

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
        expected_walk = AttentiveWalk(walk_length=self.num_powers)(input_powers)

        expander = Lambda(lambda x: K.expand_dims(x, axis=1))

        output = Concatenate(axis=1)([expander(expected_walk), expander(sigmoids)])

        return [input_rows, input_powers],  output
