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
from tensorflow.keras.layers import Layer, Embedding, Input, Lambda, Concatenate, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, constraints, regularizers
import numpy as np
import warnings
from ..mapper.adjacency_generators import AdjacencyPowerGenerator
from ..core.validation import require_integer_in_range
from .misc import deprecated_model_function


class AttentiveWalk(Layer):
    """
    This implements the graph attention as in Watch Your Step: Learning Node Embeddings via Graph Attention
    https://arxiv.org/pdf/1710.09599.pdf.

    Args:
        walk_length (int): the length of the random walks. Equivalent to the number of adjacency powers used. Defaults
            to `10` as this value was found to perform well by the authors of the paper.
        attention_initializer (str or func, optional): The initialiser to use for the attention weights.
        attention_regularizer (str or func, optional): The regulariser to use for the attention weights.
        attention_constraint (str or func, optional): The constraint to use for the attention weights.
        input_dim (tuple of ints, optional): The shape of the input to the layer.
    """

    def __init__(
        self,
        walk_length=10,
        attention_initializer="glorot_uniform",
        attention_regularizer=None,
        attention_constraint=None,
        input_dim=None,
        **kwargs,
    ):

        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = input_dim

        self.walk_length = walk_length
        self.attention_initializer = initializers.get(attention_initializer)
        self.attention_regularizer = regularizers.get(attention_regularizer)
        self.attention_constraint = constraints.get(attention_constraint)
        super().__init__(**kwargs)

    def get_config(self):
        config = {
            "walk_length": self.walk_length,
            "attention_initializer": initializers.serialize(self.attention_initializer),
            "attention_regularizer": regularizers.serialize(self.attention_regularizer),
            "attention_constraint": constraints.serialize(self.attention_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][-1],)

    compute_output_shape.__doc__ = Layer.compute_output_shape.__doc__

    def build(self, input_shapes):

        self.attention_weights = self.add_weight(
            shape=(self.walk_length,),
            initializer=self.attention_initializer,
            name="attention_weights",
            regularizer=self.attention_regularizer,
            constraint=self.attention_constraint,
        )

        self.built = True

    build.__doc__ = Layer.build.__doc__

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
    to use for the node embeddings.

    Args:
        generator (AdjacencyPowerGenerator): the generator
        num_walks (int): the number of random walks starting at each node to use when calculating the expected random
            walks. Defaults to `80` as this value was found to perform well by the authors of the paper.
        embedding dimension (int): the dimension to use for the node embeddings (must be an even number).
        attention_initializer (str or func, optional): The initialiser to use for the attention weights.
        attention_regularizer (str or func, optional): The regulariser to use for the attention weights.
        attention_constraint (str or func, optional): The constraint to use for the attention weights.
        embeddings_initializer (str or func, optional): The initialiser to use for the embeddings.
        embeddings_regularizer (str or func, optional): The regulariser to use for the embeddings.
        embeddings_constraint (str or func, optional): The constraint to use for the embeddings.
    """

    def __init__(
        self,
        generator,
        num_walks=80,
        embedding_dimension=64,
        attention_initializer="glorot_uniform",
        attention_regularizer=None,
        attention_constraint=None,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        embeddings_constraint=None,
    ):

        if not isinstance(generator, AdjacencyPowerGenerator):
            raise TypeError(
                "generator should be an instance of AdjacencyPowerGenerator."
            )

        require_integer_in_range(num_walks, "num_walks", min_val=1)
        require_integer_in_range(embedding_dimension, "embedding_dimension", min_val=2)

        self.num_walks = num_walks
        self.num_powers = generator.num_powers
        self.n_nodes = int(generator.Aadj_T.shape[0])

        if embedding_dimension % 2 != 0:
            warnings.warn(
                f"embedding_dimension: expected even number, found odd number ({embedding_dimension}). It will be rounded down to {embedding_dimension - 1}.",
                stacklevel=2,
            )
            embedding_dimension -= 1

        self.embedding_dimension = embedding_dimension

        self._left_embedding = Embedding(
            self.n_nodes,
            int(self.embedding_dimension / 2),
            input_length=None,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint,
        )
        self._right_embedding = Dense(
            self.n_nodes,
            use_bias=False,
            kernel_initializer=embeddings_initializer,
            kernel_regularizer=embeddings_regularizer,
            kernel_constraint=embeddings_constraint,
        )
        self._attentive_walk = AttentiveWalk(
            walk_length=self.num_powers,
            attention_constraint=attention_constraint,
            attention_regularizer=attention_regularizer,
            attention_initializer=attention_initializer,
        )

    def embeddings(self):
        """
        This function returns the embeddings from a model with Watch Your Step embeddings.

        Returns:
            embeddings (np.array): a numpy array of the model's embeddings.
        """
        embeddings = np.hstack(
            [
                self._left_embedding.embeddings.numpy(),
                self._right_embedding.kernel.numpy().transpose(),
            ]
        )

        return embeddings

    def in_out_tensors(self):
        """
        This function builds the layers for a keras model.

        returns:
            A tuple of (inputs, outputs) to use with a keras model.
        """

        input_rows = Input(batch_shape=(None,), name="row_node_ids", dtype="int64")
        input_powers = Input(batch_shape=(None, self.num_powers, self.n_nodes))

        vectors_left = self._left_embedding(input_rows)

        # all right embeddings are used in every batch. to avoid unnecessary lookups the right embeddings are stored
        # in a dense layer to enable efficient dot product between the left vectors in the current batch and all right
        # vectors
        outer_product = self._right_embedding(vectors_left)

        expected_walk = self.num_walks * self._attentive_walk(input_powers)

        # layer to add batch dimension of 1 to output
        expander = Lambda(lambda x: K.expand_dims(x, axis=1))

        output = Concatenate(axis=1)([expander(expected_walk), expander(outer_product)])

        return [input_rows, input_powers], output

    build = deprecated_model_function(in_out_tensors, "build")
