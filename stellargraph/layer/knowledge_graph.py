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
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Layer, Lambda, Dropout, Reshape, Embedding

from ..mapper.knowledge_graph import KGTripleGenerator


@experimental(reason="results from the reference paper have not been reproduced yet")
class ComplExScore(Layer):
    """
    ComplEx scoring Keras layer.

    Original Paper: Complex Embeddings for Simple Link Prediction, Théo Trouillon, Johannes Welbl,
    Sebastian Riedel, Éric Gaussier and Guillaume Bouchard, ICML
    2016. http://jmlr.org/proceedings/papers/v48/trouillon16.pdf

    This combines subject, relation and object embeddings into a score of the likelihood of the
    link.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        s_re, s_im, r_re, r_im, o_re, o_im = inputs

        def inner(r, s, o):
            return tf.reduce_sum(r * s * o, axis=2)

        # expansion of Re(<w_r, e_s, conjugate(e_o)>)
        score = (
            inner(r_re, s_re, o_re)
            + inner(r_re, s_im, o_im)
            + inner(r_im, s_re, o_im)
            - inner(r_im, s_im, o_re)
        )

        return score


@experimental(reason="results from the reference paper have not been reproduced yet")
class ComplEx:
    """
    Embedding layers and a ComplEx scoring layers that implement the ComplEx knowledge graph
    embedding algorithm as in http://jmlr.org/proceedings/papers/v48/trouillon16.pdf

    Args:
        generator (KGTripleGenerator): A generator of triples to feed into the model.

        k (int): the dimension of the embedding (that is, a vector in C^k is learnt for each node
            and each link type)

        embedding_initializer (str or func, optional): The initialiser to use for the embeddings.

        embedding_regularizer (str or func, optional): The regularizer to use for the embeddings.
    """

    def __init__(
        self, generator, k, embedding_initializer=None, embedding_regularizer=None,
    ):
        if not isinstance(generator, KGTripleGenerator):
            raise TypeError(
                f"generator: expected KGTripleGenerator, found {type(generator).__name__}"
            )

        graph = generator.G
        self.num_nodes = graph.number_of_nodes()
        self.num_edge_types = len(graph._edges.types)
        self.k = k
        self.embedding_initializer = initializers.get(embedding_initializer)
        self.embedding_regularizer = regularizers.get(embedding_regularizer)

    # layer names
    _NODE_REAL = "COMPLEX_NODE_REAL"
    _NODE_IMAG = "COMPLEX_NODE_IMAG"

    _REL_REAL = "COMPLEX_EDGE_TYPE_REAL"
    _REL_IMAG = "COMPLEX_EDGE_TYPE_IMAG"

    @staticmethod
    def embeddings(model):
        """
        Retrieve the embeddings for nodes/entities and edge types/relations in this model.

        Returns:
            A tuple of numpy complex arrays: the first element is the embeddings for nodes/entities
            (``shape = number of nodes × k``), the second element is the embeddings for edge
            types/relations (``shape = number of edge types x k``).
        """
        node = 1j * model.get_layer(ComplEx._NODE_IMAG).embeddings.numpy()
        node += model.get_layer(ComplEx._NODE_REAL).embeddings.numpy()

        rel = 1j * model.get_layer(ComplEx._REL_IMAG).embeddings.numpy()
        rel += model.get_layer(ComplEx._REL_REAL).embeddings.numpy()

        return node, rel

    def _embed(self, count, name):
        return Embedding(
            count,
            self.k,
            name=name,
            embeddings_initializer=self.embedding_initializer,
            embeddings_regularizer=self.embedding_regularizer,
        )

    def __call__(self, x):
        """
        Apply embedding layers to the source, relation and object input "ilocs" (sequential integer
        labels for the nodes and edge types).
        """
        s_iloc, r_iloc, o_iloc = x

        # ComplEx generates embeddings in C, which we model as separate real and imaginary
        # embeddings
        node_embeddings_real = self._embed(self.num_nodes, self._NODE_REAL)
        node_embeddings_imag = self._embed(self.num_nodes, self._NODE_IMAG)
        edge_type_embeddings_real = self._embed(self.num_nodes, self._REL_REAL)
        edge_type_embeddings_imag = self._embed(self.num_nodes, self._REL_IMAG)

        s_re = node_embeddings_real(s_iloc)
        s_im = node_embeddings_imag(s_iloc)

        r_re = edge_type_embeddings_real(r_iloc)
        r_im = edge_type_embeddings_imag(r_iloc)

        o_re = node_embeddings_real(o_iloc)
        o_im = node_embeddings_imag(o_iloc)

        scoring = ComplExScore()

        return scoring([s_re, s_im, r_re, r_im, o_re, o_im])

    def build(self):
        """
        Builds a ComplEx model.

        Returns:
            A tuple of (list of input tensors, tensor for ComplEx model score outputs)
        """
        s_iloc = Input(shape=(None,))
        r_iloc = Input(shape=(None,))
        o_iloc = Input(shape=(None,))

        x_inp = [s_iloc, r_iloc, o_iloc]
        x_out = self(x_inp)

        return x_inp, x_out
