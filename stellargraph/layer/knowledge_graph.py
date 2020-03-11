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

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Layer, Lambda, Dropout, Reshape, Embedding

from ..mapper.knowledge_graph import KGTripleGenerator, KGTripleSequence
from ..core.experimental import experimental
from ..core.validation import require_integer_in_range


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
        """
        Applies the layer.

        Args:

            inputs: a list of 6 tensors (``shape = batch size × 1 × embedding dimension k``), where
                the three consecutive pairs represent real and imaginary parts of the subject,
                relation and object embeddings, respectively, that is, ``inputs == [Re(subject),
                Im(subject), Re(relation), ...]``
        """
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


@experimental(
    reason="results from the reference paper have not been reproduced yet", issues=[862]
)
class ComplEx:
    """
    Embedding layers and a ComplEx scoring layers that implement the ComplEx knowledge graph
    embedding algorithm as in http://jmlr.org/proceedings/papers/v48/trouillon16.pdf

    Args:
        generator (KGTripleGenerator): A generator of triples to feed into the model.

        k (int): the dimension of the embedding (that is, a vector in C^k is learnt for each node
            and each link type)

        embeddings_initializer (str or func, optional): The initialiser to use for the embeddings
            (the default of random normal values matches the paper's reference implementation).

        embeddings_regularizer (str or func, optional): The regularizer to use for the embeddings.
    """

    def __init__(
        self,
        generator,
        k,
        embeddings_initializer="normal",
        embeddings_regularizer=None,
    ):
        if not isinstance(generator, KGTripleGenerator):
            raise TypeError(
                f"generator: expected KGTripleGenerator, found {type(generator).__name__}"
            )

        graph = generator.G
        self.num_nodes = graph.number_of_nodes()
        self.num_edge_types = len(graph._edges.types)
        self.k = k
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)

    # layer names
    _NODE_REAL = "COMPLEX_NODE_REAL"
    _NODE_IMAG = "COMPLEX_NODE_IMAG"

    _REL_REAL = "COMPLEX_EDGE_TYPE_REAL"
    _REL_IMAG = "COMPLEX_EDGE_TYPE_IMAG"

    @staticmethod
    def embeddings(model):
        """
        Retrieve the embeddings for nodes/entities and edge types/relations in the given model.

        Args:
            model (tensorflow.keras.Model): a Keras model created using a ``ComplEx`` instance.

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

    @staticmethod
    def rank_edges_against_all_nodes(model, test_data, known_edges_graph):
        """
        Returns the ranks of the true edges in ``test_data``, when scored against all other similar
        edges.

        For each input edge ``E = (s, r, o)``, the score of the *modified-object* edge ``(s, r, n)``
        is computed for every node ``n`` in the graph, and similarly the score of the
        *modified-subject* edge ``(n, r, o)``.

        This computes "raw" ranks: the score of each edge is ranked against all of the
        modified-object and modified subject-ones, for instance, if ``E = ("a", "X", "b")`` has
        score 3.14, and only one modified-object edge has a higher score (e.g. ``("a", "X", "c")``),
        then the raw modified-object rank for ``E`` will be 2; if all of the ``(n, "X", "b")`` edges
        have score less than 3.14, then the raw modified-subject rank for ``E`` will be 1.

        Args:
            model (tensorflow.keras.Model): a Keras model created using a ``ComplEx`` instance.

            test_data: the output of :meth:`KGTripleGenerator.flow` on some test triples

            known_edges_graph (StellarGraph):
                a graph instance containing all known edges/triples

        Returns:
            A numpy array of integer raw ranks. It has shape ``N × 2``, where N is the number of
            test triples in ``test_data``; the first column (``array[:, 0]``) holds the
            modified-object ranks, and the second (``array[:, 1]``) holds the modified-subject
            ranks.
        """

        if not isinstance(test_data, KGTripleSequence):
            raise TypeError(
                "test_data: expected KGTripleSequence; found {type(test_data).__name__}"
            )

        num_nodes = known_edges_graph.number_of_nodes()

        def ranks(pred, true_ilocs, true_is_source):
            batch_size = len(true_ilocs)
            assert pred.shape == (num_nodes, batch_size)

            # the score of the true edge, for each edge in the batch (this indexes in lock-step,
            # i.e. [pred[true_ilocs[0], range(batch_size)[0]], ...])
            true_scores = pred[true_ilocs, range(batch_size)]

            # for each column, compare all the scores against the score of the true edge
            greater = pred > true_scores

            # the raw rank is the number of elements scored higher than the true edge
            raw_rank = 1 + greater.sum(axis=0)
            assert raw_rank.shape == (batch_size,)

            return raw_rank

        n, all_r = ComplEx.embeddings(model)
        n_conj = n.conj()

        raws = []

        # run through the batches and compute the ranks for each one
        num_tested = 0
        for i in range(len(test_data)):
            ((subjects, rels, objects),) = test_data[i]
            num_tested += len(subjects)

            # batch_size x k
            ss = n[subjects, :]
            rs = all_r[rels, :]
            os = n[objects, :]

            # reproduce the scoring function for ranking the given subject and relation against all
            # other nodes (objects), and similarly given relation and object against all
            # subjects. The bulk operations give speeeeeeeeed.
            # (num_nodes x k, batch_size x k) -> num_nodes x batch_size
            mod_o_pred = np.inner(n_conj, ss * rs).real
            mod_s_pred = np.inner(n, rs * os.conj()).real

            mod_o_raw = ranks(mod_o_pred, true_ilocs=objects, true_is_source=True)
            mod_s_raw = ranks(mod_s_pred, true_ilocs=subjects, true_is_source=False)

            raws.append(np.column_stack((mod_o_raw, mod_s_raw)))

        # make one big array
        raw = np.concatenate(raws)
        # for each edge, there should be an pair of raw ranks
        assert raw.shape == (num_tested, 2)

        return raw

    def _embed(self, count, name):
        return Embedding(
            count,
            self.k,
            name=name,
            embeddings_initializer=self.embeddings_initializer,
            embeddings_regularizer=self.embeddings_regularizer,
        )

    def __call__(self, x):
        """
        Apply embedding layers to the source, relation and object input "ilocs" (sequential integer
        labels for the nodes and edge types).

        Args:
            x (list): list of 3 tensors (each batch size x 1) storing the ilocs of the subject,
                relation and object elements for each edge in the batch.
        """
        s_iloc, r_iloc, o_iloc = x

        # ComplEx generates embeddings in C, which we model as separate real and imaginary
        # embeddings
        node_embeddings_real = self._embed(self.num_nodes, self._NODE_REAL)
        node_embeddings_imag = self._embed(self.num_nodes, self._NODE_IMAG)
        edge_type_embeddings_real = self._embed(self.num_edge_types, self._REL_REAL)
        edge_type_embeddings_imag = self._embed(self.num_edge_types, self._REL_IMAG)

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
        s_iloc = Input(shape=1)
        r_iloc = Input(shape=1)
        o_iloc = Input(shape=1)

        x_inp = [s_iloc, r_iloc, o_iloc]
        x_out = self(x_inp)

        return x_inp, x_out


class DistMultScore(Layer):
    """
    DistMult scoring Keras layer.

    Original Paper: Embedding Entities and Relations for Learning and Inference in Knowledge
    Bases. Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, Li Deng. ICLR 2015

    This combines subject, relation and object embeddings into a score of the likelihood of the
    link.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        """
        Applies the layer.

        Args:

            inputs: a list of 3 tensors (``shape = batch size × 1 × embedding dimension``),
                representing the subject, relation and object embeddings, respectively, that is,
                ``inputs == [subject, relation, object]``
        """

        y_e1, m_r, y_e2 = inputs
        # y_e1^T M_r y_e2, where M_r = diag(m_r) is a diagonal matrix
        score = tf.reduce_sum(y_e1 * m_r * y_e2, axis=2)
        return score


@experimental(
    reason="results from the reference paper have not been reproduced yet", issues=[981]
)
class DistMult:
    """
    Embedding layers and a DistMult scoring layers that implement the DistMult knowledge graph
    embedding algorithm as in https://arxiv.org/pdf/1412.6575.pdf

    Args:
        generator (KGTripleGenerator): A generator of triples to feed into the model.

        embedding_dimension (int): the dimension of the embedding (that is, a vector in
            ``R^embedding_dimension`` is learnt for each node and each link type)

        embeddings_initializer (str or func, optional): The initialiser to use for the embeddings.

        embeddings_regularizer (str or func, optional): The regularizer to use for the embeddings.
    """

    def __init__(
        self,
        generator,
        embedding_dimension,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
    ):
        if not isinstance(generator, KGTripleGenerator):
            raise TypeError(
                f"generator: expected KGTripleGenerator, found {type(generator).__name__}"
            )

        require_integer_in_range(embedding_dimension, "embedding_dimension", min_val=1)

        graph = generator.G
        self.num_nodes = graph.number_of_nodes()
        self.num_edge_types = len(graph._edges.types)
        self.embedding_dimension = embedding_dimension
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)

    # layer names
    _NODE = "DISTMULT_NODE"
    _REL = "DISTMULT_EDGE_TYPE"

    @staticmethod
    def embeddings(model):
        """
        Retrieve the embeddings for nodes/entities and edge types/relations in the given model.

        Args:
            model (tensorflow.keras.Model): a Keras model created using a ``DistMult`` instance.

        Returns:
            A tuple of numpy arrays: the first element is the embeddings for nodes/entities
            (``shape = number of nodes × k``), the second element is the embeddings for edge
            types/relations (``shape = number of edge types x k``).
        """
        node = model.get_layer(DistMult._NODE).embeddings.numpy()
        rel = model.get_layer(DistMult._REL).embeddings.numpy()

        return node, rel

    def _embed(self, count, name):
        # FIXME(#980,https://github.com/tensorflow/tensorflow/issues/33755): embeddings can't use
        # constraints to be normalized: per section 4 in the paper, the embeddings should be
        # normalised to have unit norm.
        return Embedding(
            count,
            self.embedding_dimension,
            name=name,
            embeddings_initializer=self.embeddings_initializer,
            embeddings_regularizer=self.embeddings_regularizer,
        )

    def __call__(self, x):
        """
        Apply embedding layers to the source, relation and object input "ilocs" (sequential integer
        labels for the nodes and edge types).

        Args:
            x (list): list of 3 tensors (``shape = batch size × 1``) storing the ilocs of the
                subject, relation and object elements for each edge in the batch.
        """
        e1_iloc, r_iloc, e2_iloc = x

        # DistMult generates embeddings in R
        node_embeddings = self._embed(self.num_nodes, self._NODE)
        edge_type_embeddings = self._embed(self.num_edge_types, self._REL)

        y_e1 = node_embeddings(e1_iloc)
        m_r = edge_type_embeddings(r_iloc)
        y_e2 = node_embeddings(e2_iloc)

        scoring = DistMultScore()

        return scoring([y_e1, m_r, y_e2])

    def build(self):
        """
        Builds a DistMult model.

        Returns:
            A tuple of (list of input tensors, tensor for DistMult model score outputs)
        """
        e1_iloc = Input(shape=(None,))
        r_iloc = Input(shape=(None,))
        e2_iloc = Input(shape=(None,))

        x_inp = [e1_iloc, r_iloc, e2_iloc]
        x_out = self(x_inp)

        return x_inp, x_out
