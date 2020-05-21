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

from .misc import deprecated_model_function
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


class ComplEx:
    """
    Embedding layers and a ComplEx scoring layers that implement the ComplEx knowledge graph
    embedding algorithm as in http://jmlr.org/proceedings/papers/v48/trouillon16.pdf

    Args:
        generator (KGTripleGenerator): A generator of triples to feed into the model.

        embedding_dimension (int): the dimension of the embedding (that is, a vector in
            ``C^embedding_dimension`` is learnt for each node and each link type)

        embeddings_initializer (str or func, optional): The initialiser to use for the embeddings
            (the default of random normal values matches the paper's reference implementation).

        embeddings_regularizer (str or func, optional): The regularizer to use for the embeddings.
    """

    def __init__(
        self,
        generator,
        embedding_dimension,
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
        self.embedding_dimension = embedding_dimension

        def embed(count):
            return Embedding(
                count,
                embedding_dimension,
                embeddings_initializer=embeddings_initializer,
                embeddings_regularizer=embeddings_regularizer,
            )

        # ComplEx generates embeddings in C, which we model as separate real and imaginary
        # embeddings
        self._node_embeddings_real = embed(self.num_nodes)
        self._node_embeddings_imag = embed(self.num_nodes)
        self._edge_type_embeddings_real = embed(self.num_edge_types)
        self._edge_type_embeddings_imag = embed(self.num_edge_types)

    def embeddings(self):
        """
        Retrieve the embeddings for nodes/entities and edge types/relations in this ComplEx model.

        Returns:
            A tuple of numpy complex arrays: the first element is the embeddings for nodes/entities
            (``shape = number of nodes × k``), the second element is the embeddings for edge
            types/relations (``shape = number of edge types x k``).
        """
        node = 1j * self._node_embeddings_imag.embeddings.numpy()
        node += self._node_embeddings_real.embeddings.numpy()

        rel = 1j * self._edge_type_embeddings_imag.embeddings.numpy()
        rel += self._edge_type_embeddings_real.embeddings.numpy()

        return node, rel

    def rank_edges_against_all_nodes(
        self, test_data, known_edges_graph, tie_breaking="random"
    ):
        """
        Returns the ranks of the true edges in ``test_data``, when scored against all other similar
        edges.

        For each input edge ``E = (s, r, o)``, the score of the *modified-object* edge ``(s, r, n)``
        is computed for every node ``n`` in the graph, and similarly the score of the
        *modified-subject* edge ``(n, r, o)``.

        This computes "raw" and "filtered" ranks:

        raw
          The score of each edge is ranked against all of the modified-object and modified-subject
          ones, for instance, if ``E = ("a", "X", "b")`` has score 3.14, and only one
          modified-object edge has a higher score (e.g. ``F = ("a", "X", "c")``), then the raw
          modified-object rank for ``E`` will be 2; if all of the ``(n, "X", "b")`` edges have score
          less than 3.14, then the raw modified-subject rank for ``E`` will be 1.

        filtered
          The score of each edge is ranked against only the unknown modified-object and
          modified-subject edges. An edge is considered known if it is in ``known_edges_graph``
          which should typically hold every edge in the dataset (that is everything from the train,
          test and validation sets, if the data has been split). For instance, continuing the raw
          example, if the higher-scoring edge ``F`` is in the graph, then it will be ignored, giving
          a filtered modified-object rank for ``E`` of 1. (If ``F`` was not in the graph, the
          filtered modified-object rank would be 2.)

        Args:
            test_data: the output of :meth:`KGTripleGenerator.flow` on some test triples

            known_edges_graph (StellarGraph):
                a graph instance containing all known edges/triples

            tie_breaking ('random', 'top' or 'bottom'):
                How to rank true edges that tie with modified-object or modified-subject ones, see
                `Sun et al. "A Re-evaluation of Knowledge Graph Completion Methods"
                <http://arxiv.org/abs/1911.03903>`_

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

        all_node_embs, all_rel_embs = self.embeddings()
        all_node_embs_conj = all_node_embs.conj()

        raws = []
        filtereds = []

        # run through the batches and compute the ranks for each one
        num_tested = 0
        for ((subjects, rels, objects),) in test_data:
            num_tested += len(subjects)

            # batch_size x k
            ss = all_node_embs[subjects, :]
            rs = all_rel_embs[rels, :]
            os = all_node_embs[objects, :]

            # reproduce the scoring function for ranking the given subject and relation against all
            # other nodes (objects), and similarly given relation and object against all
            # subjects. The bulk operations give speeeeeeeeed.
            # (num_nodes x k, batch_size x k) -> num_nodes x batch_size
            mod_o_pred = np.inner(all_node_embs_conj, ss * rs).real
            mod_s_pred = np.inner(all_node_embs, rs * os.conj()).real

            mod_o_raw, mod_o_filt = _ranks_from_score_columns(
                mod_o_pred,
                true_modified_node_ilocs=objects,
                unmodified_node_ilocs=subjects,
                true_rel_ilocs=rels,
                modified_object=True,
                known_edges_graph=known_edges_graph,
                tie_breaking=tie_breaking,
            )
            mod_s_raw, mod_s_filt = _ranks_from_score_columns(
                mod_s_pred,
                true_modified_node_ilocs=subjects,
                true_rel_ilocs=rels,
                modified_object=False,
                unmodified_node_ilocs=objects,
                known_edges_graph=known_edges_graph,
                tie_breaking=tie_breaking,
            )

            raws.append(np.column_stack((mod_o_raw, mod_s_raw)))
            filtereds.append(np.column_stack((mod_o_filt, mod_s_filt)))

        # make one big array
        raw = np.concatenate(raws)
        filtered = np.concatenate(filtereds)
        # for each edge, there should be an pair of raw ranks
        assert raw.shape == filtered.shape == (num_tested, 2)

        return raw, filtered

    def __call__(self, x):
        """
        Apply embedding layers to the source, relation and object input "ilocs" (sequential integer
        labels for the nodes and edge types).

        Args:
            x (list): list of 3 tensors (each batch size x 1) storing the ilocs of the subject,
                relation and object elements for each edge in the batch.
        """
        s_iloc, r_iloc, o_iloc = x

        s_re = self._node_embeddings_real(s_iloc)
        s_im = self._node_embeddings_imag(s_iloc)

        r_re = self._edge_type_embeddings_real(r_iloc)
        r_im = self._edge_type_embeddings_imag(r_iloc)

        o_re = self._node_embeddings_real(o_iloc)
        o_im = self._node_embeddings_imag(o_iloc)

        scoring = ComplExScore()

        return scoring([s_re, s_im, r_re, r_im, o_re, o_im])

    def in_out_tensors(self):
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

    build = deprecated_model_function(in_out_tensors, "build")


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

        def embed(count):
            # FIXME(#980,https://github.com/tensorflow/tensorflow/issues/33755): embeddings can't use
            # constraints to be normalized: per section 4 in the paper, the embeddings should be
            # normalised to have unit norm.
            return Embedding(
                count,
                embedding_dimension,
                embeddings_initializer=embeddings_initializer,
                embeddings_regularizer=embeddings_regularizer,
            )

        # DistMult generates embeddings in R
        self._node_embeddings = embed(self.num_nodes)
        self._edge_type_embeddings = embed(self.num_edge_types)

    def embeddings(self):
        """
        Retrieve the embeddings for nodes/entities and edge types/relations in this DistMult model.

        Returns:
            A tuple of numpy arrays: the first element is the embeddings for nodes/entities
            (``shape = number of nodes × k``), the second element is the embeddings for edge
            types/relations (``shape = number of edge types x k``).
        """
        return (
            self._node_embeddings.embeddings.numpy(),
            self._edge_type_embeddings.embeddings.numpy(),
        )

    def rank_edges_against_all_nodes(
        self, test_data, known_edges_graph, tie_breaking="random"
    ):
        """
        Returns the ranks of the true edges in ``test_data``, when scored against all other similar
        edges.

        For each input edge ``E = (s, r, o)``, the score of the *modified-object* edge ``(s, r, n)``
        is computed for every node ``n`` in the graph, and similarly the score of the
        *modified-subject* edge ``(n, r, o)``.

        This computes "raw" and "filtered" ranks:

        raw
          The score of each edge is ranked against all of the modified-object and modified-subject
          ones, for instance, if ``E = ("a", "X", "b")`` has score 3.14, and only one
          modified-object edge has a higher score (e.g. ``F = ("a", "X", "c")``), then the raw
          modified-object rank for ``E`` will be 2; if all of the ``(n, "X", "b")`` edges have score
          less than 3.14, then the raw modified-subject rank for ``E`` will be 1.

        filtered
          The score of each edge is ranked against only the unknown modified-object and
          modified-subject edges. An edge is considered known if it is in ``known_edges_graph``
          which should typically hold every edge in the dataset (that is everything from the train,
          test and validation sets, if the data has been split). For instance, continuing the raw
          example, if the higher-scoring edge ``F`` is in the graph, then it will be ignored, giving
          a filtered modified-object rank for ``E`` of 1. (If ``F`` was not in the graph, the
          filtered modified-object rank would be 2.)

        Args:
            test_data: the output of :meth:`KGTripleGenerator.flow` on some test triples

            known_edges_graph (StellarGraph):
                a graph instance containing all known edges/triples

            tie_breaking ('random', 'top' or 'bottom'):
                How to rank true edges that tie with modified-object or modified-subject ones, see
                `Sun et al. "A Re-evaluation of Knowledge Graph Completion Methods"
                <http://arxiv.org/abs/1911.03903>`_

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

        all_node_embs, all_rel_embs = self.embeddings()

        raws = []
        filtereds = []

        # run through the batches and compute the ranks for each one
        num_tested = 0
        for ((subjects, rels, objects),) in test_data:
            num_tested += len(subjects)

            # batch_size x k
            ss = all_node_embs[subjects, :]
            rs = all_rel_embs[rels, :]
            os = all_node_embs[objects, :]

            # reproduce the scoring function for ranking the given subject and relation against all
            # other nodes (objects), and similarly given relation and object against all
            # subjects. The bulk operations give speeeeeeeeed.
            # (num_nodes x k, batch_size x k) -> num_nodes x batch_size
            mod_o_pred = np.inner(all_node_embs, ss * rs)
            mod_s_pred = np.inner(all_node_embs, rs * os)

            mod_o_raw, mod_o_filt = _ranks_from_score_columns(
                mod_o_pred,
                true_modified_node_ilocs=objects,
                unmodified_node_ilocs=subjects,
                true_rel_ilocs=rels,
                modified_object=True,
                known_edges_graph=known_edges_graph,
                tie_breaking=tie_breaking,
            )
            mod_s_raw, mod_s_filt = _ranks_from_score_columns(
                mod_s_pred,
                true_modified_node_ilocs=subjects,
                true_rel_ilocs=rels,
                modified_object=False,
                unmodified_node_ilocs=objects,
                known_edges_graph=known_edges_graph,
                tie_breaking=tie_breaking,
            )

            raws.append(np.column_stack((mod_o_raw, mod_s_raw)))
            filtereds.append(np.column_stack((mod_o_filt, mod_s_filt)))

        # make one big array
        raw = np.concatenate(raws)
        filtered = np.concatenate(filtereds)
        # for each edge, there should be an pair of raw ranks
        assert raw.shape == filtered.shape == (num_tested, 2)

        return raw, filtered

    def __call__(self, x):
        """
        Apply embedding layers to the source, relation and object input "ilocs" (sequential integer
        labels for the nodes and edge types).

        Args:
            x (list): list of 3 tensors (``shape = batch size × 1``) storing the ilocs of the
                subject, relation and object elements for each edge in the batch.
        """
        e1_iloc, r_iloc, e2_iloc = x

        y_e1 = self._node_embeddings(e1_iloc)
        m_r = self._edge_type_embeddings(r_iloc)
        y_e2 = self._node_embeddings(e2_iloc)

        scoring = DistMultScore()

        return scoring([y_e1, m_r, y_e2])

    def in_out_tensors(self):
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

    build = deprecated_model_function(in_out_tensors, "build")


class RotatEScore(Layer):
    def __init__(self, margin, norm_order, **kwargs):
        super().__init__(**kwargs)
        self._margin = margin
        self._norm_order = norm_order

    def get_config(self):
        return {
            **super().get_config(),
            "margin": self._margin,
            "norm_order": self._norm_order,
        }

    def call(self, inputs):
        s_re, s_im, r_re, r_im, o_re, o_im = inputs
        # expansion of s◦r - t
        re = s_re * r_re - s_im * r_im - o_re
        im = s_re * r_im + s_im * r_re - o_im
        # norm the vector: -|| ... ||_p
        return self._margin - tf.norm(
            tf.sqrt(re * re + im * im), ord=self._norm_order, axis=2
        )


@experimental(reason="demo and documentation is missing", issues=[1549, 1550])
class RotatE:
    """
    Implementation of https://arxiv.org/abs/1902.10197
    """

    def __init__(
        self,
        generator,
        embedding_dimension,
        # default taken from the paper's code: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
        margin=12.0,
        # default taken from the paper's code: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
        norm_order=2,
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
        self.embedding_dimension = embedding_dimension

        self._scoring = RotatEScore(margin=margin, norm_order=norm_order)

        def embed(count):
            return Embedding(
                count,
                embedding_dimension,
                embeddings_initializer=embeddings_initializer,
                embeddings_regularizer=embeddings_regularizer,
            )

        # RotatE generates embeddings in C, which we model as separate real and imaginary embeddings
        # for node types, and just the phase for edge types (since they have |x| = 1)
        self._node_embeddings_real = embed(self.num_nodes)
        self._node_embeddings_imag = embed(self.num_nodes)

        # it doesn't make sense to regularize the phase, because it's circular
        self._edge_type_embeddings_phase = Embedding(
            self.num_edge_types,
            embedding_dimension,
            embeddings_initializer=embeddings_initializer,
        )

    def embeddings(self):
        node = 1j * self._node_embeddings_imag.embeddings.numpy()
        node += self._node_embeddings_real.embeddings.numpy()

        phase = self._edge_type_embeddings_phase.embeddings.numpy()
        rel = 1j * np.sin(phase)
        rel += np.cos(phase)

        return node, rel

    def rank_edges_against_all_nodes(
        self, test_data, known_edges_graph, tie_breaking="random"
    ):
        if not isinstance(test_data, KGTripleSequence):
            raise TypeError(
                "test_data: expected KGTripleSequence; found {type(test_data).__name__}"
            )

        num_nodes = known_edges_graph.number_of_nodes()

        all_node_embs, all_rel_embs = self.embeddings()

        raws = []
        filtereds = []

        # run through the batches and compute the ranks for each one
        num_tested = 0
        for ((subjects, rels, objects),) in test_data:
            num_tested += len(subjects)

            # batch_size x k
            ss = all_node_embs[subjects, :]
            rs = all_rel_embs[rels, :]
            os = all_node_embs[objects, :]

            # (the margin is a fixed offset that doesn't affect relative ranks)
            mod_o_pred = -np.linalg.norm(
                (ss * rs)[None, :, :] - all_node_embs[:, None, :], axis=2
            )
            mod_s_pred = -np.linalg.norm(
                (all_node_embs)[:, None, :] * rs[None, :, :] - os[None, :, :], axis=2
            )

            mod_o_raw, mod_o_filt = _ranks_from_score_columns(
                mod_o_pred,
                true_modified_node_ilocs=objects,
                unmodified_node_ilocs=subjects,
                true_rel_ilocs=rels,
                modified_object=True,
                known_edges_graph=known_edges_graph,
                tie_breaking=tie_breaking,
            )
            mod_s_raw, mod_s_filt = _ranks_from_score_columns(
                mod_s_pred,
                true_modified_node_ilocs=subjects,
                true_rel_ilocs=rels,
                modified_object=False,
                unmodified_node_ilocs=objects,
                known_edges_graph=known_edges_graph,
                tie_breaking=tie_breaking,
            )

            raws.append(np.column_stack((mod_o_raw, mod_s_raw)))
            filtereds.append(np.column_stack((mod_o_filt, mod_s_filt)))

        # make one big array
        raw = np.concatenate(raws)
        filtered = np.concatenate(filtereds)
        # for each edge, there should be an pair of raw ranks
        assert raw.shape == filtered.shape == (num_tested, 2)

        return raw, filtered

    def __call__(self, x):
        s_iloc, r_iloc, o_iloc = x

        s_re = self._node_embeddings_real(s_iloc)
        s_im = self._node_embeddings_imag(s_iloc)

        r_phase = self._edge_type_embeddings_phase(r_iloc)

        r_re = tf.math.cos(r_phase)
        r_im = tf.math.sin(r_phase)

        o_re = self._node_embeddings_real(o_iloc)
        o_im = self._node_embeddings_imag(o_iloc)

        return self._scoring([s_re, s_im, r_re, r_im, o_re, o_im])

    def in_out_tensors(self):
        s_iloc = Input(shape=1)
        r_iloc = Input(shape=1)
        o_iloc = Input(shape=1)

        x_inp = [s_iloc, r_iloc, o_iloc]
        x_out = self(x_inp)

        return x_inp, x_out


def _ranks_from_comparisons(greater, greater_equal, tie_breaking):
    strict = 1 + greater.sum(axis=0)
    # with_ties - strict = the number of elements exactly equal (including the true edge itself)
    with_ties = greater_equal.sum(axis=0)

    if tie_breaking == "top":
        return strict
    elif tie_breaking == "bottom":
        return with_ties
    elif tie_breaking == "random":
        return np.random.randint(strict, with_ties + 1)
    else:
        raise ValueError(
            f"tie_breaking: expected 'top', 'bottom' or 'random', found {tie_breaking!r}"
        )


def _ranks_from_score_columns(
    pred,
    *,
    true_modified_node_ilocs,
    unmodified_node_ilocs,
    true_rel_ilocs,
    modified_object,
    known_edges_graph,
    tie_breaking,
):
    """
    Compute the raw and filtered ranks of a set of true edges ``E = (s, r, o)`` against all
    mutations of one end of them, e.g. ``E' = (s, r, n)`` for "modified-object".

    The raw rank is the total number of edges scored higher than the true edge ``E``, and the
    filtered rank is the total number of unknown edges (not in ``known_edges_graph``).

    Args:

        pred: a 2D array: each column represents the scores for a single true edge and its
            mutations, where the row indicates the ``n`` in ``E'`` (e.g. row 0 corresponds to ``n``
            = node with iloc 0)
        true_modified_node_ilocs: an array of ilocs of the actual node that was modified, that is,
            ``o`` for modified-object and ``s`` for modified subject``, index ``i`` corresponds to
            the iloc for column ``pred[:, i]``.
        unmodified_node_ilocs: similar to ``true_modified_node_ilocs``, except for the other end of
            the edge: the node that was not modified.
        true_rel_ilocs: similar to ``true_modified_node_ilocs``, except for the relationship type of
            the edge (``r``).
        modified_object (bool): whether the object was modified (``True``), or the subject
            (``False``)
        known_edges_graph (StellarGraph): a graph containing all the known edges that should be
            ignored when computing filtered ranks

    Returns:
        a tuple of raw ranks and filtered ranks, each is an array of integers >= 1 where index ``i``
        corresponds to the rank of the true edge among all of the scores in column ``pred[:, i]``.
    """
    batch_size = len(true_modified_node_ilocs)
    assert pred.shape == (known_edges_graph.number_of_nodes(), batch_size)
    assert unmodified_node_ilocs.shape == true_rel_ilocs.shape == (batch_size,)

    # the score of the true edge, for each edge in the batch (this indexes in lock-step,
    # i.e. [pred[true_modified_node_ilocs[0], range(batch_size)[0]], ...])
    true_scores = pred[true_modified_node_ilocs, range(batch_size)]

    # for each column, compare all the scores against the score of the true edge
    greater = pred > true_scores
    greater_equal = pred >= true_scores

    # the raw rank is the number of elements scored higher than the true edge
    raw_rank = _ranks_from_comparisons(greater, greater_equal, tie_breaking)

    # the filtered rank is the number of unknown elements scored higher, where an element is
    # known if the edge (s, r, n) (for modified-object) or (n, r, o) (for modified-subject)
    # exists in known_edges_graph.
    if modified_object:
        neigh_func = known_edges_graph.out_nodes
    else:
        neigh_func = known_edges_graph.in_nodes

    for batch_column, (unmodified, r) in enumerate(
        zip(unmodified_node_ilocs, true_rel_ilocs)
    ):
        this_neighs = neigh_func(unmodified, edge_types=[r], use_ilocs=True)
        greater[this_neighs, batch_column] = False
        greater_equal[this_neighs, batch_column] = False

    # the actual elements should be counted as equal, whether or not it was a known edge or not
    greater_equal[true_modified_node_ilocs, range(batch_size)] = True

    filtered_rank = _ranks_from_comparisons(greater, greater_equal, tie_breaking)

    assert raw_rank.shape == filtered_rank.shape == (batch_size,)
    return raw_rank, filtered_rank
