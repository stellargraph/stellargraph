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

import abc

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Layer, Lambda, Dropout, Reshape, Embedding

from .misc import deprecated_model_function
from ..mapper.knowledge_graph import KGTripleGenerator, KGTripleSequence
from ..core.experimental import experimental
from ..core.validation import require_integer_in_range, comma_sep
from ..utils.hyperbolic import *


class KGModel:
    def __init__(
        self,
        generator,
        scoring,
        embedding_dimension,
        *,
        embeddings_initializer,
        embeddings_regularizer,
    ):
        if not isinstance(generator, KGTripleGenerator):
            raise TypeError(
                f"generator: expected KGTripleGenerator, found {type(generator).__name__}"
            )

        if not isinstance(scoring, KGScore):
            raise TypeError(
                f"scoring: expected KGScore subclass, found {type(scoring).__name__}"
            )

        require_integer_in_range(embedding_dimension, "embedding_dimension", min_val=1)

        graph = generator.G
        self.num_nodes = graph.number_of_nodes()
        self.num_edge_types = len(graph._edges.types)

        self._scoring = scoring

        embeddings = scoring.embeddings(
            self.num_nodes,
            self.num_edge_types,
            embedding_dimension,
            embeddings_initializer,
            embeddings_regularizer,
        )

        self._validate_embeddings(embeddings)
        self._node_embs, self._edge_type_embs = embeddings

    def _validate_embeddings(self, embeddings):
        def error(found):
            raise ValueError(
                f"scoring: expected 'embeddings' method to return two lists of tf.keras.layers.Embedding layers, found {found}"
            )

        if len(embeddings) != 2:
            error(f"a sequence of length {len(embeddings)}")

        a, b = embeddings

        if not all(isinstance(x, list) for x in embeddings):
            error(f"a pair with types ({type(a).__name__}, {type(b).__name__})")

        if not all(isinstance(x, Embedding) for x in a + b):
            a_types = comma_sep(a, stringify=lambda x: type(x).__name__)
            b_types = comma_sep(b, stringify=lambda x: type(x).__name__)
            error(f"a pair of lists containing types ([{a_types}], [{b_types}])")

        # all good!
        return

    def embedding_arrays(self):
        """
        Retrieve each separate set of embeddings for nodes/entities and edge types/relations in this model.

        Returns:
            A tuple of lists of numpy arrays: the first element contains the embeddings for nodes/entities (for each element, ``shape
            = number of nodes × k``), the second element contains the embeddings for edge types/relations
            (``shape = number of edge types x k``), where ``k`` is some notion of the embedding
            dimension for each layer. The type of the embeddings depends on the specific scoring function chosen.
        """
        node = [e.embeddings.numpy() for e in self._node_embs]
        edge_type = [e.embeddings.numpy() for e in self._edge_type_embs]
        return self._scoring.embeddings_to_numpy(node, edge_type)

    def embeddings(self):
        """
        Retrieve the embeddings for nodes/entities and edge types/relations in this model, if there's only one set of embeddings for each of nodes and edge types.

        Returns:
            A tuple of numpy arrays: the first element is the embeddings for nodes/entities (``shape
            = number of nodes × k``), the second element is the embeddings for edge types/relations
            (``shape = number of edge types x k``), where ``k`` is some notion of the embedding
            dimension. The type of the embeddings depends on the specific scoring function chosen.
        """
        node, edge_type = self.embedding_arrays()
        if len(node) != 1 and len(edge_type) != 1:
            raise ValueError(
                f"embeddings: expected a single embedding array for nodes and for edge types from embedding_arrays, found {len(node)} node and {len(edge_type)} edge type arrays; use embedding_arrays to retrieve the lists instead"
            )

        return node[0], edge_type[0]

    def __call__(self, x):
        """
        Apply embedding layers to the source, relation and object input "ilocs" (sequential integer
        labels for the nodes and edge types).

        Args:
            x (list): list of 3 tensors (each batch size x 1) storing the ilocs of the subject,
                relation and object elements for each edge in the batch.
        """
        s_iloc, r_iloc, o_iloc = x

        sequenced = [
            (s_iloc, self._node_embs),
            (r_iloc, self._edge_type_embs),
            (o_iloc, self._node_embs),
        ]

        inp = [
            emb_layer(ilocs)
            for ilocs, emb_layers in sequenced
            for emb_layer in emb_layers
        ]

        return self._scoring(inp)

    def in_out_tensors(self):
        """
        Builds a knowledge graph model.

        Returns:
            A tuple of (list of input tensors, tensor for ComplEx model score outputs)
        """
        s_iloc = Input(shape=1)
        r_iloc = Input(shape=1)
        o_iloc = Input(shape=1)

        x_inp = [s_iloc, r_iloc, o_iloc]
        x_out = self(x_inp)

        return x_inp, x_out

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

        node_embs, edge_type_embs = self.embedding_arrays()
        extra_data = self._scoring.bulk_scoring_data(node_embs, edge_type_embs)

        raws = []
        filtereds = []

        # run through the batches and compute the ranks for each one
        num_tested = 0
        for ((subjects, rels, objects),) in test_data:
            num_tested += len(subjects)

            # batch_size x k
            ss = [e[subjects, :] for e in node_embs]
            rs = [e[rels, :] for e in edge_type_embs]
            os = [e[objects, :] for e in node_embs]

            mod_o_pred, mod_s_pred = self._scoring.bulk_scoring(
                node_embs, extra_data, ss, rs, os,
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


class KGScore(abc.ABC):
    @abc.abstractmethod
    def embeddings(
        self, num_nodes, num_edge_types, dimension, initializer, regularizer
    ):
        """
        Create appropriate embedding layer(s) for this scoring.

        Args:
            num_nodes: the number of nodes in this graph.
            num_edge_types: the number of edge types/relations in this graph.
            dimension: the requested embedding dimension, for whatever that means for this scoring.
            initializer: the initializer to use for embeddings, when required.
            regularizer: the regularizer to use for embeddings, when required.

        Returns:
            A pair of lists of :class:`tensorflow.keras.layers.Embedding` layers, corresponding to
            nodes and edge types.
        """
        ...

    def embeddings_to_numpy(self, node_embs, edge_type_embs):
        """
        Convert raw embedding NumPy arrays into "semantic" embeddings, such as complex numbers instead
        of interleaved real numbers.

        Args:
            node_embs: ``num_nodes × k`` array of all node embeddings, where ``k`` is the size of
                the embeddings returned by :meth:embeddings_to_numpy`.
            edge_type_embs: ``num_edge_type × k`` array of all edge type/relation embeddings, where
                ``k`` is the size of the embeddings returned by :meth:embeddings_to_numpy`.

        Returns:
            Model-specific NumPy arrays corresponding to some useful view of the embeddings vectors.
        """
        return node_embs, edge_type_embs

    def bulk_scoring_data(self, node_embs, edge_type_embs):
        """
        Pre-compute some data for bulk ranking, if any such data would be helpful.
        """
        return None

    @abc.abstractmethod
    def bulk_scoring(
        self, node_embs, extra_data, s_embs, r_embs, o_embs,
    ):
        """
        Compute a batch of modified-object and modified-subject scores for ranking.

        Args:
            node_embs: ``num_nodes × k`` array of all node embeddings, where ``k`` is the size of
                the embeddings returned by :meth:embeddings_to_numpy`.

            extra_data: the return value of :meth:`bulk_scoring_data`

            s_embs: ``batch_size × k`` embeddings for the true source nodes
            r_embs: ``batch_size × k`` embeddings for the true edge types/relations
            o_embs: ``batch_size × k`` embeddings for the true object nodes

        Returns:
            This should return a pair of NumPy arrays of shape ``num_nodes × batch_size``. The first
            array contains scores of the modified-object edges, and the second contains scores of
            the modified-subject edges.
        """
        ...

    # this isn't a subclass of Keras Layer, because a model or other combination of individual
    # layers is okay too, but this model will be applied by calling the instance
    @abc.abstractmethod
    def __call__(self, inputs):
        """
        Apply this scoring mechanism to the selected values from the embedding layers.

        Args:
            inputs: a list of tensors selected from each of the embedding layers, concatenated like
                ``[source, source, ..., edge types, edge_types, ..., object, object, ...]``
        """
        ...


def _numpy_complex(arrays):
    emb = 1j * arrays[1]
    emb += arrays[0]
    return emb


class ComplExScore(Layer, KGScore):
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

    def embeddings(
        self, num_nodes, num_edge_types, dimension, initializer, regularizer
    ):
        def embed(count):
            return Embedding(
                count,
                dimension,
                embeddings_initializer=initializer,
                embeddings_regularizer=regularizer,
            )

        # ComplEx generates embeddings in C, which we model as separate real and imaginary
        # embeddings
        nodes = [embed(num_nodes), embed(num_nodes)]
        edge_types = [embed(num_edge_types), embed(num_edge_types)]

        return nodes, edge_types

    def embeddings_to_numpy(self, node_embs, edge_type_embs):
        return (
            [_numpy_complex(node_embs)],
            [_numpy_complex(edge_type_embs)],
        )

    def bulk_scoring_data(self, node_embs, edge_type_embs):
        return node_embs[0].conj()

    def bulk_scoring(
        self, node_embs, node_embs_conj, s_embs, r_embs, o_embs,
    ):
        node_embs = node_embs[0]
        s_embs = s_embs[0]
        r_embs = r_embs[0]
        o_embs = o_embs[0]

        mod_o_pred = np.inner(node_embs_conj, s_embs * r_embs).real
        mod_s_pred = np.inner(node_embs, r_embs * o_embs.conj()).real
        return mod_o_pred, mod_s_pred

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


class ComplEx(KGModel):
    """
    Embedding layers and a ComplEx scoring layers that implement the ComplEx knowledge graph
    embedding algorithm as in http://jmlr.org/proceedings/papers/v48/trouillon16.pdf

    .. seealso::

       Example using ComplEx: `link prediction <https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/complex-link-prediction.html>`__

       Related models: other knowledge graph models, see :class:`.KGTripleGenerator` for a full list.

       Appropriate data generator: :class:`.KGTripleGenerator`.

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
        super().__init__(
            generator,
            ComplExScore(),
            embedding_dimension=embedding_dimension,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
        )

    build = deprecated_model_function(KGModel.in_out_tensors, "build")


class DistMultScore(Layer, KGScore):
    """
    DistMult scoring Keras layer.

    Original Paper: Embedding Entities and Relations for Learning and Inference in Knowledge
    Bases. Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, Li Deng. ICLR 2015

    This combines subject, relation and object embeddings into a score of the likelihood of the
    link.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def embeddings(
        self, num_nodes, num_edge_types, dimension, initializer, regularizer
    ):
        def embed(count):
            # FIXME(#980,https://github.com/tensorflow/tensorflow/issues/33755): embeddings can't
            # use constraints to be normalized: per section 4 in the paper, the embeddings should be
            # normalised to have unit norm.
            return Embedding(
                count,
                dimension,
                embeddings_initializer=initializer,
                embeddings_regularizer=regularizer,
            )

        # DistMult generates embeddings in R
        nodes = [embed(num_nodes)]
        edge_types = [embed(num_edge_types)]
        return nodes, edge_types

    def bulk_scoring(
        self, all_n_embs, _extra_data, s_embs, r_embs, o_embs,
    ):
        all_n_embs = all_n_embs[0]
        s_embs = s_embs[0]
        r_embs = r_embs[0]
        o_embs = o_embs[0]

        mod_o_pred = np.inner(all_n_embs, s_embs * r_embs)
        mod_s_pred = np.inner(all_n_embs, r_embs * o_embs)
        return mod_o_pred, mod_s_pred

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


class DistMult(KGModel):
    """
    Embedding layers and a DistMult scoring layers that implement the DistMult knowledge graph
    embedding algorithm as in https://arxiv.org/pdf/1412.6575.pdf

    .. seealso::

       Example using DistMult: `link prediction <https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/distmult-link-prediction.html>`__

       Related models: other knowledge graph models, see :class:`.KGTripleGenerator` for a full list.

       Appropriate data generator: :class:`.KGTripleGenerator`.

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
        super().__init__(
            generator,
            DistMultScore(),
            embedding_dimension=embedding_dimension,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
        )

    build = deprecated_model_function(KGModel.in_out_tensors, "build")


class RotatEScore(Layer, KGScore):
    def __init__(self, margin, norm_order, **kwargs):
        super().__init__(**kwargs)
        self._margin = margin
        self._norm_order = norm_order

    def embeddings(
        self, num_nodes, num_edge_types, dimension, initializer, regularizer
    ):
        def embed(count, reg=regularizer):
            return Embedding(
                count,
                dimension,
                embeddings_initializer=initializer,
                embeddings_regularizer=reg,
            )

        # RotatE generates embeddings in C, which we model as separate real and imaginary
        # embeddings for node types, and just the phase for edge types (since they have |x| = 1)
        nodes = [embed(num_nodes), embed(num_nodes)]
        # it doesn't make sense to regularize the phase, because it's circular
        edge_types = [embed(num_edge_types, reg=None)]
        return nodes, edge_types

    def embeddings_to_numpy(self, node_embs, edge_type_embs):
        nodes = _numpy_complex(node_embs)
        edge_types = 1j * np.sin(edge_type_embs[0])
        edge_types += np.cos(edge_type_embs[0])
        return [nodes], [edge_types]

    def bulk_scoring(
        self, all_n_embs, _extra_data, s_embs, r_embs, o_embs,
    ):
        all_n_embs = all_n_embs[0]
        s_embs = s_embs[0]
        r_embs = r_embs[0]
        o_embs = o_embs[0]

        # (the margin is a fixed offset that doesn't affect relative ranks)
        mod_o_pred = -np.linalg.norm(
            (s_embs * r_embs)[None, :, :] - all_n_embs[:, None, :],
            ord=self._norm_order,
            axis=2,
        )
        mod_s_pred = -np.linalg.norm(
            all_n_embs[:, None, :] * r_embs[None, :, :] - o_embs[None, :, :],
            ord=self._norm_order,
            axis=2,
        )
        return mod_o_pred, mod_s_pred

    def get_config(self):
        return {
            **super().get_config(),
            "margin": self._margin,
            "norm_order": self._norm_order,
        }

    def call(self, inputs):
        s_re, s_im, r_phase, o_re, o_im = inputs
        r_re = tf.math.cos(r_phase)
        r_im = tf.math.sin(r_phase)

        # expansion of s◦r - t
        re = s_re * r_re - s_im * r_im - o_re
        im = s_re * r_im + s_im * r_re - o_im
        # norm the vector: -|| ... ||_p
        return self._margin - tf.norm(
            tf.sqrt(re * re + im * im), ord=self._norm_order, axis=2
        )


@experimental(reason="demo and documentation is missing", issues=[1549, 1550])
class RotatE(KGModel):
    """
    Implementation of https://arxiv.org/abs/1902.10197

    .. seealso::

       Related models: other knowledge graph models, see :class:`.KGTripleGenerator` for a full list.

       Appropriate data generator: :class:`.KGTripleGenerator`.
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
        super().__init__(
            generator,
            RotatEScore(margin=margin, norm_order=norm_order),
            embedding_dimension,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
        )


class RotHEScoring(Layer, KGScore):
    def __init__(self, hyperbolic):
        self._hyperbolic = hyperbolic
        if self._hyperbolic:
            self._convert = lambda c, v: poincare_ball_exp(c, None, v)
            self._add = poincare_ball_mobius_add
            self._squared_distance = lambda c, v, w: tf.square(
                poincare_ball_distance(c, v, w)
            )
        else:
            self._convert = lambda _c, v: v
            self._add = lambda _c, v, w: v + w
            self._squared_distance = lambda _c, v, w: tf.reduce_sum(
                tf.math.squared_difference(v, w), axis=-1
            )

        super().__init__()

    def embeddings(
        self, num_nodes, num_edge_types, dimension, initializer, regularizer
    ):
        if dimension % 2 != 0:
            raise ValueError(
                f"embedding_dimension: expected an even integer, found {dimension}"
            )

        def embed(count, dim=dimension):
            return Embedding(
                count,
                dim,
                embeddings_initializer=initializer,
                embeddings_regularizer=regularizer,
            )

        nodes = [embed(num_nodes), embed(num_nodes, 1)]
        edge_types = [embed(num_edge_types), embed(num_edge_types, dimension // 2)]
        return nodes, edge_types

    def build(self, input_shapes):
        if self._hyperbolic:
            self.curvature_prime = self.add_weight(shape=(1,), name="curvature_prime")
        else:
            self.curvature_prime = None

        super().build(input_shapes)

    def _curvature(self):
        assert self.built
        if not self._hyperbolic:
            return tf.constant([0.0])

        return tf.math.softplus(self.curvature_prime)

    def _rotate(self, theta, emb):
        shape = tf.maximum(tf.shape(theta), tf.shape(emb))
        # manual rotation matrix
        cos = tf.math.cos(theta)
        sin = tf.math.sin(theta)
        evens = cos * emb[..., ::2] - sin * emb[..., 1::2]
        odds = sin * emb[..., ::2] + cos * emb[..., 1::2]
        return tf.reshape(tf.stack([evens, odds], axis=-1), shape)

    def call(self, inputs):
        e_s, b_s, r_r, theta_r, e_o, b_o = inputs

        curvature = self._curvature()

        b_s = tf.squeeze(b_s, axis=-1)
        b_o = tf.squeeze(b_o, axis=-1)

        eh_s = self._convert(curvature, e_s)
        rh_r = self._convert(curvature, r_r)
        eh_o = self._convert(curvature, e_o)

        rotated_s = self._rotate(theta_r, eh_s)
        d = self._squared_distance(
            curvature, self._add(curvature, rotated_s, rh_r), eh_o
        )

        return -d + b_s + b_o

    def bulk_scoring(
        self, all_n_embs, _extra_data, s_embs, r_embs, o_embs,
    ):
        curvature = self._curvature()

        e_all, b_all = all_n_embs
        e_all = e_all[:, None, :]
        b_all = b_all[:, None, 0]

        e_s, b_s = s_embs
        e_s = e_s[None, :, :]
        b_s = b_s[None, :, 0]

        r_r, theta_r = r_embs
        r_r = r_r[None, :, :]
        theta_r = theta_r[None, :, :]

        e_o, b_o = o_embs
        e_o = e_o[None, :, :]
        b_o = b_o[None, :, 0]

        eh_s = self._convert(curvature, e_s)
        rh_r = self._convert(curvature, r_r)

        rotated_s = self._rotate(theta_r, eh_s)
        d_mod_o = self._squared_distance(
            curvature, self._add(curvature, rotated_s, rh_r), e_all
        )
        mod_o_pred = -d_mod_o + b_s + b_all

        del eh_s, d_mod_o, rotated_s

        eh_o = self._convert(curvature, e_o)
        eh_all = self._convert(curvature, e_all)

        rotated_all = self._rotate(theta_r, eh_all)
        d_mod_s = self._squared_distance(
            curvature, self._add(curvature, rotated_all, rh_r), e_o
        )
        mod_s_pred = -d_mod_s + b_all + b_o

        return mod_o_pred.numpy(), mod_s_pred.numpy()


@experimental(reason="demo is missing", issues=[1664])
class RotH(KGModel):
    """
    Embedding layers and a RotH scoring layer that implement the RotH knowledge graph
    embedding algorithm as in https://arxiv.org/abs/2005.00545

    .. seealso::

       Related models:

       - other knowledge graph models, see :class:`.KGTripleGenerator` for a full list
       - :class:`.RotE` for the Euclidean version of this hyperbolic model

       Appropriate data generator: :class:`.KGTripleGenerator`.

    Args:
        generator (KGTripleGenerator): A generator of triples to feed into the model.

        embedding_dimension (int): the dimension of the embeddings (that is, a vector in
            ``R^embedding_dimension`` plus a bias in ``R`` is learnt for each node, along with a pair of
            vectors in ``R^embedding_dimension`` and ``R^(embedding_dimension / 2)`` for each node
            type). It must be even.

        embeddings_initializer (str or func, optional): The initialiser to use for the embeddings.

        embeddings_regularizer (str or func, optional): The regularizer to use for the embeddings.
    """

    def __init__(
        self,
        generator,
        embedding_dimension,
        embeddings_initializer="normal",
        embeddings_regularizer=None,
    ):
        super().__init__(
            generator,
            RotHEScoring(hyperbolic=True),
            embedding_dimension=embedding_dimension,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
        )


@experimental(reason="demo is missing", issues=[1664])
class RotE(KGModel):
    """
    Embedding layers and a RotE scoring layer that implement the RotE knowledge graph
    embedding algorithm as in https://arxiv.org/pdf/2005.00545.pdf

    .. seealso::

       Related models:

       - other knowledge graph models, see :class:`.KGTripleGenerator` for a full list
       - :class:`.RotH` for the hyperbolic version of this Euclidean model

       Appropriate data generator: :class:`.KGTripleGenerator`.

    Args:
        generator (KGTripleGenerator): A generator of triples to feed into the model.

        embedding_dimension (int): the dimension of the embeddings (that is, a vector in
            ``R^embedding_dimension`` plus a bias in ``R`` is learnt for each node, along with a pair of
            vectors in ``R^embedding_dimension`` and ``R^(embedding_dimension / 2)`` for each node
            type). It must be even.

        embeddings_initializer (str or func, optional): The initialiser to use for the embeddings.

        embeddings_regularizer (str or func, optional): The regularizer to use for the embeddings.
    """

    def __init__(
        self,
        generator,
        embedding_dimension,
        embeddings_initializer="normal",
        embeddings_regularizer=None,
    ):
        super().__init__(
            generator,
            RotHEScoring(hyperbolic=False),
            embedding_dimension=embedding_dimension,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
        )


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
