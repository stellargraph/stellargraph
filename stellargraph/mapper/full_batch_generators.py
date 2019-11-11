# -*- coding: utf-8 -*-
#
# Copyright 2018-2019 Data61, CSIRO
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

"""
Mappers to provide input data for the graph models in layers.

"""
__all__ = ["FullBatchNodeGenerator"]

import warnings
import operator
import random
import numpy as np
import itertools as it
import networkx as nx
import scipy.sparse as sps
from tensorflow.keras import backend as K
from functools import reduce
from tensorflow.keras.utils import Sequence

from . import FullBatchNodeSequence, SparseFullBatchNodeSequence
from ..core.graph import StellarGraphBase, GraphSchema, StellarDiGraph
from ..core.utils import is_real_iterable
from ..core.utils import GCN_Aadj_feats_op, PPNP_Aadj_feats_op


class FullBatchNodeGenerator:
    """
    A data generator for use with full-batch models on homogeneous graphs,
    e.g., GCN, GAT, SGC.
    The supplied graph G should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features to be available for all
    nodes in the graph.
    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    This generator will supply the features array and the adjacency matrix to a
    full-batch Keras graph ML model.  There is a choice to supply either a sparse
    adjacency matrix (the default) or a dense adjacency matrix, with the `sparse`
    argument.

    For these algorithms the adjacency matrix requires pre-processing and the
    'method' option should be specified with the correct pre-processing for
    each algorithm. The options are as follows:

    *   ``method='gcn'`` Normalizes the adjacency matrix for the GCN algorithm.
        This implements the linearized convolution of Eq. 8 in [1].
    *   ``method='chebyshev'``: Implements the approximate spectral convolution
        operator by implementing the k-th order Chebyshev expansion of Eq. 5 in [1].
    *   ``method='sgc'``: This replicates the k-th order smoothed adjacency matrix
        to implement the Simplified Graph Convolutions of Eq. 8 in [2].
    *   ``method='self_loops'`` or ``method='gat'``: Simply sets the diagonal elements
        of the adjacency matrix to one, effectively adding self-loops to the graph. This is
        used by the GAT algorithm of [3].
    *   ``method='ppnp'`` Calculates the personalized page rank matrix of Eq 2 in [4].

    [1] `Kipf and Welling, 2017 <https://arxiv.org/abs/1609.02907>`_.
    [2] `Wu et al. 2019 <https://arxiv.org/abs/1902.07153>`_.
    [3] `Veličković et al., 2018 <https://arxiv.org/abs/1710.10903>`_
    [4] `Klicpera et al., 2018 <https://arxiv.org/abs/1810.05997>`_.

    Example::

        G_generator = FullBatchNodeGenerator(G)
        train_data_gen = G_generator.flow(node_ids, node_targets)

        # Fetch the data from train_data_gen, and feed into a Keras model:
        x_inputs, y_train = train_data_gen[0]
        model.fit(x=x_inputs, y=y_train)

        # Alternatively, use the generator itself with model.fit_generator:
        model.fit_generator(train_gen, epochs=num_epochs, ...)

    For more information, please see the GCN/GAT, PPNP/APPNP and SGC demos:
        `<https://github.com/stellargraph/stellargraph/blob/master/demos/>`_

    Args:
        G (StellarGraphBase): a machine-learning StellarGraph-type graph
        name (str): an optional name of the generator
        method (str): Method to pre-process adjacency matrix. One of 'gcn' (default),
            'chebyshev','sgc', 'self_loops', or 'none'.
        k (None or int): This is the smoothing order for the 'sgc' method or the
            Chebyshev series order for the 'chebyshev' method. In both cases this
            should be positive integer.
        transform (callable): an optional function to apply on features and adjacency matrix
            the function takes (features, Aadj) as arguments.
        sparse (bool): If True (default) a sparse adjacency matrix is used,
            if False a dense adjacency matrix is used.
        teleport_probability (float): teleport probability between 0.0 and 1.0. "probability" of returning to the
        starting node in the propagation step as in [4].
    """

    def __init__(
        self,
        G,
        name=None,
        method="gcn",
        k=1,
        sparse=True,
        transform=None,
        teleport_probability=0.1,
    ):

        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        self.graph = G
        self.name = name
        self.k = k
        self.teleport_probability = teleport_probability
        self.method = method

        # Check if the graph has features
        G.check_graph_for_ml()

        # Create sparse adjacency matrix
        self.node_list = list(G.nodes())
        self.Aadj = nx.to_scipy_sparse_matrix(
            G, nodelist=self.node_list, dtype="float32", weight="weight", format="coo"
        )

        # Power-user feature: make the generator yield dense adjacency matrix instead
        # of the default sparse one.
        # If sparse is specified, check that the backend is tensorflow
        if sparse and K.backend() != "tensorflow":
            warnings.warn(
                "Sparse adjacency matrices are only supported in tensorflow."
                " Falling back to using a dense adjacency matrix."
            )
            self.use_sparse = False

        else:
            self.use_sparse = sparse

        # We need a schema to check compatibility with GAT, GCN
        self.schema = G.create_graph_schema(create_type_maps=True)

        # Check that there is only a single node type for GAT or GCN
        if len(self.schema.node_types) > 1:
            raise TypeError(
                "{}: node generator requires graph with single node type; "
                "a graph with multiple node types is passed. Stopping.".format(
                    type(self).__name__
                )
            )

        # Get the features for the nodes
        self.features = G.get_feature_for_nodes(self.node_list)

        if transform is not None:
            if callable(transform):
                self.features, self.Aadj = transform(
                    features=self.features, A=self.Aadj
                )
            else:
                raise ValueError("argument 'transform' must be a callable.")

        elif self.method in ["gcn", "chebyshev", "sgc"]:
            self.features, self.Aadj = GCN_Aadj_feats_op(
                features=self.features, A=self.Aadj, k=self.k, method=self.method
            )

        elif self.method in ["gat", "self_loops"]:
            self.Aadj = self.Aadj + sps.diags(
                np.ones(self.Aadj.shape[0]) - self.Aadj.diagonal()
            )

        elif self.method in ["ppnp"]:
            if self.use_sparse:
                raise ValueError(
                    "use_sparse=true' is incompatible with 'ppnp'."
                    "Set 'use_sparse=True' or consider using the APPNP model instead."
                )
            self.features, self.Aadj = PPNP_Aadj_feats_op(
                features=self.features,
                A=self.Aadj,
                teleport_probability=self.teleport_probability,
            )

        elif self.method in [None, "none"]:
            pass

        else:
            raise ValueError(
                "Undefined method for adjacency matrix transformation. "
                "Accepted: 'gcn' (default), 'chebyshev','sgc', and 'self_loops'."
            )

    def flow(self, node_ids, targets=None):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_ids: and iterable of node ids for the nodes of interest
                (e.g., training, validation, or test set nodes)
            targets: a 2D array of numeric node targets with shape `(len(node_ids), target_size)`

        Returns:
            A NodeSequence object to use with GCN or GAT models
            in Keras methods :meth:`fit_generator`, :meth:`evaluate_generator`,
            and :meth:`predict_generator`

        """
        if targets is not None:
            # Check targets is an iterable
            if not is_real_iterable(targets):
                raise TypeError("Targets must be an iterable or None")

            # Check targets correct shape
            if len(targets) != len(node_ids):
                raise TypeError("Targets must be the same length as node_ids")

        # Dictionary to store node indices for quicker node index lookups
        node_lookup = dict(zip(self.node_list, range(len(self.node_list))))

        # The list of indices of the target nodes in self.node_list
        node_indices = np.array([node_lookup[n] for n in node_ids])

        if self.use_sparse:
            return SparseFullBatchNodeSequence(
                self.features, self.Aadj, targets, node_indices
            )
        else:
            return FullBatchNodeSequence(
                self.features, self.Aadj, targets, node_indices
            )
