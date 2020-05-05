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
from ..core.graph import StellarGraph
from ..core.utils import is_real_iterable
from .sequences import PaddedGraphSequence
import numpy as np
from .base import Generator


class PaddedGraphGenerator(Generator):
    """
    A data generator for use with graph classification algorithms.

    The supplied graphs should be :class:`StellarGraph` objects with node features.
    Use the :meth:`flow` method supplying the graph indexes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    This generator supplies the features arrays and the adjacency matrices to a mini-batch Keras
    graph classification model. Differences in the number of nodes are resolved by padding each
    batch of features and adjacency matrices, and supplying a boolean mask indicating which are
    valid and which are padding.

    Args:
        graphs (list): a collection of StellarGraph objects
        name (str): an optional name of the generator
    """

    def __init__(self, graphs, name=None):

        self.node_features_size = None
        self._check_graphs(graphs)

        self.graphs = graphs
        self.name = name

    def _check_graphs(self, graphs):
        for graph in graphs:
            if not isinstance(graph, StellarGraph):
                raise TypeError(
                    f"graphs: expected every element to be a StellarGraph object, found {type(graph).__name__}."
                )

            if graph.number_of_nodes() == 0:
                # an empty graph has no information at all and breaks things like mean pooling, so
                # let's disallow them
                raise ValueError(
                    "graphs: expected every graph to be non-empty, found graph with no nodes"
                )

            # Check that there is only a single node type for GAT or GCN
            node_type = graph.unique_node_type(
                "graphs: expected only graphs with a single node type, found a graph with node types: %(found)s"
            )

            graph.check_graph_for_ml()

            # we require that all graphs have node features of the same dimensionality
            f_dim = graph.node_feature_sizes()[node_type]
            if self.node_features_size is None:
                self.node_features_size = f_dim
            elif self.node_features_size != f_dim:
                raise ValueError(
                    "graphs: expected node features for all graph to have same dimensions,"
                    f"found {self.node_features_size} vs {f_dim}"
                )

    def num_batch_dims(self):
        return 1

    def flow(
        self,
        graphs,
        targets=None,
        symmetric_normalization=True,
        batch_size=1,
        name=None,
        shuffle=False,
        seed=None,
    ):
        """
        Creates a generator/sequence object for training, evaluation, or prediction
        with the supplied graph indexes and targets.

        Args:
            graphs (iterable): an iterable of graph indexes in self.graphs or an iterable of `StellarGraph`s
                for the graphs of interest (e.g., training, validation, or test set nodes).
            targets (2d array, optional): a 2D array of numeric graph targets with shape ``(len(graphs),
                len(targets))``.
            symmetric_normalization (bool, optional): The type of normalization to be applied on the graph adjacency
                matrices. If True, the adjacency matrix is left and right multiplied by the inverse square root of the
                degree matrix; otherwise, the adjacency matrix is only left multiplied by the inverse of the degree
                matrix.
            batch_size (int, optional): The batch size.
            name (str, optional): An optional name for the returned generator object.
            shuffle (bool, optional): If True the node IDs will be shuffled at the end of each epoch.
            seed (int, optional): Random seed to use in the sequence object.

        Returns:
            A :class:`PaddedGraphSequence` object to use with Keras methods :meth:`fit`, :meth:`evaluate`, and :meth:`predict`

        """
        if targets is not None:
            # Check targets is an iterable
            if not is_real_iterable(targets):
                raise TypeError(
                    f"targets: expected an iterable or None object, found {type(targets).__name__}"
                )

            # Check targets correct shape
            if len(targets) != len(graphs):
                raise ValueError(
                    f"expected targets to be the same length as node_ids, found {len(targets)} vs {len(graphs)}"
                )

        if not isinstance(batch_size, int):
            raise TypeError(
                f"expected batch_size to be integer type, found {type(batch_size).__name__}"
            )

        if batch_size <= 0:
            raise ValueError(
                f"expected batch_size to be strictly positive integer, found {batch_size}"
            )

        if isinstance(graphs[0], StellarGraph):
            self._check_graphs(graphs)
        else:
            graphs = [self.graphs[i] for i in graphs]

        return PaddedGraphSequence(
            graphs=graphs,
            targets=targets,
            symmetric_normalization=symmetric_normalization,
            batch_size=batch_size,
            name=name,
            shuffle=shuffle,
            seed=seed,
        )
