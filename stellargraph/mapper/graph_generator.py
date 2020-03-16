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
from .sequences import GraphSequence
from ..core.experimental import experimental


@experimental(reason="Missing unit tests.", issues=[1042])
class GraphGenerator:
    """
    A data generator for use with graph classification algorithms.

    The supplied graphs should be :class:`StellarGraph` objects ready for machine learning. The generator
    requires node features to be available for all nodes in the graph.
    Use the :meth:`flow` method supplying the graph indexes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    This generator supplies the features arrays and the adjacency matrices to a
    mini-batch Keras graph classification model.

    Args:
        graphs (list): a collection of ready for machine-learning StellarGraph-type objects
        name (str): an optional name of the generator
    """

    def __init__(self, graphs, name=None):

        self.node_features_size = None
        for graph in graphs:
            if not isinstance(graph, StellarGraph):
                raise TypeError(
                    f"graphs: expected every element to be a StellarGraph object, found {type(graph).__name__}."
                )
            if len(graph.node_types) > 1:
                raise ValueError(
                    "graphs: node generator requires graphs with single node type, "
                    f"found a graph with {len(graph.node_types)} node types."
                )

            graph.check_graph_for_ml()

            # we require that all graphs have node features of the same dimensionality
            f_dim = graph.node_feature_sizes()[list(graph.node_types)[0]]
            if self.node_features_size is None:
                self.node_features_size = f_dim
            elif self.node_features_size != f_dim:
                raise ValueError(
                    "graphs: expected node features for all graph to have same dimensions,"
                    f"found {self.node_features_size} vs {f_dim}"
                )

        self.graphs = graphs
        self.name = name

    def flow(self, graph_ilocs, targets=None, batch_size=1, name=None):
        """
        Creates a generator/sequence object for training, evaluation, or prediction
        with the supplied graph indexes and targets.

        Args:
            graph_ilocs (iterable): an iterable of graph indexes in self.graphs for the graphs of interest
                (e.g., training, validation, or test set nodes).
            targets (2d array, optional): a 2D array of numeric graph targets with shape `(len(graph_ilocs),
                len(targets))`.
            batch_size (int, optional): The batch size.
            name (str, optional): An optional name for the returned generator object.

        Returns:
            A :class:`GraphSequence` object to use with Keras methods :meth:`fit`, :meth:`evaluate`, and :meth:`predict`

        """
        if targets is not None:
            # Check targets is an iterable
            if not is_real_iterable(targets):
                raise TypeError(
                    f"targets: expected an iterable or None object, found {type(targets).__name__}"
                )

            # Check targets correct shape
            if len(targets) != len(graph_ilocs):
                raise ValueError(
                    f"expected targets to be the same length as node_ids, found {len(targets)} vs {len(graph_ilocs)}"
                )

        return GraphSequence(
            graphs=[self.graphs[i] for i in graph_ilocs],
            targets=targets,
            batch_size=batch_size,
            name=name,
        )
