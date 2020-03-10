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


class GraphGenerator:
    """
    A data generator for use with graph classification algorithms.

    The supplied graphs should be :class:`StellarGraph` objects ready for machine learning. Currently the model
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

        for graph in graphs:
            if not isinstance(graph, StellarGraph):
                raise TypeError("All graphs must be StellarGraph objects.")

        self.graphs = graphs
        self.name = name
        # we assume that all graphs have node features of the same dimensionality
        self.node_features_size = graphs[0].node_features(graphs[0].nodes()).shape[1]

        # Check if the graph has features
        for graph in self.graphs:
            graph.check_graph_for_ml()

        # Check that there is only a single node type
        for graph in self.graphs:
            if len(graph.node_types) > 1:
                raise ValueError(
                    "{}: node generator requires graph with single node type; "
                    "a graph with multiple node types is passed. Stopping.".format(
                        type(self).__name__
                    )
                )

    def flow(self, graph_ilocs, targets=None, batch_size=1, name=None):
        """
        Creates a generator/sequence object for training, evaluation, or prediction
        with the supplied graph indexes and targets.

        Args:
            graph_ilocs (iterable): an iterable of graph indexes in self.graphs for the graphs of interest
                (e.g., training, validation, or test set nodes).
            targets (2d array, optional): a 2D array of numeric graph targets with shape `(len(graph_ilocs),
                len(targets))`.
            batch_size (int, optional): The batch size that defaults to 1.
            name (str, optional): An optional name for the returned generator object.

        Returns:
            A :class:`GraphSequence` object to use with Keras methods :meth:`fit`, :meth:`evaluate`, and :meth:`predict`

        """
        if targets is not None:
            # Check targets is an iterable
            if not is_real_iterable(targets):
                raise TypeError(
                    "{}: Targets must be an iterable or None".format(
                        type(self).__name__
                    )
                )

            # Check targets correct shape
            if len(targets) != len(graph_ilocs):
                raise ValueError(
                    "{}: Targets must be the same length as node_ids".format(
                        type(self).__name__
                    )
                )

        return GraphSequence(
            graphs=[self.graphs[i] for i in graph_ilocs],
            targets=targets,
            batch_size=batch_size,
            name=name,
        )
