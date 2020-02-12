# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from stellargraph.core.graph import StellarGraph
from stellargraph.core.element_data import ExternalIdIndex


class _GraphWithNodeRemoved(StellarGraph):
    """
    A wrapper around StellarGraph to mimic a node being removed.

    Args:
        graph (StellarGraph): graph to remove a node from
        remove_node (any): node to remove
    """

    def __init__(self, graph, remove_node):
        # this doesn't call super().__init__() which should be fine since this doesn't need access
        # to any members of the superclass. It's only subclassing StellarGraph in order to avoid
        # failing the type checking done throughout the library.
        if not graph.has_node(remove_node):
            raise ValueError(
                f"remove_node: tried to remove '{remove_node}', but it doesn't exist in the provided graph."
            )
        self._graph = graph
        self._remove_node = remove_node

    def __getattr__(self, item):
        return getattr(self._graph, item)

    def node_features(self, nodes, node_type=None):
        features = self._graph.node_features(nodes=nodes, node_type=node_type)
        index = ExternalIdIndex(nodes)
        node_iloc = index.to_iloc([self._remove_node])

        if index.is_valid(node_iloc):
            features[node_iloc, :] = 0

        return features


def node_importance(graph: StellarGraph, predict, nodes=None):
    """
    Calculate importance of nodes for the prediction output.

    Args:
        graph (StellarGraph): Graph to predict on
        predict (callable): Function takes the graph as argument and returns a single numeric
            prediction value.
        nodes (any, optional): IDs of nodes to calculate importance scores for. Defaults to using
            all nodes in the provided graph.

    Returns:
        Numpy array containing the node importance scores in the same order as the provided
        ``nodes`` parameter. If using all nodes, the scores are in the same order as
        ``graph.nodes()``.
    """
    pred_baseline = predict(graph)

    if nodes is None:
        nodes = graph.nodes()

    def importance_of(node):
        perturbed = _GraphWithNodeRemoved(graph, remove_node=node)
        pred = predict(perturbed)
        return pred_baseline - pred

    return np.array([importance_of(node) for node in nodes])
