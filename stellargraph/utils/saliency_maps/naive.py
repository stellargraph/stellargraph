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
from scipy.sparse import coo_matrix
from ...core.graph import StellarGraph
from ...core.element_data import ExternalIdIndex


class PerturbedGraph:
    def __init__(self, graph: StellarGraph, *, node=None, edge=None):
        self._graph = graph
        self._node = node
        self._edge = edge

    def __getattr__(self, item):
        return getattr(self._graph, item)

    def _index(self, nodes):
        if nodes is None:
            return self._graph._nodes._id_index
        else:
            return ExternalIdIndex(list(nodes))

    def to_adjacency_matrix(self, nodes=None):
        adj = self._graph.to_adjacency_matrix(nodes=nodes)

        if self._edge is not None:
            src_iloc, dst_iloc = self._index(nodes).to_iloc(self._edge)
            adj[src_iloc, dst_iloc] = 0
            if not self._graph.is_directed() and src_iloc != dst_iloc:
                adj[dst_iloc, src_iloc] = 0

        return adj

    def node_features(self, nodes, node_type=None):
        features = self._graph.node_features(nodes=nodes, node_type=node_type)

        if self._node is not None:
            node_iloc = self._index(nodes).to_iloc([self._node])
            features[node_iloc, :] = 0

        return features


def node_importance(graph: StellarGraph, predict, nodes=None):
    pred_baseline = predict(graph)

    if nodes is None:
        nodes = graph.nodes()

    def importance(node):
        perturbed = PerturbedGraph(graph, node=node)
        pred = predict(perturbed)
        return pred_baseline - pred

    return np.array([importance(node) for node in nodes])


def edge_importance(graph: StellarGraph, predict, edges=None):
    pred_baseline = predict(graph)
    n = graph.number_of_nodes()
    if edges is None:
        edges = graph.edges()

    def importance(e):
        src, dst = e
        perturbed = PerturbedGraph(graph, edge=[src, dst])
        pred = predict(perturbed)
        return src, dst, pred_baseline - pred

    if not graph.is_directed():
        edges = set((undirected for e in edges for undirected in [e, e[::-1]]))
    srcs, dsts, importances = zip(*(importance(e) for e in edges))
    srcs = graph._nodes._id_index.to_iloc(srcs)
    dsts = graph._nodes._id_index.to_iloc(dsts)

    return coo_matrix((importances, (srcs, dsts)), shape=(n, n)).todense()
