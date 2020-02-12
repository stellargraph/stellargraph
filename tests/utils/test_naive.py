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

from ..test_utils.graphs import example_graph
from stellargraph.core import StellarGraph
from stellargraph.utils.saliency_maps.naive import edge_importance, node_importance
import numpy as np
import pytest


def predict_node_based(graph: StellarGraph):
    return graph.node_features(graph.nodes()).sum()


def predict_edge_based(graph: StellarGraph):
    return graph.to_adjacency_matrix().sum()


def expected_undirected_edge_importance(graph, edges=None):
    n = graph.number_of_nodes()
    adj = graph.to_adjacency_matrix().todense()
    if edges is not None:
        srcs, dsts = zip(*edges)
        src_ilocs = graph._nodes._id_index.to_iloc(srcs)
        dst_ilocs = graph._nodes._id_index.to_iloc(dsts)
        srcs = np.concatenate([src_ilocs, dst_ilocs], axis=0)
        dsts = np.concatenate([dst_ilocs, src_ilocs], axis=0)
        mask = np.ones((n, n), dtype=bool)
        mask[srcs, dsts] = False
        adj[mask] = 0
    return adj.T + adj - np.eye(adj.shape[0]) * np.array(adj.diagonal())


def expected_node_importance(graph, nodes=None):
    expected = graph.node_features(graph.nodes()).sum(axis=-1)
    if nodes is not None:
        node_ilocs = graph._nodes._id_index.to_iloc(nodes)
        return expected[node_ilocs]
    else:
        return expected


@pytest.mark.parametrize("edges", [None, [(1, 2), (4, 2)]])
def test_edge_importance(edges):
    graph = example_graph(feature_size=4)
    expected = expected_undirected_edge_importance(graph, edges=edges)
    importances = edge_importance(graph, predict=predict_edge_based, edges=edges)
    assert np.allclose(importances, expected)


def test_edge_importance_zeros():
    graph = example_graph(feature_size=4)
    importances = edge_importance(graph, predict=predict_node_based)
    n = graph.number_of_nodes()
    assert np.allclose(importances, np.zeros((n, n)))


@pytest.mark.parametrize("nodes", [None, [1, 3]])
def test_node_importance(nodes):
    graph = example_graph(feature_size=4)
    expected = expected_node_importance(graph, nodes=nodes)
    importances = node_importance(graph, predict=predict_node_based, nodes=nodes)
    assert np.allclose(importances, expected)


def test_node_importance_zeros():
    n_feat = 4
    graph = example_graph(feature_size=n_feat)
    importances = node_importance(graph, predict=predict_edge_based)
    assert np.allclose(importances, np.zeros((graph.number_of_nodes(), n_feat)))
