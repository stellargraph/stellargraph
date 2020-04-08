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

from tests.test_utils.graphs import example_graph
from stellargraph.core import StellarGraph
from stellargraph.layer import GCN
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.interpretability.saliency_maps.brute_force import (
    BruteForce,
    node_importance,
)
import numpy as np
import pytest
import tensorflow as tf


def predict_node_based(graph: StellarGraph):
    return graph.node_features(graph.nodes()).sum()


def predict_node_based_expected(graph, nodes=None):
    expected = graph.node_features(graph.nodes()).sum(axis=-1)
    if nodes is not None:
        node_ilocs = graph._nodes.ids.to_iloc(nodes)
        return expected[node_ilocs]
    else:
        return expected


def predict_edge_based(graph: StellarGraph):
    return graph.to_adjacency_matrix().sum()


def test_node_importance_zeros():
    n_feat = 4
    graph = example_graph(feature_size=n_feat)

    # with 'predict_edge_based', node features should have zero influence on prediction
    importances = node_importance(graph, predict=predict_edge_based)
    assert np.allclose(importances, np.zeros((graph.number_of_nodes(), n_feat)))


@pytest.mark.parametrize("nodes", [None, [1, 3]])
def test_node_importance(nodes):
    graph = example_graph(feature_size=4)
    expected = predict_node_based_expected(graph, nodes=nodes)
    importances = node_importance(graph, predict=predict_node_based, nodes=nodes)
    assert np.allclose(importances, expected)


def gcn_model(graph):
    gen = FullBatchNodeGenerator(graph)

    # build gcn model
    inputs, outputs = GCN(
        [4], generator=gen, kernel_initializer="ones"
    ).in_out_tensors()
    pred = tf.keras.layers.Dense(units=1, kernel_initializer="ones")(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=pred)
    model.compile(optimizer="adam", loss=tf.keras.losses.binary_crossentropy)

    return model


def check_gcn_importances(graph, target_node, importances):
    # importance should be greater than zero for nodes connected to target node (inc. self)
    connected_ilocs = graph._nodes.ids.to_iloc(
        graph.neighbors(target_node) + [target_node]
    )
    print(importances)
    assert np.all(importances[connected_ilocs] > 0)

    # importance should be zero for disconnected nodes
    disconnected = np.ones(graph.number_of_nodes(), np.bool)
    disconnected[connected_ilocs] = 0
    assert np.all(importances[disconnected] == 0)


def test_node_importance_gcn():
    graph = example_graph(feature_size=4)
    target_node = 1

    model = gcn_model(graph)

    def predict(graph):
        seq = FullBatchNodeGenerator(graph).flow(node_ids=[target_node])
        return np.squeeze(model.predict(seq))

    importances = node_importance(graph, predict=predict)

    check_gcn_importances(graph, target_node, importances)


def test_brute_force_class_node_importance_gcn():
    graph = example_graph(feature_size=4)
    target_node = 1

    model = gcn_model(graph)
    bf = BruteForce(graph, model, FullBatchNodeGenerator)
    importances = bf.node_importance(target_node, 0)

    check_gcn_importances(graph, target_node, importances)
