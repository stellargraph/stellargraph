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


from tensorflow.keras import backend as K
from stellargraph.layer.cluster_gcn import *
from stellargraph.mapper import ClusterNodeGenerator
from stellargraph.core.graph import StellarGraph

import networkx as nx
import pandas as pd
import numpy as np
from tensorflow import keras
import pytest


def create_graph_features():
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c", "d"])
    G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c"), ("b", "d")])
    G = G.to_undirected()
    return G, np.array([[1, 1], [1, 0], [0, 1], [0.5, 1]])


def test_ClusterNodeGenerator_init():
    G, features = create_graph_features()
    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_type_name="node", node_features=node_features)

    with pytest.raises(ValueError):
        generator = ClusterNodeGenerator(G, k=0)

    # k must be integer
    with pytest.raises(ValueError):
        generator = ClusterNodeGenerator(G, k=0.5)

    with pytest.raises(ValueError):
        generator = ClusterNodeGenerator(G, q=0)

    # q must be integer
    with pytest.raises(ValueError):
        generator = ClusterNodeGenerator(G, q=1.0)

    # k is not exactly divisible by q
    with pytest.raises(ValueError):
        generator = ClusterNodeGenerator(G, k=5, q=2)

    # one cluster, k=len(clusters), not divisible by q
    with pytest.raises(ValueError):
        generator = ClusterNodeGenerator(G, clusters=[["a", "b", "c", "d"]], q=2)

    generator = ClusterNodeGenerator(G, clusters=[["a", "b", "c", "d"]], q=1)
    assert generator.k == 1
    assert generator.q == 1

    # two clusters, k=len(clusters).
    generator = ClusterNodeGenerator(G, clusters=[["a", "d"], ["b", "c"]], q=1)
    assert generator.k == 2
    assert generator.q == 1


def test_ClusterNodeSquence():
    G, features = create_graph_features()
    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_type_name="node", node_features=node_features)

    generator = ClusterNodeGenerator(G, k=1, q=1).flow(node_ids=["a", "b", "c", "d"])

    assert len(generator) == 1

    generator = ClusterNodeGenerator(G, k=4, q=1).flow(node_ids=["a", "b", "c", "d"])
    assert len(generator) == 4

    generator = ClusterNodeGenerator(G, k=4, q=1).flow(node_ids=["a", "b", "c", "d"])

    # ClusterNodeSequence returns the following:
    #      [features, target_node_indices, adj_cluster], cluster_targets
    for batch in generator:
        assert len(batch) == 2
        # The first dimension is the batch dimension necessary to make this work with Keras
        assert batch[0][0].shape == (1, 1, 2)
        assert batch[0][1].shape == (1, 1, 1)
        # one node so that adjacency matrix is 1x1
        assert batch[0][2].shape == (1, 1, 1)
        # no targets given
        assert batch[1] == None

    # Use 2 clusters
    generator = ClusterNodeGenerator(G, k=2, q=1).flow(node_ids=["a", "b", "c", "d"])
    assert len(generator) == 2

    # ClusterNodeSequence returns the following:
    #      [features, target_node_indices, adj_cluster], cluster_targets
    for batch in generator:
        assert len(batch) == 2
        # The first dimension is the batch dimension necessary to make this work with Keras
        assert batch[0][0].shape == (1, 2, 2)
        assert batch[0][1].shape == (1, 1, 2)
        # two nodes so that adjacency matrix is 2x2
        assert batch[0][2].shape == (1, 2, 2)
        # no targets given
        assert batch[1] == None
