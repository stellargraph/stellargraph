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
from stellargraph.mapper import ClusterNodeGenerator, ClusterNodeSequence
from stellargraph.core.graph import StellarGraph

import networkx as nx
import pandas as pd
import numpy as np
import pytest


def create_graph_features():
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c", "d"])
    G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c"), ("b", "d")])
    return G, np.array([[1, 1], [1, 0], [0, 1], [0.5, 1]])


def create_stellargraph():
    Gnx, features = create_graph_features()
    nodes = Gnx.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(Gnx, node_type_name="node", node_features=node_features)

    return G


def test_ClusterNodeSequence_init():

    G = create_stellargraph()

    nsg = ClusterNodeSequence(graph=G, clusters=[list(G.nodes())])
    assert len(nsg) == 1

    nsg = ClusterNodeSequence(graph=G, clusters=[["a"], ["b", "d"], ["c"]])
    assert len(nsg) == 3

    # If targets are given, so should node_ids that correspond to these targets
    with pytest.raises(ValueError):
        ClusterNodeSequence(
            graph=G, clusters=[list(G.nodes())], q=1, targets=np.array([[0, 1]])
        )

    # targets and node_id should have the same length.
    with pytest.raises(ValueError):
        ClusterNodeSequence(
            graph=G,
            clusters=[list(G.nodes())],
            q=1,
            targets=np.array([[0, 1], [1, 0]]),
            node_ids=["a"],
        )

    # len(clusters) is not exactly divisible by q
    with pytest.raises(ValueError):
        ClusterNodeSequence(graph=G, clusters=[list(G.nodes())], q=2)


def test_ClusterNodeSequence_getitem():

    G = create_stellargraph()

    nsg = ClusterNodeSequence(
        graph=G, clusters=[["a"], ["b"], ["c"], ["d"]], node_ids=["a", "b", "d"]
    )

    # 4 clusters with each cluster having a single node
    assert len(nsg) == 4

    for cluster in list(nsg):
        print(cluster)
        assert len(cluster) == 2
        # [features, target_node_indices, adj_cluster], cluster_targets
        assert len(cluster[0][0]) == 1
        assert len(cluster[0][1]) == 1
        assert cluster[0][2].shape == (
            1,
            1,
            1,
        )  # one node per cluster so adjacency matrix is 1x1
        assert cluster[1] is None  # no targets given

    # only 3 node_ids were specified
    assert len(nsg.node_order) == 3

    nodes = set()
    for node in nsg.node_order:
        nodes.add(node)

    assert len(nodes.intersection(["a", "b", "d"])) == 3


def test_ClusterNodeGenerator_init():

    G = create_stellargraph()

    with pytest.raises(ValueError):
        generator = ClusterNodeGenerator(G, k=0)

    # k must be integer
    with pytest.raises(TypeError):
        generator = ClusterNodeGenerator(G, k=0.5)

    with pytest.raises(ValueError):
        generator = ClusterNodeGenerator(G, q=0)

    # q must be integer
    with pytest.raises(TypeError):
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

    # lam has to be in the interval [0., 1.] and float
    with pytest.raises(TypeError):
        generator = ClusterNodeGenerator(G, k=1, q=1, lam=-1)

    with pytest.raises(TypeError):
        generator = ClusterNodeGenerator(G, k=1, q=1, lam=1)

    with pytest.raises(ValueError):
        generator = ClusterNodeGenerator(G, k=1, q=1, lam=2.5)


def test_ClusterNodeSquence():

    G = create_stellargraph()

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
        assert batch[1] is None
