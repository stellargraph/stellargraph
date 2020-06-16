# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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

import pandas as pd
import numpy as np
import pytest

from ..test_utils.graphs import example_graph_random


# FIXME (#535): Consider using graph fixtures
def create_stellargraph():
    nodes = pd.DataFrame([[1, 1], [1, 0], [0, 1], [0.5, 1]], index=["a", "b", "c", "d"])
    edges = pd.DataFrame(
        [("a", "b", 1.0), ("b", "c", 0.4), ("a", "c", 2.0), ("b", "d", 10.0)],
        columns=["source", "target", "weight"],
    )
    return StellarGraph(nodes, edges)


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
        generator = ClusterNodeGenerator(G, clusters=0)

    # clusters must be integer if not list
    with pytest.raises(TypeError):
        generator = ClusterNodeGenerator(G, clusters=0.5)

    # q must be greater than 0
    with pytest.raises(ValueError):
        generator = ClusterNodeGenerator(G, q=0)

    # q must be integer
    with pytest.raises(TypeError):
        generator = ClusterNodeGenerator(G, q=1.0)

    # number of clusters is not exactly divisible by q
    with pytest.raises(ValueError):
        generator = ClusterNodeGenerator(G, clusters=5, q=2)

    # one cluster, k=len(clusters), not divisible by q
    with pytest.raises(ValueError):
        generator = ClusterNodeGenerator(G, clusters=[["a", "b", "c", "d"]], q=2)

    # this should be ok
    generator = ClusterNodeGenerator(G, clusters=[["a", "b", "c", "d"]], q=1)
    assert generator.k == 1
    assert generator.q == 1

    # two clusters, len(clusters).
    generator = ClusterNodeGenerator(G, clusters=[["a", "d"], ["b", "c"]], q=1)
    assert generator.k == 2
    assert generator.q == 1

    # lam has to be in the interval [0., 1.] and float
    with pytest.raises(TypeError):
        generator = ClusterNodeGenerator(G, clusters=1, q=1, lam=-1)

    with pytest.raises(TypeError):
        generator = ClusterNodeGenerator(G, clusters=1, q=1, lam=1)

    with pytest.raises(ValueError):
        generator = ClusterNodeGenerator(G, clusters=1, q=1, lam=2.5)


def test_ClusterNodeSquence():

    G = create_stellargraph()

    generator = ClusterNodeGenerator(G, clusters=1, q=1).flow(
        node_ids=["a", "b", "c", "d"]
    )

    assert len(generator) == 1

    generator = ClusterNodeGenerator(G, clusters=4, q=1).flow(
        node_ids=["a", "b", "c", "d"]
    )
    assert len(generator) == 4

    generator = ClusterNodeGenerator(G, clusters=4, q=1).flow(
        node_ids=["a", "b", "c", "d"]
    )

    # ClusterNodeSequence returns the following:
    #      [features, target_node_indices, adj_cluster], cluster_targets
    for batch in generator:
        assert len(batch) == 2
        # The first dimension is the batch dimension necessary to make this work with Keras
        assert batch[0][0].shape == (1, 1, 2)
        assert batch[0][1].shape == (1, 1)
        # one node so that adjacency matrix is 1x1
        assert batch[0][2].shape == (1, 1, 1)
        # no targets given
        assert batch[1] is None

    # Use 2 clusters
    generator = ClusterNodeGenerator(G, clusters=2, q=1).flow(
        node_ids=["a", "b", "c", "d"]
    )
    assert len(generator) == 2

    # ClusterNodeSequence returns the following:
    #      [features, target_node_indices, adj_cluster], cluster_targets
    for batch in generator:
        assert len(batch) == 2
        # The first dimension is the batch dimension necessary to make this work with Keras
        assert batch[0][0].shape == (1, 2, 2)
        assert batch[0][1].shape == (1, 2)
        # two nodes so that adjacency matrix is 2x2
        assert batch[0][2].shape == (1, 2, 2)
        # no targets given
        assert batch[1] is None


def test_cluster_weighted():

    G = create_stellargraph()

    unweighted = ClusterNodeGenerator(G, clusters=1, q=1, weighted=False).flow(
        node_ids=["a", "b", "c", "d"]
    )
    weighted = ClusterNodeGenerator(G, clusters=1, q=1, weighted=True).flow(
        node_ids=["a", "b", "c", "d"]
    )

    assert len(unweighted) == len(weighted) == 1
    unweighted_features, _ = unweighted[0]
    weighted_features, _ = weighted[0]

    def canonical(adj):
        return np.sort(adj.ravel())

    assert not np.allclose(
        canonical(weighted_features[2]), canonical(unweighted_features[2])
    )


@pytest.mark.benchmark(group="ClusterGCN generator")
@pytest.mark.parametrize("q", [1, 2, 10])
def test_benchmark_ClusterGCN_generator(benchmark, q):
    G = example_graph_random(feature_size=10, n_nodes=1000, n_edges=5000)

    generator = ClusterNodeGenerator(G, clusters=10, q=q)
    seq = generator.flow(G.nodes())

    # iterate over all the batches
    benchmark(lambda: list(seq))


def test_ClusterNodeSequence_cluster_without_targets():
    G = create_stellargraph()
    generator = ClusterNodeGenerator(G, clusters=2, q=1)
    seq = generator.flow(node_ids=["a"], targets=[0])
    _ = list(seq)
