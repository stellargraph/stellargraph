# -*- coding: utf-8 -*-
#
# Copyright 2018 Data61, CSIRO
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


"""
Mapper tests:

"""
from stellargraph.core.graph import *
from stellargraph.mapper.node_mappers import *

import networkx as nx
import numpy as np
import random
import pytest


def example_graph_1(feature_size=None):
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_nodes_from([1, 2, 3, 4], label="default")
    G.add_edges_from(elist, label="default")

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = np.ones(feature_size)
        return StellarGraph(G, node_features="feature")

    else:
        return StellarGraph(G)


def example_graph_2(feature_size=None):
    G = nx.Graph()
    elist = [(1, 2), (1, 3), (1, 4), (3, 2), (3, 5)]
    G.add_nodes_from([1, 2, 3, 4, 5], label="default")
    G.add_edges_from(elist, label="default")

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = int(v) * np.ones(feature_size, dtype="int")
        return StellarGraph(G, node_features="feature")

    else:
        return StellarGraph(G)


def example_graph_3(feature_size=None, n_edges=20, n_nodes=6, n_isolates=1):
    G = nx.Graph()
    n_noniso = n_nodes - n_isolates
    edges = [
        (random.randint(0, n_noniso - 1), random.randint(0, n_noniso - 1))
        for _ in range(n_edges)
    ]
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges, label="default")

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = int(v) * np.ones(feature_size, dtype="int")
        return StellarGraph(G, node_features="feature")

    else:
        return StellarGraph(G)


def example_digraph_2(feature_size=None):
    G = nx.DiGraph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = np.ones(feature_size)
        return StellarDiGraph(G, node_features="feature")

    else:
        return StellarDiGraph(G)


def example_hin_1(feature_size_by_type=None):
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3], label="A")
    G.add_nodes_from([4, 5, 6], label="B")
    G.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label="R")
    G.add_edges_from([(4, 5)], label="F")

    # Add example features
    if feature_size_by_type is not None:
        for v, vdata in G.nodes(data=True):
            nt = vdata["label"]
            vdata["feature"] = int(v) * np.ones(feature_size_by_type[nt], dtype="int")
        return StellarGraph(G, node_features="feature")

    else:
        return StellarGraph(G)


def example_hin_2(feature_size_by_type=None):
    nodes_type_1 = [0, 1, 2, 3]
    nodes_type_2 = [4, 5]

    # Create isolated graphs
    G = nx.Graph()
    G.add_nodes_from(nodes_type_1, label="t1")
    G.add_nodes_from(nodes_type_2, label="t2")
    G.add_edges_from([(0, 4), (1, 4), (2, 5), (3, 5)], label="e1")

    # Add example features
    if feature_size_by_type is not None:
        for v, vdata in G.nodes(data=True):
            nt = vdata["label"]
            vdata["feature"] = int(v) * np.ones(feature_size_by_type[nt], dtype="int")

        G = StellarGraph(G, node_features="feature")

    else:
        G = StellarGraph(G)

    return G, nodes_type_1, nodes_type_2


def example_hin_3(feature_size_by_type=None):
    nodes_type_1 = [0, 1, 2]
    nodes_type_2 = [4, 5, 6]

    # Create isolated graphs
    G = nx.Graph()
    G.add_nodes_from(nodes_type_1, label="t1")
    G.add_nodes_from(nodes_type_2, label="t2")
    G.add_edges_from([(0, 4), (1, 5)], label="e1")
    G.add_edges_from([(0, 2)], label="e2")

    # Node 2 has no edges of type 1
    # Node 1 has no edges of type 2
    # Node 6 has no edges

    # Add example features
    if feature_size_by_type is not None:
        for v, vdata in G.nodes(data=True):
            nt = vdata["label"]
            vdata["feature"] = (int(v) + 10) * np.ones(
                feature_size_by_type[nt], dtype="int"
            )

        G = StellarGraph(G, node_features="feature")

    else:
        G = StellarGraph(G)

    return G, nodes_type_1, nodes_type_2


def test_nodemapper_constructor_nx():
    """
    GraphSAGENodeGenerator requires a StellarGraph object
    """
    G = nx.Graph()
    G.add_nodes_from(range(4))

    with pytest.raises(TypeError):
        GraphSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2])


def test_nodemapper_constructor_no_feats():
    """
    GraphSAGENodeGenerator requires the graph to have features
    """
    n_feat = 4

    G = example_graph_1()
    with pytest.raises(RuntimeError):
        GraphSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2])


def test_nodemapper_constructor():
    n_feat = 4

    G = example_graph_1(feature_size=n_feat)

    generator = GraphSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2])

    mapper = generator.flow(list(G))

    assert generator.batch_size == 2
    assert mapper.data_size == 4
    assert len(mapper.ids) == 4


def test_nodemapper_1():
    n_feat = 4
    n_batch = 2

    # test graph
    G1 = example_graph_1(n_feat)

    mapper1 = GraphSAGENodeGenerator(G1, batch_size=n_batch, num_samples=[2, 2]).flow(
        G1.nodes()
    )
    assert len(mapper1) == 2

    G2 = example_graph_2(n_feat)

    mapper2 = GraphSAGENodeGenerator(G2, batch_size=n_batch, num_samples=[2, 2]).flow(
        G2.nodes()
    )
    assert len(mapper2) == 3

    for mapper in [mapper1, mapper2]:
        for ii in range(2):
            nf, nl = mapper[ii]
            assert len(nf) == 3
            assert nf[0].shape == (n_batch, 1, n_feat)
            assert nf[1].shape == (n_batch, 2, n_feat)
            assert nf[2].shape == (n_batch, 2 * 2, n_feat)
            assert nl is None

    # Check beyond the graph lengh
    with pytest.raises(IndexError):
        nf, nl = mapper1[len(mapper1)]

    # Check the last batch
    nf, nl = mapper2[len(mapper2) - 1]
    assert nf[0].shape == (1, 1, n_feat)
    assert nf[1].shape == (1, 2, n_feat)
    assert nf[2].shape == (1, 2 * 2, n_feat)


def test_nodemapper_with_labels():
    n_feat = 4
    n_batch = 2

    # test graph
    G2 = example_graph_2(n_feat)
    nodes = list(G2)
    labels = [n * 2 for n in nodes]

    gen = GraphSAGENodeGenerator(G2, batch_size=n_batch, num_samples=[2, 2]).flow(
        nodes, labels
    )
    assert len(gen) == 3

    for ii in range(3):
        nf, nl = gen[ii]

        # Check sizes - note batch sizes are (2,2,1) for each iteration
        assert len(nf) == 3
        assert nf[0].shape[1:] == (1, n_feat)
        assert nf[1].shape[1:] == (2, n_feat)
        assert nf[2].shape[1:] == (2 * 2, n_feat)

        # Check labels
        assert all(int(a) == int(2 * b) for a, b in zip(nl, nf[0][:, 0, 0]))

    # Check beyond the graph lengh
    with pytest.raises(IndexError):
        nf, nl = gen[len(gen)]


def test_nodemapper_no_samples():
    n_feat = 4
    n_batch = 2

    # test graph
    G = example_graph_1(feature_size=n_feat)
    mapper = GraphSAGENodeGenerator(G, batch_size=n_batch, num_samples=[0]).flow(
        G.nodes()
    )

    # This is an edge case, are we sure we want this behaviour?
    assert len(mapper) == 2
    for ii in range(len(mapper)):
        nf, nl = mapper[ii]
        assert len(nf) == 2
        assert nf[0].shape == (n_batch, 1, n_feat)
        assert nf[1].shape == (n_batch, 0, n_feat)
        assert nl is None


def test_nodemapper_isolated_nodes():
    n_feat = 4
    n_batch = 2

    # test graph
    G = example_graph_3(feature_size=n_feat, n_nodes=6, n_isolates=1, n_edges=20)

    # Check connectedness
    ccs = list(nx.connected_components(G))
    assert len(ccs) == 2

    n_isolates = [5]
    assert nx.degree(G, n_isolates[0]) == 0

    # Check both isolated and non-isolated nodes have same sampled feature shape
    for head_nodes in [[1], [2], n_isolates]:
        mapper = GraphSAGENodeGenerator(G, batch_size=n_batch, num_samples=[2, 2]).flow(
            head_nodes
        )
        nf, nl = mapper[0]
        assert nf[0].shape == (1, 1, n_feat)
        assert nf[1].shape == (1, 2, n_feat)
        assert nf[2].shape == (1, 4, n_feat)

    # One isolate and one non-isolate
    mapper = GraphSAGENodeGenerator(G, batch_size=n_batch, num_samples=[2, 2]).flow(
        [1, 5]
    )
    nf, nl = mapper[0]
    assert nf[0].shape == (2, 1, n_feat)
    assert nf[1].shape == (2, 2, n_feat)
    assert nf[2].shape == (2, 4, n_feat)

    # Isolated nodes have the "dummy node" as neighbours
    # Currently, the dummy node has zeros for features – this could change
    assert pytest.approx(nf[1][1]) == 0
    assert pytest.approx(nf[2][2:]) == 0


def test_nodemapper_incorrect_targets():
    """
    Tests checks on target shape
    """
    n_feat = 4
    n_batch = 2

    # test graph
    G = example_graph_1(feature_size=n_feat)

    with pytest.raises(TypeError):
        GraphSAGENodeGenerator(G, batch_size=n_batch, num_samples=[0]).flow(list(G), 1)

    with pytest.raises(ValueError):
        GraphSAGENodeGenerator(G, batch_size=n_batch, num_samples=[0]).flow(
            list(G), targets=[]
        )


def test_hinnodemapper_constructor():
    feature_sizes = {"A": 10, "B": 10}
    G = example_hin_1(feature_sizes)

    # Should fail when head nodes are of different type
    with pytest.raises(ValueError):
        HinSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2]).flow(G.nodes())

    gen = HinSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2])
    mapper = gen.flow([0, 1, 2, 3])
    assert gen.batch_size == 2
    assert mapper.data_size == 4
    assert len(mapper.ids) == 4


def test_hinnodemapper_constructor_all_options():
    feature_sizes = {"A": 10, "B": 10}
    G = example_hin_1(feature_sizes)

    gen = HinSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2])

    nodes_of_type_a = G.nodes_of_type("A")
    mapper = gen.flow(nodes_of_type_a)
    assert gen.batch_size == 2
    assert mapper.data_size == len(nodes_of_type_a)


def test_hinnodemapper_constructor_no_features():
    G = example_hin_1(feature_size_by_type=None)
    with pytest.raises(RuntimeError):
        mapper = HinSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2]).flow(
            G.nodes()
        )


def test_hinnodemapper_level_1():
    batch_size = 2
    feature_sizes = {"t1": 1, "t2": 2}
    G, nodes_type_1, nodes_type_2 = example_hin_2(feature_sizes)

    mapper = HinSAGENodeGenerator(G, batch_size=batch_size, num_samples=[2]).flow(
        nodes_type_2
    )

    schema = G.create_graph_schema()
    sampling_adj = schema.type_adjacency_list(["t2"], 1)

    assert len(mapper) == 1

    # Get a batch!
    batch_feats, batch_targets = mapper[0]

    # Check shapes are (batch_size, nsamples, feature_size)
    assert np.shape(batch_feats[0]) == (2, 1, 2)
    assert np.shape(batch_feats[1]) == (2, 2, 1)

    # Check the types
    assert np.all(batch_feats[0] >= 4)
    assert np.all(batch_feats[1] < 4)


def test_hinnodemapper_level_2():
    batch_size = 2
    feature_sizes = {"t1": 1, "t2": 2}
    G, nodes_type_1, nodes_type_2 = example_hin_2(feature_sizes)

    mapper = HinSAGENodeGenerator(G, batch_size=batch_size, num_samples=[2, 3]).flow(
        nodes_type_2
    )

    schema = G.create_graph_schema()
    sampling_adj = schema.type_adjacency_list(["t2"], 2)

    assert len(mapper) == 1

    # Get a batch!
    batch_feats, batch_targets = mapper[0]

    # Check types match adjacency list
    assert len(batch_feats) == len(sampling_adj)
    for bf, adj in zip(batch_feats, sampling_adj):
        nt = adj[0]
        assert bf.shape[0] == batch_size
        assert bf.shape[2] == feature_sizes[nt]

        batch_node_types = {schema.get_node_type(n) for n in np.ravel(bf)}

        assert len(batch_node_types) == 1
        assert nt in batch_node_types


def test_hinnodemapper_with_labels():
    batch_size = 2
    feature_sizes = {"t1": 1, "t2": 2}
    G, nodes_type_1, nodes_type_2 = example_hin_2(feature_sizes)

    labels = [n * 2 for n in nodes_type_1]

    gen = HinSAGENodeGenerator(G, batch_size=batch_size, num_samples=[2, 3]).flow(
        nodes_type_1, labels
    )
    assert len(gen) == 2

    for ii in range(2):
        nf, nl = gen[ii]

        # Check sizes of neighbours and features (in bipartite graph)
        assert len(nf) == 3
        assert nf[0].shape == (2, 1, 1)
        assert nf[1].shape == (2, 2, 2)
        assert nf[2].shape == (2, 2 * 3, 1)

        # Check labels
        assert all(int(a) == int(2 * b) for a, b in zip(nl, nf[0][:, 0, 0]))

    # Check beyond the graph lengh
    with pytest.raises(IndexError):
        nf, nl = gen[len(gen)]


def test_hinnodemapper_no_neighbors():
    batch_size = 3
    feature_sizes = {"t1": 1, "t2": 1}
    G, nodes_type_1, nodes_type_2 = example_hin_3(feature_sizes)

    mapper = HinSAGENodeGenerator(G, batch_size=batch_size, num_samples=[2, 1]).flow(
        nodes_type_2
    )

    schema = G.create_graph_schema()
    sampling_adj = schema.type_adjacency_list(["t2"], 2)

    assert len(mapper) == 1

    # Get a batch!
    batch_feats, batch_targets = mapper[0]
    assert len(batch_feats) == len(sampling_adj)

    # Head nodes
    assert np.all(np.ravel(batch_feats[0]) == np.array([14, 15, 16]))

    # Next level - node 6 has no neighbours
    assert np.all(batch_feats[1][:, 0, 0] == np.array([10, 11, 0]))

    # Following level has two edge types
    # First edge type (e1): Node 0 has 4, node 1 has 5, and node 6 sampling has terminated
    assert np.all(batch_feats[2][:, 0, 0] == np.array([14, 15, 0]))

    # Second edge type (e2): Node 0 has 2, node 1 has none, and node 6 sampling has terminated
    assert np.all(batch_feats[3][:, 0, 0] == np.array([12, 0, 0]))
