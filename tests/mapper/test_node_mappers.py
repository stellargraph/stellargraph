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

GraphSAGENodeMapper(
        G: nx.Graph,
        ids: List[Any],
        sampler: Callable[[List[Any]], List[List[Any]]],
        batch_size: int,
        num_samples: List[int],
        target_id: AnyStr = None,
        feature_size: Optional[int] = None,
        name: AnyStr = None,
    )
g
"""
from stellargraph.data.stellargraph import *
from stellargraph.mapper.node_mappers import *

import networkx as nx
import numpy as np
import itertools as it
import pytest


def example_graph_1(feature_size=None):
    G = StellarGraph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_nodes_from([1, 2, 3, 4], label="default")
    G.add_edges_from(elist, label="default")

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = np.ones(feature_size)
    return G


def example_graph_2(feature_size=None):
    G = StellarGraph()
    elist = [(1, 2), (1, 3), (1, 4), (3, 2), (3, 5)]
    G.add_nodes_from([1, 2, 3, 4, 5], label="default")
    G.add_edges_from(elist, label="default")

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = int(v) * np.ones(feature_size, dtype="int")
    return G


def example_digraph_2(feature_size=None):
    G = StellarDiGraph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = np.ones(feature_size)
    return G


def example_hin_1(feature_size_by_type=None):
    G = StellarGraph()
    G.add_nodes_from([0, 1, 2, 3], label="A")
    G.add_nodes_from([4, 5, 6], label="B")
    G.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label="R")
    G.add_edges_from([(4, 5)], label="F")

    # Add example features
    if feature_size_by_type is not None:
        for v, vdata in G.nodes(data=True):
            nt = vdata["label"]
            vdata["feature"] = int(v) * np.ones(feature_size_by_type[nt], dtype="int")
    return G


def example_hin_2(feature_size_by_type=None):
    nodes_type_1 = [0, 1, 2, 3]
    nodes_type_2 = [4, 5]

    # Create isolated graphs
    G = StellarGraph()
    G.add_nodes_from(nodes_type_1, label="t1")
    G.add_nodes_from(nodes_type_2, label="t2")
    G.add_edges_from([(0, 4), (1, 4), (2, 5), (3, 5)], label="e1")

    # Add example features
    if feature_size_by_type is not None:
        for v, vdata in G.nodes(data=True):
            nt = vdata["label"]
            vdata["feature"] = int(v) * np.ones(feature_size_by_type[nt], dtype="int")
    return G, nodes_type_1, nodes_type_2


def example_hin_3(feature_size_by_type=None):
    nodes_type_1 = [0, 1, 2]
    nodes_type_2 = [4, 5, 6]

    # Create isolated graphs
    G = StellarGraph()
    G.add_nodes_from(nodes_type_1, label="t1")
    G.add_nodes_from(nodes_type_2, label="t2")
    G.add_edges_from([(0, 4), (1, 5)], label="e1")
    G.add_edges_from([(0, 2)], label="e2")

    # Node 2 has no edges of type 1
    # Node 1 has no edges of type 2

    # Add example features
    if feature_size_by_type is not None:
        for v, vdata in G.nodes(data=True):
            nt = vdata["label"]
            vdata["feature"] = (int(v) + 10) * np.ones(
                feature_size_by_type[nt], dtype="int"
            )
    return G, nodes_type_1, nodes_type_2


def test_nodemapper_constructor_nx():
    G = nx.Graph()
    G.add_nodes_from(range(4))

    with pytest.raises(TypeError):
        GraphSAGENodeMapper(G, G.nodes(), batch_size=2, num_samples=[2, 2])


def test_nodemapper_constructor_no_feats():
    n_feat = 4

    G = example_graph_1()
    with pytest.raises(RuntimeError):
        GraphSAGENodeMapper(G, G.nodes(), batch_size=2, num_samples=[2, 2])


def test_nodemapper_constructor():
    n_feat = 4

    G = example_graph_1(feature_size=n_feat)

    # Should raise an error if not set up
    with pytest.raises(RuntimeError):
        GraphSAGENodeMapper(G, G.nodes(), batch_size=2, num_samples=[2, 2])

    G.fit_attribute_spec()
    mapper = GraphSAGENodeMapper(G, G.nodes(), batch_size=2, num_samples=[2, 2])

    assert mapper.batch_size == 2
    assert mapper.data_size == 4
    assert len(mapper.ids) == 4


def test_nodemapper_1():
    n_feat = 4
    n_batch = 2

    # test graph
    G1 = example_graph_1(n_feat)
    G1.fit_attribute_spec()
    mapper1 = GraphSAGENodeMapper(
        G1, G1.nodes(), batch_size=n_batch, num_samples=[2, 2]
    )
    assert len(mapper1) == 2

    G2 = example_graph_2(n_feat)
    G2.fit_attribute_spec()
    mapper2 = GraphSAGENodeMapper(
        G2, G2.nodes(), batch_size=n_batch, num_samples=[2, 2]
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


def test_nodemapper_no_samples():
    n_feat = 4
    n_batch = 2

    # test graph
    G = example_graph_1(feature_size=n_feat)
    G.fit_attribute_spec()
    mapper = GraphSAGENodeMapper(G, G.nodes(), batch_size=n_batch, num_samples=[0])

    # This is an edge case, are we sure we want this behaviour?
    assert len(mapper) == 2
    for ii in range(len(mapper)):
        nf, nl = mapper[ii]
        assert len(nf) == 2
        assert nf[0].shape == (n_batch, 1, n_feat)
        assert nf[1].shape == (n_batch, 0, n_feat)
        assert nl is None


def test_nodemapper_with_targets():
    n_feat = 4
    n_batch = 2

    # test graph
    G = example_graph_1(feature_size=n_feat)

    # Set target attribute
    for n in G:
        G.node[n]["target"] = np.random.choice([0, 1])

    G.fit_attribute_spec()

    nodes = list(G)
    targets = G.get_target_for_nodes(nodes)
    mapper = GraphSAGENodeMapper(
        G, nodes, batch_size=n_batch, num_samples=[1], targets=targets
    )

    assert len(mapper) == 2
    for ii in range(len(mapper)):
        nf, nl = mapper[ii]
        assert len(nf) == 2
        assert nf[0].shape == (n_batch, 1, n_feat)
        assert nf[1].shape == (n_batch, 1, n_feat)
        assert type(nl) == np.ndarray


def test_nodemapper_incorrect_targets():
    n_feat = 4
    n_batch = 2

    # test graph
    G = example_graph_1(feature_size=n_feat)
    G.fit_attribute_spec()

    with pytest.raises(TypeError):
        GraphSAGENodeMapper(G, list(G), batch_size=n_batch, num_samples=[0], targets=1)

    with pytest.raises(ValueError):
        GraphSAGENodeMapper(G, list(G), batch_size=n_batch, num_samples=[0], targets=[])


def test_hinnodemapper_constructor():
    feature_sizes = {"A": 10, "B": 10}
    G = example_hin_1(feature_sizes)
    G.fit_attribute_spec()

    # Should fail when head nodes are of different type
    with pytest.raises(ValueError):
        HinSAGENodeMapper(G, G.nodes(), batch_size=2, num_samples=[2, 2])

    mapper = HinSAGENodeMapper(G, [0, 1, 2, 3], batch_size=2, num_samples=[2, 2])
    assert mapper.batch_size == 2
    assert mapper.data_size == 4
    assert len(mapper.ids) == 4


def test_hinnodemapper_constructor_all_options():
    feature_sizes = {"A": 10, "B": 10}
    G = example_hin_1(feature_sizes)
    G.fit_attribute_spec()

    mapper = HinSAGENodeMapper(
        G, G.nodes(), node_type="A", batch_size=2, num_samples=[2, 2]
    )
    assert mapper.batch_size == 2
    assert mapper.data_size == len(G)
    assert len(mapper.ids) == len(G)


def test_hinnodemapper_constructor_no_features():
    G = example_hin_1(feature_size_by_type=None)
    G.fit_attribute_spec()
    with pytest.raises(RuntimeError):
        mapper = HinSAGENodeMapper(G, G.nodes(), batch_size=2, num_samples=[2, 2])


def test_hinnodemapper_level_1():
    batch_size = 2
    feature_sizes = {"t1": 1, "t2": 2}
    G, nodes_type_1, nodes_type_2 = example_hin_2(feature_sizes)
    G.fit_attribute_spec()

    mapper = HinSAGENodeMapper(
        G, nodes_type_2, node_type="t2", batch_size=batch_size, num_samples=[2]
    )

    schema = G.create_graph_schema()
    sampling_adj = schema.get_type_adjacency_list(["t2"], 1)

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
    G.fit_attribute_spec()

    mapper = HinSAGENodeMapper(
        G, nodes_type_2, node_type="t2", batch_size=batch_size, num_samples=[2, 3]
    )

    schema = G.create_graph_schema()
    sampling_adj = schema.get_type_adjacency_list(["t2"], 2)

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


def test_hinnodemapper_no_neighbors():
    batch_size = 3
    feature_sizes = {"t1": 1, "t2": 1}
    G, nodes_type_1, nodes_type_2 = example_hin_3(feature_sizes)
    G.fit_attribute_spec()

    mapper = HinSAGENodeMapper(
        G, nodes_type_2, node_type="t2", batch_size=batch_size, num_samples=[2, 1]
    )

    schema = G.create_graph_schema()
    sampling_adj = schema.get_type_adjacency_list(["t2"], 2)

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
