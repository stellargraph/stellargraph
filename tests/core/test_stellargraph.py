# -*- coding: utf-8 -*-
#
# Copyright 2017-2020 Data61, CSIRO
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

import pytest
import pandas as pd
import networkx as nx
import random
from stellargraph.core.graph import *
from stellargraph.data.converter import *
from ..test_utils.alloc import snapshot, allocation_benchmark


def create_graph_1(is_directed=False, return_nx=False):
    g = nx.DiGraph() if is_directed else nx.Graph()
    g.add_nodes_from([0, 1, 2, 3], label="movie")
    g.add_nodes_from([4, 5], label="user")
    g.add_edges_from([(4, 0), (4, 1), (5, 1), (4, 2), (5, 3)], label="rating")
    if return_nx:
        return nx.MultiDiGraph(g) if is_directed else nx.MultiGraph(g)
    return StellarDiGraph(g) if is_directed else StellarGraph(g)


def example_stellar_graph_1(feature_name=None, feature_size=10):
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_nodes_from([1, 2, 3, 4], label="default")
    G.add_edges_from(elist, label="default")

    # Add some numeric node attributes
    if feature_name:
        for v in G.nodes():
            G.nodes[v][feature_name] = v * np.ones(feature_size)

        return StellarGraph(G, node_features=feature_name)
    else:
        return StellarGraph(G)


def example_hin_1(feature_name=False, for_nodes=None, feature_sizes={}):
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3], label="A")
    G.add_nodes_from([4, 5, 6], label="B")
    G.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label="R")
    G.add_edges_from([(4, 5)], label="F")

    # Add some numeric node attributes
    if feature_name:
        if for_nodes is None:
            for_nodes = list(G.nodes())

        for v in for_nodes:
            fs = feature_sizes.get(G.nodes[v]["label"], 10)
            G.nodes[v][feature_name] = v * np.ones(fs)

        return StellarGraph(G, node_features=feature_name)
    else:
        return StellarGraph(G)


def example_stellar_graph_1_nx(feature_name=None):
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_nodes_from([1, 2, 3, 4], label="default")
    G.add_edges_from(elist, label="default")

    # Add some numeric node attributes
    if feature_name:
        for v in G.nodes():
            G.nodes[v][feature_name] = v * np.ones(10)

    return G


def example_hin_1_nx(feature_name=False, for_nodes=[]):
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3], label="A")
    G.add_nodes_from([4, 5, 6], label="B")
    G.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label="R")
    G.add_edges_from([(4, 5)], label="F")

    if feature_name:
        for v in for_nodes:
            G.nodes[v][feature_name] = v * np.ones(10)
    return G


def test_graph_constructor():
    sg = StellarGraph()
    assert sg.is_directed() == False
    assert sg._graph._node_type_attr == "label"
    assert sg._graph._edge_type_attr == "label"

    sg = StellarGraph(node_type_name="type", edge_type_name="type")
    assert sg.is_directed() == False
    assert sg._graph._node_type_attr == "type"
    assert sg._graph._edge_type_attr == "type"


def test_digraph_constructor():
    sg = StellarDiGraph()
    assert sg.is_directed() == True
    assert sg._graph._node_type_attr == "label"
    assert sg._graph._edge_type_attr == "label"

    sg = StellarDiGraph(node_type_name="type", edge_type_name="type")
    assert sg.is_directed() == True
    assert sg._graph._node_type_attr == "type"
    assert sg._graph._edge_type_attr == "type"


def test_info():
    sg = create_graph_1()
    info_str = sg.info()
    info_str = sg.info(show_attributes=False)
    # How can we check this?


def test_graph_from_nx():
    Gnx = nx.karate_club_graph()
    sg = StellarGraph(Gnx)

    nodes_1 = sorted(Gnx.nodes(data=False))
    nodes_2 = sorted(sg.nodes())
    assert nodes_1 == nodes_2

    edges_1 = sorted(Gnx.edges(data=False))
    edges_2 = sorted(sg.edges())
    assert edges_1 == edges_2


def test_homogeneous_graph_schema():
    Gnx = nx.karate_club_graph()
    for sg in [
        StellarGraph(Gnx),
        StellarGraph(Gnx, node_type_name="type", edge_type_name="type"),
    ]:
        schema = sg.create_graph_schema()

        assert "default" in schema.schema
        assert len(schema.node_types) == 1
        assert len(schema.edge_types) == 1


def test_graph_schema():
    g = create_graph_1(return_nx=True)
    sg = StellarGraph(g)
    schema = sg.create_graph_schema(create_type_maps=True)

    assert "movie" in schema.schema
    assert "user" in schema.schema
    assert len(schema.schema["movie"]) == 1
    assert len(schema.schema["user"]) == 1

    # Test node type lookup
    for n, ndata in g.nodes(data=True):
        assert ndata["label"] == schema.get_node_type(n)

    # Test edge type lookup
    node_labels = nx.get_node_attributes(g, "label")
    for n1, n2, k, edata in g.edges(keys=True, data=True):
        assert (node_labels[n1], edata["label"], node_labels[n2]) == tuple(
            schema.get_edge_type((n1, n2, k))
        )

    # Test undirected graph types
    assert schema.get_edge_type((4, 0, 0)) == ("user", "rating", "movie")
    assert schema.get_edge_type((0, 4, 0)) == ("movie", "rating", "user")


def test_graph_schema_sampled():
    sg = create_graph_1()

    # Will fail if create_type_maps=True and nodes/edges specified
    with pytest.raises(ValueError):
        sg.create_graph_schema(nodes=[0, 4])

    schema = sg.create_graph_schema(create_type_maps=False, nodes=[0, 4])

    assert "movie" in schema.schema
    assert "user" in schema.schema
    assert len(schema.schema["movie"]) == 1
    assert len(schema.schema["user"]) == 1

    # Node and edge type lookups will fail with no type maps
    with pytest.raises(RuntimeError):
        schema.get_node_type(0)

    with pytest.raises(RuntimeError):
        schema.get_edge_type((4, 0, 0))


def test_digraph_schema():
    g = create_graph_1(is_directed=True, return_nx=True)
    sg = StellarDiGraph(g)
    schema = sg.create_graph_schema()

    assert "movie" in schema.schema
    assert "user" in schema.schema
    assert len(schema.schema["user"]) == 1
    assert len(schema.schema["movie"]) == 0

    # Test node type lookup
    for n, ndata in g.nodes(data=True):
        assert ndata["label"] == schema.get_node_type(n)

    # Test edge type lookup
    node_labels = nx.get_node_attributes(g, "label")
    for n1, n2, k, edata in g.edges(keys=True, data=True):
        assert (node_labels[n1], edata["label"], node_labels[n2]) == tuple(
            schema.get_edge_type((n1, n2, k))
        )

    assert schema.get_edge_type((4, 0, 0)) == ("user", "rating", "movie")
    with pytest.raises(IndexError):
        schema.get_edge_type((0, 4, 0))


def test_get_index_for_nodes():
    sg = example_stellar_graph_1(feature_name="feature", feature_size=8)
    aa = sg.get_index_for_nodes([1, 2, 3, 4])
    assert aa == [0, 1, 2, 3]

    sg = example_hin_1(feature_name="feature")
    aa = sg.get_index_for_nodes([0, 1, 2, 3])
    assert aa == [0, 1, 2, 3]
    aa = sg.get_index_for_nodes([0, 1, 2, 3], "A")
    assert aa == [0, 1, 2, 3]
    aa = sg.get_index_for_nodes([4, 5, 6])
    assert aa == [0, 1, 2]
    aa = sg.get_index_for_nodes([4, 5, 6], "B")
    assert aa == [0, 1, 2]
    with pytest.raises(ValueError):
        aa = sg.get_index_for_nodes([1, 2, 5])


def test_feature_conversion_from_nodes():
    sg = example_stellar_graph_1(feature_name="feature", feature_size=8)
    aa = sg.node_features([1, 2, 3, 4])
    assert aa[:, 0] == pytest.approx([1, 2, 3, 4])

    assert aa.shape == (4, 8)
    assert sg.node_feature_sizes()["default"] == 8

    sg = example_hin_1(
        feature_name="feature",
        for_nodes=[0, 1, 2, 3, 4, 5],
        feature_sizes={"A": 4, "B": 2},
    )
    aa = sg.node_features([0, 1, 2, 3], "A")
    assert aa[:, 0] == pytest.approx([0, 1, 2, 3])
    assert aa.shape == (4, 4)

    fs = sg.node_feature_sizes()
    assert fs["A"] == 4
    assert fs["B"] == 2

    ab = sg.node_features([4, 5], "B")
    assert ab.shape == (2, 2)
    assert ab[:, 0] == pytest.approx([4, 5])

    # Test mixed types
    with pytest.raises(ValueError):
        ab = sg.node_features([1, 5])

    # Test incorrect manual node_type
    with pytest.raises(ValueError):
        ab = sg.node_features([4, 5], "A")

    # Test feature for node with no set attributes
    ab = sg.node_features([4, 5, 6], "B")
    assert ab.shape == (3, 2)
    assert ab[:, 0] == pytest.approx([4, 5, 0])


def test_null_node_feature():
    sg = example_stellar_graph_1(feature_name="feature", feature_size=6)
    aa = sg.node_features([1, None, 2, None])
    assert aa.shape == (4, 6)
    assert aa[:, 0] == pytest.approx([1, 0, 2, 0])

    sg = example_hin_1(feature_name="feature", feature_sizes={"A": 4, "B": 2})

    # Test feature for null node, without node type
    ab = sg.node_features([None, 5, None])
    assert ab.shape == (3, 2)
    assert ab[:, 0] == pytest.approx([0, 5, 0])

    # Test feature for null node, node type
    ab = sg.node_features([None, 6, None], "B")
    assert ab.shape == (3, 2)
    assert ab[:, 0] == pytest.approx([0, 6, 0])

    # Test feature for null node, wrong type
    with pytest.raises(ValueError):
        sg.node_features([None, 5, None], "A")

    # Test null-node with no type
    with pytest.raises(ValueError):
        sg.node_features([None, None])


def test_node_types():
    sg = example_stellar_graph_1(feature_name="feature", feature_size=6)
    assert sg.node_types == {"default"}

    sg = example_hin_1(feature_name="feature", feature_sizes={"A": 4, "B": 2})
    assert sg.node_types == {"A", "B"}

    sg = example_hin_1()
    assert sg.node_types == {"A", "B"}


def test_feature_conversion_from_dataframe():
    g = example_stellar_graph_1_nx()

    # Create features for nodes
    df = pd.DataFrame({v: np.ones(10) * float(v) for v in list(g)}).T
    gs = StellarGraph(g, node_features=df)

    aa = gs.node_features([1, 2, 3, 4])
    assert aa[:, 0] == pytest.approx([1, 2, 3, 4])

    # Check None identifier
    aa = gs.node_features([1, 2, None, None])
    assert aa[:, 0] == pytest.approx([1, 2, 0, 0])

    g = example_hin_1_nx()

    df = {
        t: pd.DataFrame(
            {
                v: np.ones(10) * float(v)
                for v, vdata in g.nodes(data=True)
                if vdata["label"] == t
            }
        ).T
        for t in ["A", "B"]
    }
    gs = StellarGraph(g, node_features=df)

    aa = gs.node_features([0, 1, 2, 3], "A")
    assert aa[:, 0] == pytest.approx([0, 1, 2, 3])
    assert aa.shape == (4, 10)

    ab = gs.node_features([4, 5], "B")
    assert ab.shape == (2, 10)
    assert ab[:, 0] == pytest.approx([4, 5])

    # Test mixed types
    with pytest.raises(ValueError):
        ab = gs.node_features([1, 5])

    # Test incorrect manual node_type
    with pytest.raises(ValueError):
        ab = gs.node_features([4, 5], "A")

    # Test feature for node with no set attributes
    ab = gs.node_features([4, None, None], "B")
    assert ab.shape == (3, 10)
    assert ab[:, 0] == pytest.approx([4, 0, 0])


def test_feature_conversion_from_iterator():
    g = example_stellar_graph_1_nx()

    # Create features for nodes
    node_features = [(v, np.ones(10) * float(v)) for v in list(g)]
    gs = StellarGraph(g, node_features=node_features)

    aa = gs.node_features([1, 2, 3, 4])
    assert aa[:, 0] == pytest.approx([1, 2, 3, 4])

    # Check None identifier
    aa = gs.node_features([1, 2, None, None])
    assert aa[:, 0] == pytest.approx([1, 2, 0, 0])

    g = example_hin_1_nx()
    nf = {
        t: [
            (v, np.ones(10) * float(v))
            for v, vdata in g.nodes(data=True)
            if vdata["label"] == t
        ]
        for t in ["A", "B"]
    }
    gs = StellarGraph(g, node_features=nf)

    aa = gs.node_features([0, 1, 2, 3], "A")
    assert aa[:, 0] == pytest.approx([0, 1, 2, 3])
    assert aa.shape == (4, 10)

    ab = gs.node_features([4, 5], "B")
    assert ab.shape == (2, 10)
    assert ab[:, 0] == pytest.approx([4, 5])

    # Test mixed types
    with pytest.raises(ValueError):
        ab = gs.node_features([1, 5])

    # Test incorrect manual node_type
    with pytest.raises(ValueError):
        ab = gs.node_features([4, 5], "A")

    # Test feature for node with no set attributes
    ab = gs.node_features([4, None, None], "B")
    assert ab.shape == (3, 10)
    assert ab[:, 0] == pytest.approx([4, 0, 0])

    # Test an iterator over all types
    g = example_hin_1_nx()
    nf = [
        (v, np.ones(5 if vdata["label"] == "A" else 10) * float(v))
        for v, vdata in g.nodes(data=True)
    ]
    gs = StellarGraph(g, node_features=nf)

    aa = gs.node_features([0, 1, 2, 3], "A")
    assert aa[:, 0] == pytest.approx([0, 1, 2, 3])
    assert aa.shape == (4, 5)

    ab = gs.node_features([4, 5], "B")
    assert ab.shape == (2, 10)
    assert ab[:, 0] == pytest.approx([4, 5])


def example_benchmark_graph(
    feature_size=None, n_nodes=100, n_edges=200, n_types=4, features_in_nodes=True
):
    G = nx.Graph()

    G.add_nodes_from(range(n_nodes))
    edges = [
        (random.randint(0, n_nodes - 1), random.randint(0, n_nodes - 1))
        for _ in range(n_edges)
    ]
    G.add_edges_from(edges)

    for v in G.nodes():
        G.nodes[v]["label"] = v % n_types

    # Add example features
    if feature_size is None:
        node_features = None
    elif features_in_nodes:
        node_features = "feature"
        for v in G.nodes():
            G.nodes[v][node_features] = np.ones(feature_size)
    else:
        node_features = {}
        for ty in range(n_types):
            type_nodes = range(ty, n_nodes, n_types)
            if len(type_nodes) > 0:
                node_features[ty] = pd.DataFrame(
                    [np.ones(feature_size)] * len(type_nodes), index=type_nodes
                )

    return G, node_features


@pytest.mark.benchmark(group="StellarGraph neighbours")
def test_benchmark_get_neighbours(benchmark):
    g, node_features = example_benchmark_graph()
    num_nodes = g.number_of_nodes()
    sg = StellarGraph(g, node_features=node_features)

    # get the neigbours of every node in the graph
    def f():
        for i in range(num_nodes):
            g.neighbors(i)

    benchmark(f)


@pytest.mark.benchmark(group="StellarGraph node features")
@pytest.mark.parametrize("num_types", [1, 4])
@pytest.mark.parametrize("type_arg", ["infer", "specify"])
def test_benchmark_get_features(benchmark, num_types, type_arg):
    SAMPLE_SIZE = 50
    N_NODES = 500
    N_EDGES = 1000
    g, node_features = example_benchmark_graph(
        feature_size=10, n_nodes=N_NODES, n_edges=N_EDGES, n_types=num_types
    )
    num_nodes = g.number_of_nodes()

    sg = StellarGraph(g, node_features=node_features)

    ty_ids = [(ty, range(ty, num_nodes, num_types)) for ty in range(num_types)]

    if type_arg == "specify":
        # pass through the type
        node_type = lambda ty: ty
    else:
        # leave the argument as None, and so use inference of the type
        node_type = lambda ty: None

    def f():
        # look up a random subset of the nodes for a random type, similar to what an algorithm that
        # does sampling might ask for
        ty, all_ids = random.choice(ty_ids)
        selected_ids = random.choices(all_ids, k=SAMPLE_SIZE)
        sg.node_features(selected_ids, node_type(ty))

    benchmark(f)


@pytest.mark.benchmark(group="StellarGraph creation", timer=snapshot)
# various element counts, to give an indication of the relationship
# between those and memory use (0,0 gives the overhead of the
# StellarGraph object itself, without any data)
@pytest.mark.parametrize("num_nodes,num_edges", [(0, 0), (100, 200), (1000, 5000)])
# features or not, to capture their cost
@pytest.mark.parametrize("feature_size", [None, 100])
def test_allocation_benchmark_creation_from_networkx(
    allocation_benchmark, feature_size, num_nodes, num_edges
):
    g, node_features = example_benchmark_graph(
        feature_size, num_nodes, num_edges, features_in_nodes=True
    )

    def f():
        return StellarGraph(g, node_features=node_features)

    allocation_benchmark(f)


def example_weighted_hin(is_directed=True):
    graph = nx.MultiDiGraph() if is_directed else nx.MultiGraph()
    graph.add_nodes_from([0, 1], label="A")
    graph.add_nodes_from([2, 3], label="B")
    graph.add_weighted_edges_from([(0, 1, 0.0), (0, 1, 1.0)], label="AA")
    graph.add_weighted_edges_from([(1, 2, 10.0), (1, 3, 10.0)], label="AB")
    return StellarDiGraph(graph) if is_directed else StellarGraph(graph)


def example_unweighted_hom(is_directed=True):
    graph = nx.MultiDiGraph() if is_directed else nx.MultiGraph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (0, 1), (1, 2), (1, 3)])
    return StellarDiGraph(graph) if is_directed else StellarGraph(graph)


def assert_items_equal(l1, l2):
    return sorted(l1) == sorted(l2)


@pytest.mark.parametrize("is_directed", [True, False])
def test_neighbors_weighted_hin(is_directed):
    graph = example_weighted_hin(is_directed=is_directed)
    assert_items_equal(graph.neighbors(1), [0, 0, 2, 3])
    assert_items_equal(
        graph.neighbors(1, include_edge_weight=True),
        [(0, 0.0), (0, 1.0), (2, 10.0), (3, 10.0)],
    )
    assert_items_equal(
        graph.neighbors(1, include_edge_weight=True, edge_types=["AB"]),
        [(2, 10.0), (3, 10.0)],
    )


@pytest.mark.parametrize("is_directed", [True, False])
def test_neighbors_unweighted_hom(is_directed):
    graph = example_unweighted_hom(is_directed=is_directed)
    assert_items_equal(graph.neighbors(1), [0, 0, 2, 3])
    assert_items_equal(
        graph.neighbors(1, include_edge_weight=True),
        [(0, None), (0, None), (2, None), (3, None)],
    )
    assert_items_equal(
        graph.neighbors(1, include_edge_weight=True, edge_types=["AB"]), []
    )


def test_undirected_hin_neighbor_methods():
    graph = example_weighted_hin(is_directed=True)
    assert_items_equal(graph.neighbors(1), graph.in_nodes(1))
    assert_items_equal(graph.neighbors(1), graph.out_nodes(1))


def test_in_nodes_weighted_hin():
    graph = example_weighted_hin()
    assert_items_equal(graph.in_nodes(1), [0, 0])
    assert_items_equal(
        graph.in_nodes(1, include_edge_weight=True), [(0, 0.0), (0, 1.0)]
    )
    assert_items_equal(
        graph.in_nodes(1, include_edge_weight=True, edge_types=["AB"]), []
    )


def test_in_nodes_unweighted_hom():
    graph = example_unweighted_hom()
    assert_items_equal(graph.in_nodes(1), [0, 0])
    assert_items_equal(
        graph.in_nodes(1, include_edge_weight=True), [(0, None), (0, None)]
    )
    assert_items_equal(
        graph.in_nodes(1, include_edge_weight=True, edge_types=["AA"]), []
    )


def test_out_nodes_weighted_hin():
    graph = example_weighted_hin()
    assert_items_equal(graph.out_nodes(1), [2, 3])
    assert_items_equal(
        graph.out_nodes(1, include_edge_weight=True), [(2, 10.0), (3, 10.0)]
    )
    assert_items_equal(
        graph.out_nodes(1, include_edge_weight=True, edge_types=["AA"]), []
    )


def test_out_nodes_unweighted_hom():
    graph = example_unweighted_hom()
    assert_items_equal(graph.out_nodes(1), [2, 3])
    assert_items_equal(
        graph.out_nodes(1, include_edge_weight=True), [(2, None), (3, None)]
    )
    assert_items_equal(
        graph.out_nodes(1, include_edge_weight=True, edge_types=["AB"]), []
    )
