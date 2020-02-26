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

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import random
from stellargraph.core.graph import *
from stellargraph.core.experimental import ExperimentalWarning
from ..test_utils.alloc import snapshot, allocation_benchmark
from ..test_utils.graphs import (
    example_graph_nx,
    example_graph,
    example_hin_1_nx,
    example_hin_1,
)

from .. import test_utils


pytestmark = test_utils.ignore_stellargraph_experimental_mark


# FIXME (#535): Consider using graph fixtures
def create_graph_1(is_directed=False, return_nx=False):
    g = nx.DiGraph() if is_directed else nx.Graph()
    g.add_nodes_from([0, 1, 2, 3], label="movie")
    g.add_nodes_from([4, 5], label="user")
    g.add_edges_from([(4, 0), (4, 1), (5, 1), (4, 2), (5, 3)], label="rating")
    if return_nx:
        return nx.MultiDiGraph(g) if is_directed else nx.MultiGraph(g)
    return StellarDiGraph(g) if is_directed else StellarGraph(g)


def example_benchmark_graph(
    feature_size=None, n_nodes=100, n_edges=200, n_types=4, features_in_nodes=True
):
    node_ids = np.arange(n_nodes)
    edges = pd.DataFrame(
        np.random.randint(0, n_nodes, size=(n_edges, 2)), columns=["source", "target"]
    )

    if feature_size is None:
        features = []
    else:
        features = np.ones((n_nodes, feature_size))

    all_nodes = pd.DataFrame(features, index=node_ids)
    nodes = {ty: all_nodes[node_ids % n_types == ty] for ty in range(n_types)}

    return nodes, edges


def example_benchmark_graph_nx(
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


def test_graph_constructor():
    graphs = [StellarGraph(), StellarGraph({}, {}), StellarGraph(nodes={}, edges={})]
    for sg in graphs:
        assert sg.is_directed() == False
        assert sg.number_of_nodes() == 0
        assert sg.number_of_edges() == 0


def test_graph_constructor_positional():
    # ok:
    StellarGraph({}, {}, is_directed=True)
    with pytest.raises(
        TypeError, match="takes from 1 to 3 positional arguments but 4 were given"
    ):
        # not ok:
        StellarGraph({}, {}, True)


def test_graph_constructor_legacy():
    with pytest.warns(DeprecationWarning, match="edge_weight_label"):
        StellarGraph(edge_weight_label="x")

    # can't pass edges when using the legacy NetworkX form
    with pytest.raises(
        ValueError, match="edges: expected no value when using legacy NetworkX"
    ):
        StellarGraph(nx.Graph(), {})

    # can't pass graph when using one of the other arguments
    with pytest.raises(
        ValueError,
        match="graph: expected no value when using 'nodes' and 'edges' parameters",
    ):
        StellarGraph({}, graph=nx.Graph())


def test_digraph_constructor():
    graphs = [
        StellarDiGraph(),
        StellarDiGraph({}, {}),
        StellarDiGraph(nodes={}, edges={}),
    ]
    for sg in graphs:
        assert sg.is_directed() == True
        assert sg.number_of_nodes() == 0
        assert sg.number_of_edges() == 0


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
    schema = sg.create_graph_schema()

    assert "movie" in schema.schema
    assert "user" in schema.schema
    assert len(schema.schema["movie"]) == 1
    assert len(schema.schema["user"]) == 1


def test_graph_schema_sampled():
    sg = create_graph_1()

    schema = sg.create_graph_schema(nodes=[0, 4])

    assert "movie" in schema.schema
    assert "user" in schema.schema
    assert len(schema.schema["movie"]) == 1
    assert len(schema.schema["user"]) == 1


def test_digraph_schema():
    g = create_graph_1(is_directed=True, return_nx=True)
    sg = StellarDiGraph(g)
    schema = sg.create_graph_schema()

    assert "movie" in schema.schema
    assert "user" in schema.schema
    assert len(schema.schema["user"]) == 1
    assert len(schema.schema["movie"]) == 0


def test_graph_schema_no_edges():
    nodes = pd.DataFrame(index=[0])
    g = StellarGraph(nodes=nodes, edges={})
    schema = g.create_graph_schema()
    assert len(schema.node_types) == 1
    assert len(schema.edge_types) == 0


@pytest.mark.benchmark(group="StellarGraph create_graph_schema")
@pytest.mark.parametrize("num_types", [1, 4])
def test_benchmark_graph_schema(benchmark, num_types):
    nodes, edges = example_benchmark_graph(
        n_nodes=1000, n_edges=5000, n_types=num_types
    )
    sg = StellarGraph(nodes=nodes, edges=edges)

    benchmark(sg.create_graph_schema)


def test_get_index_for_nodes():
    sg = example_graph(feature_size=8)
    aa = sg._get_index_for_nodes([1, 2, 3, 4])
    assert list(aa) == [0, 1, 2, 3]

    sg = example_hin_1(feature_sizes={})
    aa = sg._get_index_for_nodes([0, 1, 2, 3])
    assert list(aa) == [0, 1, 2, 3]
    aa = sg._get_index_for_nodes([0, 1, 2, 3], "A")
    assert list(aa) == [0, 1, 2, 3]
    aa = sg._get_index_for_nodes([4, 5, 6])
    assert list(aa) == [4, 5, 6]
    aa = sg._get_index_for_nodes([4, 5, 6], "B")
    assert list(aa) == [4, 5, 6]
    aa = sg._get_index_for_nodes([1, 2, 5])
    assert list(aa) == [1, 2, 5]


def test_feature_conversion_from_nodes():
    sg = example_graph(feature_size=8)
    aa = sg.node_features([1, 2, 3, 4])
    assert aa[:, 0] == pytest.approx([1, 2, 3, 4])

    assert aa.shape == (4, 8)
    assert sg.node_feature_sizes()["default"] == 8


def test_node_features_missing_id():
    sg = example_graph(feature_size=6)
    with pytest.raises(KeyError, match=r"\[1000, 2000\]"):
        sg.node_features([1, 1000, None, 2000])


def test_null_node_feature():
    sg = example_graph(feature_size=6)
    aa = sg.node_features([1, None, 2, None])
    assert aa.shape == (4, 6)
    assert aa[:, 0] == pytest.approx([1, 0, 2, 0])

    sg = example_hin_1(feature_sizes={"A": 4, "B": 2})

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
    sg = example_graph(feature_size=6)
    assert sg.node_types == {"default"}

    sg = example_hin_1(feature_sizes={"A": 4, "B": 2})
    assert sg.node_types == {"A", "B"}

    sg = example_hin_1()
    assert sg.node_types == {"A", "B"}


def test_feature_conversion_from_dataframe():
    g = example_graph_nx()

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
    g = example_graph_nx()

    # Create features for nodes
    node_features = [(v, np.ones(10) * float(v)) for v in list(g)]
    gs = StellarGraph(g, node_features=node_features)

    aa = gs.node_features([1, 2, 3, 4])
    assert aa[:, 0] == pytest.approx([1, 2, 3, 4])

    # Check None identifier
    aa = gs.node_features([1, 2, None, None])
    assert aa[:, 0] == pytest.approx([1, 2, 0, 0])

    # Test adjacency matrix
    adj_expected = np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 0]])

    A = gs.to_adjacency_matrix()
    assert A.dtype == "float32"
    assert np.allclose(A.toarray(), adj_expected)

    # Test adjacency matrix with node arguement
    A = gs.to_adjacency_matrix(nodes=[3, 2])
    assert A.dtype == "float32"
    assert np.allclose(A.toarray(), adj_expected[[2, 1]][:, [2, 1]])

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


def test_edges_include_edge_type():
    g = example_hin_1()

    r = {(src, dst, "R") for src, dst in [(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)]}
    f = {(4, 5, "F")}
    expected = normalize_edges(r | f, directed=False)
    assert normalize_edges(g.edges(include_edge_type=True), directed=False) == expected


def numpy_to_list(x):
    if isinstance(x, np.ndarray):
        return list(x)
    if isinstance(x, dict):
        return {numpy_to_list(k): numpy_to_list(v) for k, v in x.items()}
    if isinstance(x, list):
        return [numpy_to_list(v) for v in x]
    if isinstance(x, tuple):
        return tuple(numpy_to_list(v) for v in x)
    return x


def normalize_edges(edges, directed):
    if directed:
        return {(src, tgt): data for src, tgt, *data in edges}
    return {(min(src, tgt), max(src, tgt)): data for src, tgt, *data in edges}


def assert_networkx(g_nx, expected_nodes, expected_edges, *, directed):
    assert numpy_to_list(dict(g_nx.nodes(data=True))) == expected_nodes

    computed_edges = numpy_to_list(normalize_edges(g_nx.edges(data=True), directed))
    assert computed_edges == normalize_edges(expected_edges, directed)


@pytest.mark.parametrize("has_features", [False, True])
@pytest.mark.parametrize("include_features", [False, True])
def test_to_networkx(has_features, include_features):
    if has_features:
        a_size = 4
        b_size = 5
        feature_sizes = {"A": a_size, "B": b_size}
    else:
        a_size = b_size = 0
        feature_sizes = None

    if include_features:
        feature_name = "feature"
    else:
        feature_name = None

    g = example_hin_1(feature_sizes)
    g_nx = g.to_networkx(feature_name=feature_name)

    node_def = {"A": (a_size, [0, 1, 2, 3]), "B": (b_size, [4, 5, 6])}

    def node_attrs(label, x, size):
        d = {"label": label}
        if feature_name:
            d[feature_name] = [x] * size
        return d

    expected_nodes = {
        x: node_attrs(label, x, size)
        for label, (size, ids) in node_def.items()
        for x in ids
    }

    edge_def = {"R": [(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], "F": [(4, 5)]}
    expected_edges = [
        (src, tgt, {"label": label, "weight": 1.0 if label == "R" else 10.0})
        for label, pairs in edge_def.items()
        for src, tgt in pairs
    ]

    assert_networkx(g_nx, expected_nodes, expected_edges, directed=False)


def test_to_networkx_edge_attributes():
    nodes = pd.DataFrame([], index=[1, 10, 100])
    edges = pd.DataFrame(
        [(1, 10, 11), (10, 100, 110)], columns=["source", "target", "weight"]
    )
    g = StellarGraph(nodes=nodes, edges={"foo": edges})
    g_nx = g.to_networkx()

    expected_nodes = {k: {"label": "default", "feature": []} for k in [1, 10, 100]}
    expected_edges = [
        (src, dst, {"label": "foo", "weight": src + dst})
        for src, dst in [(1, 10), (10, 100)]
    ]

    assert_networkx(g_nx, expected_nodes, expected_edges, directed=False)


def test_networkx_attribute_message():
    ug = StellarGraph()
    dg = StellarDiGraph()

    with pytest.raises(
        AttributeError, match="The 'StellarGraph' type no longer inherits"
    ):
        # this graph is undirected and the corresponding networkx type doesn't have this
        # attribute, but there's no reason to be too precise
        ug.successors

    with pytest.raises(
        AttributeError, match="The 'StellarDiGraph' type no longer inherits"
    ):
        dg.successors

    # make sure that the user doesn't get spammed with junk about networkx when they're just making
    # a normal typo with the new StellarGraph
    with pytest.raises(AttributeError, match="has no attribute 'not_networkx_attr'$"):
        ug.not_networkx_attr

    with pytest.raises(AttributeError, match="has no attribute 'not_networkx_attr'$"):
        dg.not_networkx_attr

    # getting an existing attribute via `getattr` should work fine
    assert getattr(ug, "is_directed")() == False
    assert getattr(dg, "is_directed")() == True

    # calling __getattr__ directly is... unconventional, but it should work
    assert ug.__getattr__("is_directed")() == False
    assert dg.__getattr__("is_directed")() == True


@pytest.mark.benchmark(group="StellarGraph neighbours")
def test_benchmark_get_neighbours(benchmark):
    nodes, edges = example_benchmark_graph()
    sg = StellarGraph(nodes=nodes, edges=edges)
    num_nodes = sg.number_of_nodes()

    # get the neigbours of every node in the graph
    def f():
        for i in range(num_nodes):
            sg.neighbors(i)

    benchmark(f)


@pytest.mark.benchmark(group="StellarGraph node features")
@pytest.mark.parametrize("num_types", [1, 4])
@pytest.mark.parametrize("type_arg", ["infer", "specify"])
def test_benchmark_get_features(benchmark, num_types, type_arg):
    SAMPLE_SIZE = 50
    N_NODES = 500
    N_EDGES = 1000
    nodes, edges = example_benchmark_graph(
        feature_size=10, n_nodes=N_NODES, n_edges=N_EDGES, n_types=num_types
    )

    sg = StellarGraph(nodes=nodes, edges=edges)
    num_nodes = sg.number_of_nodes()

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


@pytest.mark.benchmark(group="StellarGraph creation (time)")
# various element counts, to give an indication of the relationship
# between those and memory use (0,0 gives the overhead of the
# StellarGraph object itself, without any data)
@pytest.mark.parametrize("num_nodes,num_edges", [(0, 0), (100, 200), (1000, 5000)])
# features or not, to capture their cost
@pytest.mark.parametrize("feature_size", [None, 100])
def test_benchmark_creation(benchmark, feature_size, num_nodes, num_edges):
    nodes, edges = example_benchmark_graph(
        feature_size, num_nodes, num_edges, features_in_nodes=True
    )

    def f():
        return StellarGraph(nodes=nodes, edges=edges)

    benchmark(f)


@pytest.mark.benchmark(group="StellarGraph creation", timer=snapshot)
# various element counts, to give an indication of the relationship
# between those and memory use (0,0 gives the overhead of the
# StellarGraph object itself, without any data)
@pytest.mark.parametrize("num_nodes,num_edges", [(0, 0), (100, 200), (1000, 5000)])
# features or not, to capture their cost
@pytest.mark.parametrize("feature_size", [None, 100])
def test_allocation_benchmark_creation(
    allocation_benchmark, feature_size, num_nodes, num_edges
):
    nodes, edges = example_benchmark_graph(
        feature_size, num_nodes, num_edges, features_in_nodes=True
    )

    def f():
        return StellarGraph(nodes=nodes, edges=edges)

    allocation_benchmark(f)


def example_weighted_hin(is_directed=True):
    edge_cols = ["source", "target", "weight"]
    cls = StellarDiGraph if is_directed else StellarGraph
    return cls(
        nodes={"A": pd.DataFrame(index=[0, 1]), "B": pd.DataFrame(index=[2, 3])},
        edges={
            "AA": pd.DataFrame(
                [(0, 1, 0.0), (0, 1, 1.0)], columns=edge_cols, index=[0, 1]
            ),
            "AB": pd.DataFrame(
                [(1, 2, 10.0), (1, 3, 10.0)], columns=edge_cols, index=[2, 3]
            ),
        },
    )


def example_unweighted_hom(is_directed=True):
    graph = nx.MultiDiGraph() if is_directed else nx.MultiGraph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (0, 1), (1, 2), (1, 3)])
    return StellarDiGraph(graph) if is_directed else StellarGraph(graph)


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


def assert_items_equal(l1, l2):
    assert sorted(l1) == sorted(l2)


@pytest.mark.parametrize("is_directed", [True, False])
def test_neighbors_unweighted_hom(is_directed):
    graph = example_unweighted_hom(is_directed=is_directed)
    assert_items_equal(graph.neighbors(1), [0, 0, 2, 3])
    assert_items_equal(
        graph.neighbors(1, include_edge_weight=True), [(0, 1), (0, 1), (2, 1), (3, 1)],
    )
    assert_items_equal(
        graph.neighbors(1, include_edge_weight=True, edge_types=["AB"]), []
    )


def test_undirected_hin_neighbor_methods():
    graph = example_weighted_hin(is_directed=False)
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
    assert_items_equal(graph.in_nodes(1, include_edge_weight=True), [(0, 1), (0, 1)])
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
    assert_items_equal(graph.out_nodes(1, include_edge_weight=True), [(2, 1), (3, 1)])
    assert_items_equal(
        graph.out_nodes(1, include_edge_weight=True, edge_types=["AB"]), []
    )


@pytest.mark.parametrize("is_directed", [False, True])
def test_isolated_node_neighbor_methods(is_directed):
    cls = StellarDiGraph if is_directed else StellarGraph
    graph = cls(
        nodes=pd.DataFrame(index=[1]), edges=pd.DataFrame(columns=["source", "target"])
    )
    assert graph.neighbors(1) == []
    assert graph.in_nodes(1) == []
    assert graph.out_nodes(1) == []


@pytest.mark.parametrize("is_directed", [False, True])
def test_info_homogeneous(is_directed):

    g = example_graph(node_label="ABC", edge_label="xyz", is_directed=is_directed)
    info = g.info()
    if is_directed:
        assert "StellarDiGraph: Directed multigraph" in info
    else:
        assert "StellarGraph: Undirected multigraph" in info
    assert "Nodes: 4, Edges: 4" in info

    assert " ABC: [4]" in info
    assert " Edge types: ABC-xyz->ABC" in info

    assert " ABC-xyz->ABC: [4]" in info


def test_info_heterogeneous():
    g = example_hin_1()
    info = g.info()
    assert "StellarGraph: Undirected multigraph" in info
    assert "Nodes: 7, Edges: 6" in info

    assert " A: [4]" in info
    assert " Edge types: A-R->B" in info
    assert " B: [3]" in info
    assert " Edge types: B-F->B, B-R->A" in info

    assert " A-R->B: [5]"
    assert " B-F->B: [1]"


def test_edges_include_weights():
    g = example_weighted_hin()
    edges, weights = g.edges(include_edge_weight=True)
    nxg = g.to_networkx()
    assert len(edges) == len(weights) == len(nxg.edges())

    grouped = (
        pd.DataFrame(edges, columns=["source", "target"])
        .assign(weight=weights)
        .groupby(["source", "target"])
        .agg(list)
    )
    for (src, tgt), row in grouped.iterrows():
        assert sorted(row["weight"]) == sorted(
            [data["weight"] for data in nxg.get_edge_data(src, tgt).values()]
        )


def test_adjacency_types_undirected():
    g = example_hin_1(is_directed=False)
    adj = g._adjacency_types(g.create_graph_schema())

    assert adj == {
        ("A", "R", "B"): {0: [4], 1: [4, 5], 2: [4], 3: [5]},
        ("B", "R", "A"): {4: [0, 1, 2], 5: [1, 3]},
        ("B", "F", "B"): {4: [5], 5: [4]},
    }


def test_adjacency_types_directed():
    g = example_hin_1(is_directed=True)
    adj = g._adjacency_types(g.create_graph_schema())

    assert adj == {
        ("A", "R", "B"): {1: [4, 5], 2: [4]},
        ("B", "R", "A"): {4: [0], 5: [3]},
        ("B", "F", "B"): {4: [5]},
    }


def test_to_adjacency_matrix_weighted_undirected():
    g = example_hin_1(is_directed=False, self_loop=True)

    matrix = g.to_adjacency_matrix(weighted=True).todense()
    actual = np.zeros((7, 7), dtype=matrix.dtype)
    actual[0, 4] = actual[4, 0] = 1
    actual[1, 5] = actual[5, 1] = 1
    actual[1, 4] = actual[4, 1] = 1
    actual[2, 4] = actual[4, 2] = 1
    actual[3, 5] = actual[5, 3] = 1
    actual[4, 5] = actual[5, 4] = 10
    actual[5, 5] = 11 + 12
    assert np.array_equal(matrix, actual)

    # just to confirm, it should be symmetric
    assert np.array_equal(matrix, matrix.T)

    # use a funny order to verify order
    subgraph = g.to_adjacency_matrix([1, 6, 5], weighted=True).todense()
    # indices are relative to the specified list
    one, six, five = 0, 1, 2
    actual = np.zeros((3, 3), dtype=subgraph.dtype)
    actual[one, five] = actual[five, one] = 1
    actual[five, five] = 11 + 12
    assert np.array_equal(subgraph, actual)


def test_to_adjacency_matrix_weighted_directed():
    g = example_hin_1(is_directed=True, self_loop=True)

    matrix = g.to_adjacency_matrix(weighted=True).todense()
    actual = np.zeros((7, 7))
    actual[4, 0] = 1
    actual[1, 5] = 1
    actual[1, 4] = 1
    actual[2, 4] = 1
    actual[5, 3] = 1
    actual[4, 5] = 10
    actual[5, 5] = 11 + 12

    assert np.array_equal(matrix, actual)

    # use a funny order to verify order
    subgraph = g.to_adjacency_matrix([1, 6, 5], weighted=True).todense()
    # indices are relative to the specified list
    one, six, five = 0, 1, 2
    actual = np.zeros((3, 3), dtype=subgraph.dtype)
    actual[one, five] = 1
    actual[five, five] = 11 + 12
    assert np.array_equal(subgraph, actual)


def test_to_adjacency_matrix():
    g = example_hin_1(is_directed=False, self_loop=True)

    matrix = g.to_adjacency_matrix().todense()
    actual = np.zeros((7, 7), dtype=matrix.dtype)
    actual[0, 4] = actual[4, 0] = 1
    actual[1, 5] = actual[5, 1] = 1
    actual[1, 4] = actual[4, 1] = 1
    actual[2, 4] = actual[4, 2] = 1
    actual[3, 5] = actual[5, 3] = 1
    actual[4, 5] = actual[5, 4] = 1
    actual[5, 5] = 2
    assert np.array_equal(matrix, actual)


def test_edge_weights_undirected():
    g = example_hin_1(is_directed=False, self_loop=True)

    assert g._edge_weights(5, 5) == [11.0, 12.0]
    assert g._edge_weights(4, 5) == [10.0]
    assert g._edge_weights(5, 4) == [10.0]
    assert g._edge_weights(0, 4) == [1]
    assert g._edge_weights(4, 0) == [1]


def test_edge_weights_directed():
    g = example_hin_1(is_directed=True, self_loop=True)

    assert g._edge_weights(5, 5) == [11.0, 12.0]
    assert g._edge_weights(4, 5) == [10.0]
    assert g._edge_weights(5, 4) == []
    assert g._edge_weights(0, 4) == []
    assert g._edge_weights(4, 0) == [1]


def test_node_type():
    g = example_hin_1()
    assert g.node_type(0) == "A"
    assert g.node_type(4) == "B"

    with pytest.raises(KeyError, match="1234"):
        g.node_type(1234)


def test_from_networkx_empty():
    empty = StellarGraph.from_networkx(nx.Graph())
    assert not empty.is_directed()
    assert isinstance(empty, StellarGraph)

    empty = StellarGraph.from_networkx(nx.DiGraph())
    assert empty.is_directed()
    assert isinstance(empty, StellarDiGraph)


def test_from_networkx_smoke():
    g = nx.MultiGraph()
    g.add_node(1, node_label="a", features=[1])
    g.add_node(2)
    g.add_node(3, features=[2, 3, 4, 5])
    g.add_edge(1, 2, weight_attr=123)
    g.add_edge(2, 2, edge_label="X", weight_attr=456)
    g.add_edge(1, 2, edge_label="Y")
    g.add_edge(1, 1)

    with pytest.warns(
        UserWarning,
        match=r"found the following nodes \(of type 'b'\) without features, using 4-dimensional zero vector: 2",
    ):
        from_nx = StellarGraph.from_networkx(
            g,
            edge_weight_attr="weight_attr",
            node_type_attr="node_label",
            edge_type_attr="edge_label",
            node_type_default="b",
            edge_type_default="X",
            node_features="features",
        )

    raw = StellarGraph(
        nodes={
            "a": pd.DataFrame([1], index=[1]),
            "b": pd.DataFrame([(0, 0, 0, 0), (2, 3, 4, 5)], index=[2, 3]),
        },
        edges={
            "X": pd.DataFrame(
                [(1, 2, 123.0), (2, 2, 456.0), (1, 1, 1.0)],
                columns=["source", "target", "weight"],
            ),
            "Y": pd.DataFrame(
                [(1, 2, 1.0)], columns=["source", "target", "weight"], index=[3]
            ),
        },
    )

    def both(f, numpy=False):
        if numpy:
            assert np.array_equal(f(from_nx), f(raw))
        else:
            assert f(from_nx) == f(raw)

    both(lambda g: sorted(g.nodes()))
    nodes = raw.nodes()

    for n in nodes:
        both(lambda g: g.node_type(n))
        both(lambda g: g.node_features([n]), numpy=True)

    both(
        lambda g: dict(zip(*g.edges(include_edge_type=True, include_edge_weight=True)))
    )


def test_from_networkx_deprecations():
    with pytest.warns(None) as record:
        StellarGraph.from_networkx(
            nx.Graph(), edge_weight_label="w", node_type_name="n", edge_type_name="e",
        )

    assert "node_type_name" in str(record.pop(DeprecationWarning).message)
    assert "edge_type_name" in str(record.pop(DeprecationWarning).message)
    assert "edge_weight_label" in str(record.pop(DeprecationWarning).message)


@pytest.mark.parametrize("is_directed", [False, True])
@pytest.mark.parametrize(
    "nodes",
    [
        # no nodes = empty subgraph
        [],
        # no edges
        [0],
        # self loop
        [5],
        # various nodes with various edges, in a few different common types that might be used
        [0, 1, 4, 5],
        np.array([0, 1, 4, 5]),
        pd.Index([0, 1, 4, 5]),
    ],
)
def test_subgraph(is_directed, nodes):
    g = example_hin_1(feature_sizes={}, is_directed=is_directed, self_loop=True)
    sub = g.subgraph(nodes)

    # assume NetworkX's subgraph algorithm works
    expected = StellarGraph.from_networkx(g.to_networkx().subgraph(nodes))

    assert sub.is_directed() == is_directed

    assert set(sub.nodes()) == set(expected.nodes())

    sub_edges, sub_weights = sub.edges(include_edge_type=True, include_edge_weight=True)
    exp_edges, exp_weights = expected.edges(
        include_edge_type=True, include_edge_weight=True
    )
    assert normalize_edges(sub_edges, is_directed) == normalize_edges(
        exp_edges, is_directed
    )
    np.testing.assert_array_equal(sub_weights, exp_weights)

    for node in nodes:
        assert sub.node_type(node) == g.node_type(node)
        np.testing.assert_array_equal(
            sub.node_features([node]), g.node_features([node])
        )


def test_subgraph_missing_node():
    g = example_hin_1()
    with pytest.raises(KeyError, match="12345"):
        sub = g.subgraph([0, 1, 12345])


@pytest.mark.parametrize("is_directed", [False, True])
def test_connected_components(is_directed):
    nodes = pd.DataFrame(index=range(6))
    edges = pd.DataFrame([(0, 2), (2, 5), (1, 4)], columns=["source", "target"])

    if is_directed:
        g = StellarDiGraph(nodes, edges)
    else:
        g = StellarGraph(nodes, edges)

    a, b, c = g.connected_components()

    # (weak) connected components are the same for both directed and undirected graphs
    assert set(a) == {0, 2, 5}
    assert set(b) == {1, 4}
    assert set(c) == {3}

    # check that `connected_components` works with `subgraph`
    assert set(g.subgraph(a).edges()) == {(0, 2), (2, 5)}
