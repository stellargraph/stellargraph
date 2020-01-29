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


from stellargraph import StellarGraph, StellarDiGraph
import networkx as nx
import pandas as pd
import numpy as np
import random
import pytest


def create_graph_features():
    # APPNP, ClusterGCN, GCN, PPNP, node_mappers
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c"])
    graph.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
    graph = graph.to_undirected()
    return graph, np.array([[1, 1], [1, 0], [0, 1]])


def relational_create_graph_features(is_directed=False):
    # RGCN, relational node mappers
    r1 = {"label": "r1"}
    r2 = {"label": "r2"}
    nodes = ["a", "b", "c"]
    features = np.array([[1, 1], [1, 0], [0, 1]])
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )

    graph = nx.MultiDiGraph() if is_directed else nx.MultiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from([("a", "b", r1), ("b", "c", r1), ("a", "c", r2)])

    SG = StellarDiGraph if is_directed else StellarGraph
    return SG(graph, node_features=node_features), features


def example_graph_1_nx(
    feature_size=None, label="default", feature_name="feature", is_directed=False
):
    # stellargraph
    graph = nx.DiGraph() if is_directed else nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    graph.add_nodes_from([1, 2, 3, 4], label=label)
    graph.add_edges_from(elist, label=label)

    # Add example features
    if feature_size is not None:
        for v in graph.nodes():
            graph.nodes[v][feature_name] = np.ones(feature_size)

    return graph


def _repeated_features(values_to_repeat, width):
    column = np.expand_dims(values_to_repeat, axis=1)
    return column.repeat(width, axis=1)


def example_graph_1(
    feature_size=None, label="default", feature_name="feature", is_directed=False
):
    # attr2vec, graphattention, graphsage, node mappers (2), link mappers, types, stellargraph, unsupervised sampler
    elist = pd.DataFrame([(1, 2), (2, 3), (1, 4), (3, 2)], columns=["source", "target"])
    if feature_size is not None:
        features = _repeated_features(np.ones(4), feature_size)
    else:
        features = []

    nodes = pd.DataFrame(features, index=[1, 2, 3, 4])

    cls = StellarDiGraph if is_directed else StellarGraph
    return cls(nodes={label: nodes}, edges={label: elist})


def example_graph_2(feature_size=None, label="default") -> StellarGraph:
    # unsupervised sampler, link mapper
    elist = pd.DataFrame([(1, 2), (2, 3), (1, 4), (4, 2)], columns=["source", "target"])
    nodes = [1, 2, 3, 4]
    if feature_size is not None:
        features = _repeated_features(nodes, feature_size)
    else:
        features = []

    nodes = pd.DataFrame(features, index=nodes)
    return StellarGraph(nodes={label: nodes}, edges={label: elist})


def example_hin_1_nx(feature_name=None, for_nodes=None, feature_sizes=None):
    # stellargraph
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3], label="A")
    graph.add_nodes_from([4, 5, 6], label="B")
    graph.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label="R")
    graph.add_edges_from([(4, 5)], label="F")

    if feature_name is not None:
        if for_nodes is None:
            for_nodes = list(graph.nodes())

        if feature_sizes is None:
            feature_sizes = dict()

        for v in for_nodes:
            fs = feature_sizes.get(graph.nodes[v]["label"], 10)
            graph.nodes[v][feature_name] = v * np.ones(fs)

    return graph


def example_hin_1(feature_sizes=None) -> StellarGraph:
    def features(label, ids):
        if feature_sizes is None:
            return []
        else:
            feature_size = feature_sizes.get(label, 10)
            return _repeated_features(ids, feature_size)

    a_ids = [0, 1, 2, 3]
    a = pd.DataFrame(features("A", a_ids), index=a_ids)

    b_ids = [4, 5, 6]
    b = pd.DataFrame(features("B", b_ids), index=b_ids)

    r = pd.DataFrame(
        [(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], columns=["source", "target"]
    )
    f = pd.DataFrame([(4, 5)], columns=["source", "target"], index=[6])

    return StellarGraph(nodes={"A": a, "B": b}, edges={"R": r, "F": f})


def create_test_graph_nx(is_directed=False):
    # unsupervised sampler
    graph = nx.DiGraph() if is_directed else nx.Graph()
    edges = [
        ("0", 1),
        ("0", 2),
        (1, 3),
        (1, 4),
        (3, 6),
        (4, 7),
        (4, 8),
        (2, 5),
        (5, 9),
        (5, 10),
        ("0", "0"),
        (1, 1),
        (3, 3),
        (6, 6),
        (4, 4),
        (7, 7),
        (8, 8),
        (2, 2),
        (5, 5),
        (9, 9),
        ("self loner", "self loner"),  # an isolated node with a self link
    ]

    graph.add_edges_from(edges)

    graph.add_node("loner")  # an isolated node without self link
    return graph


def create_test_graph(is_directed=False):
    # biased random walker, breadth first walker, directed breadth first walker, uniform random walker
    if is_directed:
        return StellarDiGraph(create_test_graph_nx(is_directed))
    else:
        return StellarGraph(create_test_graph_nx(is_directed))


def create_stellargraph():
    # cluster gcn, cluster gcn node mapper
    Gnx, features = create_graph_features()
    nodes = Gnx.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    graph = StellarGraph(Gnx, node_features=node_features)

    return graph


def example_graph_1_saliency_maps(feature_size=None):
    # saliency gcn, saliency gat
    graph = nx.Graph()
    elist = [(0, 1), (0, 2), (2, 3), (3, 4), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    graph.add_nodes_from([0, 1, 2, 3, 4], label="default")
    graph.add_edges_from(elist, label="default")

    # Add example features
    if feature_size is not None:
        for v in graph.nodes():
            graph.nodes[v]["feature"] = np.ones(feature_size)
        return StellarGraph(graph, node_features="feature")

    else:
        return StellarGraph(graph)


def example_graph_random(feature_size=4, n_edges=20, n_nodes=6, n_isolates=1):
    # core/utils, link mapper, node mapper graph 3
    graph = nx.Graph()
    n_noniso = n_nodes - n_isolates
    edges = [
        (random.randint(0, n_noniso - 1), random.randint(0, n_noniso - 1))
        for _ in range(n_edges)
    ]
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edges, label="default")

    # Add example features
    if feature_size is not None:
        for v in graph.nodes():
            graph.nodes[v]["feature"] = int(v) * np.ones(feature_size, dtype="int")
        return StellarGraph(graph, node_features="feature")

    else:
        return StellarGraph(graph)


def node_features(seed=0) -> pd.DataFrame:
    random = np.random.RandomState(seed)
    node_data_np = random.rand(10, 10)
    return pd.DataFrame(node_data_np)


@pytest.fixture
def petersen_graph() -> StellarGraph:
    nxg = nx.petersen_graph()
    return StellarGraph(nxg, node_features=node_features())


@pytest.fixture
def line_graph() -> StellarGraph:
    nxg = nx.MultiGraph()
    nxg.add_nodes_from(range(10))
    nxg.add_edges_from([(i, i + 1) for i in range(9)])
    return StellarGraph(nxg, node_features=node_features())
