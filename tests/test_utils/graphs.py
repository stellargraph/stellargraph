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
    # APPNP, ClusterGCN, GCN, PPNP, node_mappers, full_batch_generators
    features = np.array([[1, 1], [1, 0], [0, 1]])
    nodes = pd.DataFrame(features, index=["a", "b", "c"])
    edges = pd.DataFrame(
        [("a", "b"), ("b", "c"), ("a", "c")], columns=["source", "target"]
    )
    return StellarGraph(nodes, edges), features


def relational_create_graph_features(is_directed=False):
    # RGCN, relational node mappers
    r1 = {"label": "r1"}
    r2 = {"label": "r2"}
    features = np.array([[1, 1], [1, 0], [0, 1]])
    nodes = pd.DataFrame(features, index=["a", "b", "c"])
    edges = {
        "r1": pd.DataFrame([("a", "b"), ("b", "c")], columns=["source", "target"]),
        "r2": pd.DataFrame([("a", "c")], columns=["source", "target"], index=[2]),
    }
    SG = StellarDiGraph if is_directed else StellarGraph
    return SG(nodes, edges), features


def example_graph_nx(
    feature_size=None, label="default", feature_name="feature", is_directed=False
):
    graph = nx.DiGraph() if is_directed else nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (4, 2)]
    graph.add_nodes_from([1, 2, 3, 4], label=label)
    graph.add_edges_from(elist, label=label)

    # Add example features
    if feature_size is not None:
        for v in graph.nodes():
            graph.nodes[v][feature_name] = int(v) * np.ones(feature_size)

    return graph


def repeated_features(values_to_repeat, width):
    if width is None:
        return []

    column = np.expand_dims(values_to_repeat, axis=1)
    return column.repeat(width, axis=1)


def example_graph(
    feature_size=None,
    node_label="default",
    edge_label="default",
    feature_name="feature",
    is_directed=False,
):
    elist = pd.DataFrame([(1, 2), (2, 3), (1, 4), (4, 2)], columns=["source", "target"])
    nodes = [1, 2, 3, 4]
    features = repeated_features(nodes, feature_size)

    nodes = pd.DataFrame(features, index=nodes)

    cls = StellarDiGraph if is_directed else StellarGraph
    return cls(nodes={node_label: nodes}, edges={edge_label: elist})


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


def example_hin_1(
    feature_sizes=None, is_directed=False, self_loop=False
) -> StellarGraph:
    def features(label, ids):
        if feature_sizes is None:
            return []
        else:
            feature_size = feature_sizes.get(label, 10)
            return repeated_features(ids, feature_size)

    a_ids = [0, 1, 2, 3]
    a = pd.DataFrame(features("A", a_ids), index=a_ids)

    b_ids = [4, 5, 6]
    b = pd.DataFrame(features("B", b_ids), index=b_ids)

    r = pd.DataFrame(
        [(4, 0), (1, 5), (1, 4), (2, 4), (5, 3)], columns=["source", "target"]
    )
    f_edges, f_index = [(4, 5)], [6]
    if self_loop:
        # make it a multigraph
        f_edges.extend([(5, 5), (5, 5)])
        f_index.extend([7, 8])

    # add some weights for the f edges, but not others
    f_columns = ["source", "target", "weight"]
    for i, src_tgt in enumerate(f_edges):
        f_edges[i] = src_tgt + (10 + i,)

    f = pd.DataFrame(f_edges, columns=f_columns, index=f_index)

    cls = StellarDiGraph if is_directed else StellarGraph
    return cls(nodes={"A": a, "B": b}, edges={"R": r, "F": f})


def create_test_graph(is_directed=False):
    # biased random walker, breadth first walker, directed breadth first walker, uniform random walker

    nodes = pd.DataFrame(
        index=["0", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "self loner", "loner"]
    )
    edges = pd.DataFrame(
        [
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
        ],
        columns=["source", "target"],
    )
    cls = StellarDiGraph if is_directed else StellarGraph
    return cls(nodes, edges)


def example_graph_1_saliency_maps(feature_size=None):
    # saliency gcn, saliency gat
    nlist = [0, 1, 2, 3, 4]
    if feature_size is None:
        nodes = pd.DataFrame(index=nlist)
    else:
        # Example features
        nodes = pd.DataFrame(np.ones((len(nlist), feature_size)), index=nlist)

    elist = [(0, 1), (0, 2), (2, 3), (3, 4), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    edges = pd.DataFrame(elist, columns=["source", "target"])

    return StellarGraph(nodes, edges)


def example_graph_random(
    feature_size=4, n_edges=20, n_nodes=6, n_isolates=1, is_directed=False,
):
    # core/utils, link mapper, node mapper graph 3

    if is_directed:
        cls = StellarDiGraph
    else:
        cls = StellarGraph

    node_ids = range(n_nodes)
    if feature_size is None:
        nodes = pd.DataFrame(index=node_ids)
    else:
        nodes = pd.DataFrame(
            np.random.random((len(node_ids), feature_size)), index=node_ids
        )

    n_noniso = n_nodes - n_isolates
    elist = [
        (random.randint(0, n_noniso - 1), random.randint(0, n_noniso - 1))
        for _ in range(n_edges)
    ]
    edges = pd.DataFrame(elist, columns=["source", "target"])
    return cls(nodes, edges)


def node_features(seed=0) -> pd.DataFrame:
    random = np.random.RandomState(seed)
    node_data_np = random.rand(10, 10)
    return pd.DataFrame(node_data_np)


@pytest.fixture
def petersen_graph() -> StellarGraph:
    nxg = nx.petersen_graph()
    return StellarGraph.from_networkx(nxg, node_features=node_features())


@pytest.fixture
def line_graph() -> StellarGraph:
    nodes = node_features()
    edges = pd.DataFrame([(i, i + 1) for i in range(9)], columns=["source", "target"])
    return StellarGraph(nodes, edges)


@pytest.fixture
def knowledge_graph():
    nodes = ["a", "b", "c", "d"]

    edge_counter = 0

    def edge_df(*elements):
        nonlocal edge_counter
        end = edge_counter + len(elements)
        index = range(edge_counter, end)
        edge_counter = end
        return pd.DataFrame(elements, columns=["source", "target"], index=index)

    edges = {
        "W": edge_df(("a", "b")),
        "X": edge_df(("a", "b"), ("b", "c")),
        "Y": edge_df(("b", "a")),
        "Z": edge_df(("d", "b")),
    }

    return StellarDiGraph(nodes=pd.DataFrame(index=nodes), edges=edges)


@pytest.fixture
def tree_graph() -> StellarGraph:
    nodes = pd.DataFrame(index=["root", "0", 1, 2, "c1.1", "c2.1", "c2.2"])
    edges = pd.DataFrame(
        [
            ("root", 2),
            ("root", 1),
            ("root", "0"),
            (2, "c2.1"),
            (2, "c2.2"),
            (1, "c1.1"),
        ],
        columns=["source", "target"],
    )

    return StellarDiGraph(nodes, edges)


@pytest.fixture
def barbell():
    return StellarGraph.from_networkx(nx.barbell_graph(m1=10, m2=11))
