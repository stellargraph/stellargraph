# -*- coding: utf-8 -*-
#
# Copyright 2017-2018 Data61, CSIRO
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
import networkx as nx
from stellar.data.stellargraph import *


def create_graph_1(sg=StellarGraph()):
    sg.add_nodes_from([0, 1, 2, 3], label="movie")
    sg.add_nodes_from([4, 5], label="user")
    sg.add_edges_from([(4, 0), (4, 1), (5, 1), (4, 2), (5, 3)], label="rating")
    return sg


def create_graph_2(sg=StellarGraph()):
    sg.add_nodes_from([0, 1, 2, 3], label="movie")
    sg.add_nodes_from([4, 5], label="user")
    sg.add_edges_from([(4, 0), (4, 1), (5, 1), (4, 2), (5, 3)], label="rating")
    sg.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label="another")
    sg.add_edges_from([(4, 5)], label="friend")
    return sg


def test_graph_constructor():
    sg = StellarGraph()
    assert sg.is_directed() == False
    assert sg._node_type_attr == "label"
    assert sg._edge_type_attr == "label"

    sg = StellarGraph(node_type_name="type", edge_type_name="type")
    assert sg.is_directed() == False
    assert sg._node_type_attr == "type"
    assert sg._edge_type_attr == "type"


def test_digraph_constructor():
    sg = StellarDiGraph()
    assert sg.is_directed() == True
    assert sg._node_type_attr == "label"
    assert sg._edge_type_attr == "label"

    sg = StellarDiGraph(node_type_name="type", edge_type_name="type")
    assert sg.is_directed() == True
    assert sg._node_type_attr == "type"
    assert sg._edge_type_attr == "type"


def test_info():
    sg = create_graph_2()
    info_str = sg.info()
    info_str = sg.info(show_attributes=False)
    # How can we check this?


def test_graph_from_nx():
    Gnx = nx.karate_club_graph()
    sg = StellarGraph(Gnx)

    nodes_1 = sorted(Gnx.nodes(data=False))
    nodes_2 = sorted(sg.nodes(data=False))
    assert nodes_1 == nodes_2

    edges_1 = sorted(Gnx.edges(data=False))
    edges_2 = sorted(sg.edges(keys=False, data=False))
    assert edges_1 == edges_2


def test_homogeneous_graph_schema():
    Gnx = nx.karate_club_graph()
    for sg in [
        StellarGraph(Gnx),
        StellarGraph(Gnx, node_type_name="type", edge_type_name="type"),
    ]:
        schema = sg.create_graph_schema()

        assert "" in schema.schema
        assert len(schema.node_types) == 1
        assert len(schema.edge_types) == 1


def test_graph_schema():
    sg = create_graph_1()
    schema = sg.create_graph_schema(create_type_maps=True)

    assert "movie" in schema.schema
    assert "user" in schema.schema
    assert len(schema.schema["movie"]) == 1
    assert len(schema.schema["user"]) == 1

    # Test node type lookup
    for n, ndata in sg.nodes(data=True):
        assert ndata["label"] == schema.get_node_type(n)

    # Test edge type lookup
    node_labels = nx.get_node_attributes(sg, "label")
    for n1, n2, k, edata in sg.edges(keys=True, data=True):
        assert (node_labels[n1], edata["label"], node_labels[n2]) == tuple(
            schema.get_edge_type((n1, n2, k))
        )

    # Test edge_types_for_node
    ets = schema.edge_types_for_node_type("user")
    assert len(ets) == 1

    ets = schema.edge_types_for_node_type("movie")
    assert len(ets) == 1

    # Test undirected graph types
    assert schema.get_edge_type((4, 0, 0)) == ("user", "rating", "movie")
    assert schema.get_edge_type((0, 4, 0)) == ("movie", "rating", "user")


def test_digraph_schema():
    sg = create_graph_1(StellarDiGraph())
    schema = sg.create_graph_schema()

    assert "movie" in schema.schema
    assert "user" in schema.schema
    assert len(schema.schema["user"]) == 1
    assert len(schema.schema["movie"]) == 0

    # Test node type lookup
    for n, ndata in sg.nodes(data=True):
        assert ndata["label"] == schema.get_node_type(n)

    # Test edge type lookup
    node_labels = nx.get_node_attributes(sg, "label")
    for n1, n2, k, edata in sg.edges(keys=True, data=True):
        assert (node_labels[n1], edata["label"], node_labels[n2]) == tuple(
            schema.get_edge_type((n1, n2, k))
        )

    ets = schema.edge_types_for_node_type("user")
    assert len(ets) == 1

    ets = schema.edge_types_for_node_type("movie")
    assert len(ets) == 0

    assert schema.get_edge_type((4, 0, 0)) == ("user", "rating", "movie")
    assert schema.get_edge_type((0, 4, 0)) == None


def test_graph_schema_node_types():
    for sg in [
        create_graph_1(),
        create_graph_2(),
        create_graph_1(StellarDiGraph()),
        create_graph_2(StellarDiGraph()),
    ]:
        schema = sg.create_graph_schema(create_type_maps=True)

        for n, ndata in sg.nodes(data=True):
            assert schema.get_node_type(n) == ndata["label"]


def test_graph_schema_edge_types():
    for sg in [create_graph_1(), create_graph_2()]:
        schema = sg.create_graph_schema(create_type_maps=True)

        for n1, n2, k in sg.edges(keys=True):
            et = (
                sg.node[n1]["label"],
                sg.adj[n1][n2][k]["label"],
                sg.node[n2]["label"],
            )
            assert schema.get_edge_type((n1, n2, k)) == et

            # Check the is_of_edge_type function: it should work either way round for undirected
            # graphs
            assert schema.is_of_edge_type((n1, n2, k), et)
            assert schema.is_of_edge_type((n1, n2, k), (et[2], et[1], et[0]))

    for sg in [create_graph_1(StellarDiGraph()), create_graph_2(StellarDiGraph())]:
        schema = sg.create_graph_schema(create_type_maps=True)

        for n1, n2, k in sg.edges(keys=True):
            et = (
                sg.node[n1]["label"],
                sg.adj[n1][n2][k]["label"],
                sg.node[n2]["label"],
            )
            assert schema.get_edge_type((n1, n2, k)) == et

            # Check the is_of_edge_type function: it should only work one way for directed graphs
            assert schema.is_of_edge_type((n1, n2, k), et)
            assert ~schema.is_of_edge_type((n1, n2, k), (et[2], et[1], et[0]))


def test_graph_schema_sampling():
    for sg in [
        create_graph_1(),
        create_graph_2(),
        create_graph_1(StellarDiGraph()),
        create_graph_2(StellarDiGraph()),
    ]:
        schema = sg.create_graph_schema()
        type_list = schema.get_type_adjacency_list(["user", "movie"], n_hops=2)

        assert type_list[0][0] == "user"
        assert type_list[1][0] == "movie"

        for lt in type_list:
            adj_types = [t.n2 for t in schema.schema[lt[0]]]
            list_types = [type_list[adj_n][0] for adj_n in lt[1]]

            if len(list_types) > 0:
                assert set(adj_types) == set(list_types)


def test_graph_schema_sampling_layout_1():
    sg = create_graph_1()
    schema = sg.create_graph_schema(create_type_maps=True)
    sampling_layout = schema.get_sampling_layout(["user"], [2, 2])

    assert len(sampling_layout) == 1

    assert sampling_layout[0][2] == ("user", [2, 3])
    assert sampling_layout[0][1] == ("movie", [1])
    assert sampling_layout[0][0] == ("user", [0])

    sg = create_graph_2()
    schema = sg.create_graph_schema(create_type_maps=True)
    sampling_layout = schema.get_sampling_layout(["user"], [1, 2])

    assert len(sampling_layout) == 1

    assert sampling_layout[0] == [
        ("user", [0]),
        ("movie", [1]),
        ("user", [2]),
        ("movie", [3]),
        ("user", [4]),
        ("user", [5]),
        ("movie", [6]),
        ("user", [7]),
        ("movie", [8]),
        ("user", [9]),
        ("user", [10]),
    ]


def test_graph_schema_sampling_layout_multiple():
    sg = create_graph_1()
    schema = sg.create_graph_schema(create_type_maps=True)
    sampling_layout = schema.get_sampling_layout(["user", "movie"], [1, 2, 2])

    assert len(sampling_layout) == 2

    assert sampling_layout[0] == [
        ("user", [0]),
        ("movie", []),
        ("movie", [1]),
        ("user", []),
        ("user", [2]),
        ("movie", []),
        ("movie", [3, 4]),
        ("user", []),
    ]
    assert sampling_layout[1] == [
        ("user", []),
        ("movie", [0]),
        ("movie", []),
        ("user", [1]),
        ("user", []),
        ("movie", [2]),
        ("movie", []),
        ("user", [3, 4]),
    ]
