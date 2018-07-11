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
    sg.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label="rating")
    return sg


def create_graph_2(sg=StellarGraph()):
    sg.add_nodes_from([0, 1, 2, 3], label="movie")
    sg.add_nodes_from([4, 5], label="user")
    sg.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label="rating")
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
    node_labels = nx.get_node_attributes(sg, 'label')
    for n1, n2, k, edata in sg.edges(keys=True, data=True):
        assert (node_labels[n1], edata["label"], node_labels[n2]) == tuple(
            schema.get_edge_type((n1, n2, k))
        )

def test_digraph_schema():
    sg = create_graph_1(StellarDiGraph())
    schema = sg.create_graph_schema()

    assert "movie" in schema.schema
    assert "user" in schema.schema
    assert len(schema.schema["movie"]) == 1
    assert len(schema.schema["user"]) == 0

    # Test node type lookup
    for n, ndata in sg.nodes(data=True):
        assert ndata["label"] == schema.get_node_type(n)

    # Test edge type lookup
    node_labels = nx.get_node_attributes(sg, 'label')
    for n1, n2, k, edata in sg.edges(keys=True, data=True):
        assert (node_labels[n1], edata["label"], node_labels[n2]) == tuple(
            schema.get_edge_type((n1, n2, k))
        )

def test_graph_schema_sampling():
    for sg in [
        create_graph_1(),
        create_graph_2(),
        create_graph_1(StellarDiGraph()),
        create_graph_2(StellarDiGraph())
    ]:
        schema = sg.create_graph_schema()
        type_list = schema.get_type_adjacency_list(["user", "movie"], n_hops=2)

        assert type_list[0][0] == 'user'
        assert type_list[1][0] == 'movie'

        for lt in type_list:
            adj_types = [t.n2 for t in schema.schema[lt[0]]]
            list_types = [type_list[adj_n][0] for adj_n in lt[1]]

            if len(list_types) > 0:
                assert set(adj_types) == set(list_types)


test_graph_schema_sampling()