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
from stellargraph.core.schema import *
import itertools as it


@pytest.fixture
def example_graph_schema():
    def create_schema(aa=1, ab=1, ba=1, bb=1):
        # graph schema
        schema = {"A": [], "B": []}
        for ii in range(aa):
            schema["A"].append(EdgeType("A", "a%d" % ii, "A"))
        for ii in range(ab):
            schema["A"].append(EdgeType("A", "ab%d" % ii, "B"))
        for ii in range(ba):
            schema["B"].append(EdgeType("B", "ab%d" % ii, "A"))
        for ii in range(bb):
            schema["B"].append(EdgeType("B", "b%d" % ii, "B"))

        return GraphSchema(
            is_directed=False,
            node_types=sorted(schema.keys()),
            edge_types=sorted(set(it.chain(*schema.values()))),
            schema=schema,
        )

    return create_schema


def test_homogeneous_graph_schema(example_graph_schema):
    # Create dummy graph schema
    gs = example_graph_schema(bb=0)

    assert gs.node_index("A") == 0
    assert gs.node_index("B") == 1

    assert gs.edge_index(EdgeType("A", "a0", "A")) == 0
    assert gs.edge_index(EdgeType("A", "ab0", "B")) == 1
    assert gs.edge_index(EdgeType("B", "ab0", "A")) == 2

    # Directed schema
    gs = example_graph_schema(ba=0, bb=0)

    assert gs.edge_index(EdgeType("A", "a0", "A")) == 0
    assert gs.edge_index(EdgeType("A", "ab0", "B")) == 1

    with pytest.raises(ValueError):
        gs.edge_index(EdgeType("B", "ab0", "A"))


def test_graph_schema_sampling(example_graph_schema):
    # Create dummy graph schema
    schema = example_graph_schema(bb=0)

    # Only accepts lists & tuples
    with pytest.raises(TypeError):
        schema.type_adjacency_list("A", 2)

    # Only accepts int for n_hops
    with pytest.raises(TypeError):
        schema.type_adjacency_list("A", None)

    type_list = schema.type_adjacency_list(["A", "B"], n_hops=2)

    assert type_list[0][0] == "A"
    assert type_list[1][0] == "B"

    for lt in type_list:
        adj_types = [t.n2 for t in schema.schema[lt[0]]]
        list_types = [type_list[adj_n][0] for adj_n in lt[1]]

        if len(list_types) > 0:
            assert set(adj_types) == set(list_types)


def test_graph_schema_sampling_layout_1(example_graph_schema):
    # Create dummy graph schema
    schema = example_graph_schema(aa=0, bb=0)

    sampling_layout = schema.sampling_layout(["A"], [2, 2])

    assert len(sampling_layout) == 1

    assert sampling_layout[0][2] == ("A", [2, 3])
    assert sampling_layout[0][1] == ("B", [1])
    assert sampling_layout[0][0] == ("A", [0])

    # Check error handling
    schema = example_graph_schema(ab=2, ba=2, bb=0)

    # Only accepts lists & tuples
    with pytest.raises(TypeError):
        schema.sampling_layout("A", [1, 2])

    # Multiple edge types
    schema = example_graph_schema(ab=2, ba=2, bb=0)
    sampling_layout = schema.sampling_layout(("A",), [1, 2])

    assert len(sampling_layout) == 1

    assert sampling_layout[0] == [
        ("A", [0]),
        ("A", [1]),
        ("B", [2]),
        ("B", [3]),
        ("A", [4]),
        ("B", [5]),
        ("B", [6]),
        ("A", [7]),
        ("A", [8]),
        ("A", [9]),
        ("A", [10]),
    ]


def test_graph_schema_sampling_tree(example_graph_schema):
    # Create dummy graph schema
    schema = example_graph_schema(bb=0)
    type_list = schema.type_adjacency_list(["A", "B"], 3)
    _, type_tree = schema.sampling_tree(["A", "B"], 3)

    # Check that the tree corresponds to the adjacency list
    def check_tree(tree):
        items = []
        for x in tree:
            chd = check_tree(x[2])
            assert x[1] == type_list[x[0]][0]
            assert chd == type_list[x[0]][1]
            items.append(x[0])
        return items

    check_tree(type_tree)


def test_graph_schema_sampling_layout_multiple(example_graph_schema):
    # Create dummy graph schema
    schema = example_graph_schema(bb=0)
    sampling_layout = schema.sampling_layout(["A", "B"], [2, 2])

    assert len(sampling_layout) == 2
    assert all((x1[0] == x2[0]) for x1, x2 in zip(*sampling_layout))

    print(sampling_layout[0])

    assert sampling_layout[0] == [
        ("A", [0]),
        ("B", []),
        ("A", [1]),
        ("B", [2]),
        ("A", []),
        ("A", [3, 5]),
        ("B", [4, 6]),
        ("A", [7, 8]),
        ("A", []),
        ("B", []),
    ]
    assert sampling_layout[1] == [
        ("A", []),
        ("B", [0]),
        ("A", []),
        ("B", []),
        ("A", [1]),
        ("A", []),
        ("B", []),
        ("A", []),
        ("A", [2, 4]),
        ("B", [3, 5]),
    ]
