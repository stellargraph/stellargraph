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
from stellar.data.explorer import UniformRandomWalk


def create_test_graph():
    """
    Creates a simple graph for testing the BreadthFirstWalk class. The node ids are string or integers.

    :return: A simple graph with 13 nodes and 24 edges (including self loops for all but two of the nodes) in
    networkx format.
    """
    g = nx.Graph()
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
        ("self lonely", "self lonely"),  # an isolated node with a self link
    ]

    g.add_edges_from(edges)

    g.add_node("lonely")  # an isolated node without self link

    return g


class TestUniformRandomWalk(object):
    def test_parameter_checking(self):
        g = create_test_graph()
        urw = UniformRandomWalk(g)

        nodes = ["0"]
        n = 1
        length = 2
        seed = None

        # nodes should be a list of node ids even for a single node
        with pytest.raises(ValueError):
            urw.run(nodes=None, n=n, length=length, seed=seed)
        with pytest.raises(ValueError):
            urw.run(
                nodes="0", n=n, length=length, seed=seed
            )  # can't just pass a node id, need list, e.g., ["0"]
        with pytest.raises(ValueError):
            urw.run(
                nodes=(1, 2), n=n, length=length, seed=seed
            )  # tuple is not accepted, only list
        # n has to be positive integer
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=0, length=length, seed=seed)
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=-121, length=length, seed=seed)
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=21.4, length=length, seed=seed)
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=-0.5, length=length, seed=seed)
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=0.0001, length=length, seed=seed)
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n="2", length=length, seed=seed)

        # length has to be positive integer
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=n, length=0, seed=seed)
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=n, length=-5, seed=seed)
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=n, length=11.9, seed=seed)
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=n, length=-9.9, seed=seed)
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=n, length="10", seed=seed)

        # seed has to be None, 0,  or positive integer
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=n, length=length, seed=-1)
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=n, length=length, seed=1010.8)

        # If no root nodes are given, an empty list is returned which is not an error but I thought this method
        # is the best for checking this behaviour.
        nodes = []
        subgraph = urw.run(nodes=nodes, n=n, length=length, seed=None)
        assert len(subgraph) == 0

    def test_walk_generation_single_root_node(self):

        g = create_test_graph()
        urw = UniformRandomWalk(g)

        nodes = ["0"]
        n = 1
        length = 1
        seed = 42

        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs[0]) == length

        length = 2
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        for subgraph in subgraphs:
            assert len(subgraph) == length

        length = 2
        n = 2
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length

        n = 3
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length

    def test_walk_generation_many_root_nodes(self):

        g = create_test_graph()
        urw = UniformRandomWalk(g)

        nodes = ["0", 2]
        n = 1
        length = 1
        seed = None

        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n * len(nodes)
        for i, subgraph in enumerate(subgraphs):
            assert len(subgraph) == length  # should be 1
            assert subgraph[0] == nodes[i]  # should equal the root node

        length = 2
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n * len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) <= length

        n = 2
        length = 2
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n * len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) <= length

        length = 3
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n * len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) <= length

        n = 5
        length = 10
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n * len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) <= length

    def test_walk_generation_lonely_root_node(self):

        g = create_test_graph()
        urw = UniformRandomWalk(g)

        nodes = ["lonely"]  # this node has no edges including itself
        n = 1
        length = 1
        seed = None

        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == 1
        assert (
            len(subgraphs[0]) == 1
        )  # always 1 since only the root node can every be added to the walk

        n = 10
        length = 1
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert (
                len(subgraph) == 1
            )  # always 1 since only the root node can ever be added to the walk

        n = 10
        length = 10
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert (
                len(subgraph) == 1
            )  # always 1 since only the root node can ever be added to the walk

    def test_walk_generation_self_lonely_root_node(self):

        g = create_test_graph()
        urw = UniformRandomWalk(g)

        nodes = ["self lonely"]  # this node has link to self but no other edges
        n = 1
        length = 1
        seed = None

        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == 1
        assert len(subgraphs[0]) == 1

        n = 10
        length = 1
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length
            for node in subgraph:
                assert node == "self lonely"  # all nodes should be the same node

        n = 1
        length = 99
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length
            for node in subgraph:
                assert node == "self lonely"  # all nodes should be the same node

        n = 10
        length = 10
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length
            for node in subgraph:
                assert node == "self lonely"  # all nodes should be the same node
