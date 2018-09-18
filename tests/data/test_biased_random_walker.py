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
from stellargraph.data.explorer import BiasedRandomWalk
from stellargraph.core.graph import StellarGraph


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

    g = StellarGraph(g)

    return g


class TestBiasedRandomWalk(object):
    def test_parameter_checking(self):
        g = create_test_graph()
        biasedrw = BiasedRandomWalk(g)

        nodes = ["0"]
        n = 1
        length = 2
        p = 1.
        q = 1.
        seed = None

        with pytest.raises(ValueError):
            # nodes should be a list of node ids even for a single node
            biasedrw.run(nodes=None, n=n, p=p, q=q, length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(
                nodes="0", n=n, p=p, q=q, length=length, seed=seed
            )  # can't just pass a node id, need list, e.g., ["0"]

        # n has to be positive integer
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=0, p=p, q=q, length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=-121, p=p, q=q, length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=21.4, p=p, q=q, length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=-0.5, p=p, q=q, length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=0.0001, p=p, q=q, length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n="2", p=p, q=q, length=length, seed=seed)

        # p has to be > 0.
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=0., q=q, length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=-0.25, q=q, length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=-1, q=q, length=length, seed=seed)

        # q has to be > 0.
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=0., length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=-0.9, length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=-75, length=length, seed=seed)

        # length has to be positive integer
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=0, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=-5, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=11.9, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=-9.9, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=q, length="10", seed=seed)

        # seed has to be None, 0,  or positive integer
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=-1)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=1010.8)

        # If no root nodes are given, an empty list is returned which is not an error but I thought this method
        # is the best for checking this behaviour.
        nodes = []
        subgraph = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=None)
        assert len(subgraph) == 0

    def test_walk_generation_single_root_node(self):

        g = create_test_graph()
        biasedrw = BiasedRandomWalk(g)

        nodes = ["0"]
        n = 1
        length = 1
        seed = 42
        p = 0.25
        q = 0.5

        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs[0]) == length

        length = 2
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        for subgraph in subgraphs:
            assert len(subgraph) == length

        length = 2
        n = 2
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length

        n = 3
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length

    def test_walk_generation_many_root_nodes(self):

        g = create_test_graph()
        biasedrw = BiasedRandomWalk(g)

        nodes = ["0", 2]
        n = 1
        length = 1
        seed = None
        p = 1.
        q = 0.3

        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n * len(nodes)
        for i, subgraph in enumerate(subgraphs):
            assert len(subgraph) == length  # should be 1
            assert subgraph[0] == nodes[i]  # should equal the root node

        length = 2
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n * len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) <= length

        n = 2
        length = 2
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n * len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) <= length

        length = 3
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n * len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) <= length

        n = 5
        length = 10
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n * len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) <= length

    def test_walk_generation_lonely_root_node(self):

        g = create_test_graph()
        biasedrw = BiasedRandomWalk(g)

        nodes = ["lonely"]  # this node has no edges including itself
        n = 1
        length = 1
        seed = None
        p = 0.5
        q = 1.

        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == 1
        assert (
            len(subgraphs[0]) == 1
        )  # always 1 since only the root node can every be added to the walk

        n = 10
        length = 1
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert (
                len(subgraph) == 1
            )  # always 1 since only the root node can ever be added to the walk

        n = 10
        length = 10
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert (
                len(subgraph) == 1
            )  # always 1 since only the root node can ever be added to the walk

    def test_walk_generation_self_lonely_root_node(self):

        g = create_test_graph()
        biasedrw = BiasedRandomWalk(g)

        nodes = ["self lonely"]  # this node has link to self but no other edges
        n = 1
        length = 1
        seed = None
        p = 1.
        q = 1.

        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == 1
        assert len(subgraphs[0]) == 1

        n = 10
        length = 1
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length
            for node in subgraph:
                assert node == "self lonely"  # all nodes should be the same node

        n = 1
        length = 99
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length
            for node in subgraph:
                assert node == "self lonely"  # all nodes should be the same node

        n = 10
        length = 10
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length
            for node in subgraph:
                assert node == "self lonely"  # all nodes should be the same node

    def test_walk_biases(self):
        graph = nx.Graph()
        # a square with a triangle:
        #   0-3
        #  /| |
        # 1-2-4
        graph.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
        graph = StellarGraph(graph)
        biasedrw = BiasedRandomWalk(graph)

        # there's 18 total walks of length 4 starting at 0 in `graph`,
        # and the non-tiny transition probabilities are always equal
        # so with a large enough sample, all the possible paths for a
        # given p, q should come up.
        nodes = [0]
        n = 1000
        seed = None
        length = 4

        always = 1e-100
        never = 1e100

        # always return to the last visited node
        p = always
        q = never
        walks = {
            tuple(w)
            for w in biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        }
        assert walks == {(0, 1, 0, 1), (0, 2, 0, 2), (0, 3, 0, 3)}

        # always explore (when possible)
        p = never
        q = always
        walks = {
            tuple(w)
            for w in biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        }
        assert walks == {
            # follow the square
            (0, 2, 4, 3),
            (0, 3, 4, 2),
            # go around the triangle (2 is a neighbour of 0 and so
            # isn't exploring, but q = never < 1)
            (0, 1, 2, 4),
        }

        # always go to a neighbour, if possible, otherwise equal
        # chance of returning or exploring
        p = never
        q = never
        walks = {
            tuple(w)
            for w in biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        }
        assert walks == {
            # follow the triangle
            (0, 1, 2, 0),
            (0, 2, 1, 0),
            # all explorations around the square should appear (none
            # are neighbours)
            (0, 3, 0, 1),
            (0, 3, 0, 2),
            (0, 3, 0, 3),
            (0, 3, 4, 3),
            (0, 3, 4, 2),
        }
