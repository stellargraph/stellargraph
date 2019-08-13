# -*- coding: utf-8 -*-
#
# Copyright 2017-2019 Data61, CSIRO
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
from stellargraph.data.explorer import DirectedBreadthFirstNeighbours
from stellargraph.core.graph import StellarGraph, StellarDiGraph

def create_simple_graph():
    """
    Creates a simple directed graph for testing. The node ids are string or integers.

    :return: A small, directed graph with 4 nodes and 6 edges in StellarDiGraph format.
    """

    g = nx.DiGraph()
    edges = [
        ("root", 2),
        ("root", 1),
        ("root", "0"),
        (2, "c2.1"),
        (2, "c2.2"),
        (1, "c1.1")
    ]
    g.add_edges_from(edges)
    return StellarDiGraph(g)


def create_test_graph():
    """
    Creates a simple graph for testing. The node ids are string or integers.

    :return: A simple graph with 13 nodes and 24 edges (including self loops for all but two of the nodes) in
    StellarDiGraph format.
    """
    g = nx.DiGraph()
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
        (
            "self loner",
            "self loner",
        ),  # node that is not connected with any other nodes but has self loop
    ]

    g.add_edges_from(edges)
    g.add_node(
        "loner"
    )  # node that is not connected to any other nodes and not having a self loop

    return StellarDiGraph(g)


class TestDirectedBreadthFirstNeighbours(object):
    def test_parameter_checking(self):
        g = create_test_graph()
        bfw = DirectedBreadthFirstNeighbours(g)

        nodes = ["0", 1]
        n = 1
        in_size = [1]
        out_size = [1]

        with pytest.raises(ValueError):
            # nodes should be a list of node ids even for a single node
            bfw.run(nodes=None, n=n, in_size=in_size, out_size=out_size)
        with pytest.raises(ValueError):
            bfw.run(nodes=0, n=n, in_size=in_size, out_size=out_size)

        # n has to be positive integer
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=-1, in_size=in_size, out_size=out_size)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=10.1, in_size=in_size, out_size=out_size)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=0, in_size=in_size, out_size=out_size)

        # sizes have to be list of positive integers
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=0, out_size=out_size)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=[-5], out_size=out_size)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=-1, in_size=[2.4], out_size=out_size)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=(1, 2), out_size=out_size)

        # seed must be positive integer or 0
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=out_size, seed=-1235)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=out_size, seed=10.987665)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=out_size, seed=-982.4746)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=out_size, seed="don't be random")

        # If no root nodes are given, an empty list is returned, which is not an error.
        nodes = []
        subgraph = bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=out_size)
        assert len(subgraph) == 0

    def test_benchmark_bfs_walk(self, benchmark):
        g = create_test_graph()
        bfw = DirectedBreadthFirstNeighbours(g)

        nodes = ["0"]
        n = 5
        in_size = [5, 5]
        out_size = [5, 5]

        benchmark(lambda: bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=out_size))
