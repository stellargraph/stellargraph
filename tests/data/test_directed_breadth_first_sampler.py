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

import random
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
        (1, "c1.1"),
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

        # sizes have to be list of non-negative integers
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=0, out_size=out_size)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=[-5])
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=-1, in_size=[2.4], out_size=out_size)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=(1, 2))

        # sizes have to have same length
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=[1, 2], out_size=[3])

        # okay to have zero sized samples
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=[0, 0])
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=[1, 0])
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=[0, 5])
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=[0, 0], out_size=out_size)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=[1, 0], out_size=out_size)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=[0, 5], out_size=out_size)
        # Is it okay if a zero appears in the same place for both in and out sizes? For now, yes.
        subgraph = bfw.run(nodes=nodes, n=n, in_size=[5, 0], out_size=[1, 0])
        assert len(subgraph) == len(nodes)

        # seed must be positive integer or 0
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=out_size, seed=-1235)
        with pytest.raises(ValueError):
            bfw.run(
                nodes=nodes, n=n, in_size=in_size, out_size=out_size, seed=10.987665
            )
        with pytest.raises(ValueError):
            bfw.run(
                nodes=nodes, n=n, in_size=in_size, out_size=out_size, seed=-982.4746
            )
        with pytest.raises(ValueError):
            bfw.run(
                nodes=nodes,
                n=n,
                in_size=in_size,
                out_size=out_size,
                seed="don't be random",
            )

        # If no root nodes are given, an empty list is returned, which is not an error.
        nodes = []
        subgraph = bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=out_size)
        assert len(subgraph) == 0

    def test_zero_hops(self):
        g = create_simple_graph()
        bfw = DirectedBreadthFirstNeighbours(g)
        # Check basic data structure
        # - The following case should be [[["root"]]]
        subgraph = bfw.run(nodes=["root"], n=1, in_size=[], out_size=[])
        assert len(subgraph) == 1
        assert len(subgraph[0]) == 1

    def test_one_hop(self):
        g = create_simple_graph()
        bfw = DirectedBreadthFirstNeighbours(g)
        # - The following case should be [[["root"], [None], [child]]]
        subgraph = bfw.run(nodes=["root"], n=1, in_size=[1], out_size=[1])
        assert len(subgraph) == 1
        assert len(subgraph[0]) == 3
        assert len(subgraph[0][0]) == 1
        assert subgraph[0][0][0] == "root"
        assert len(subgraph[0][1]) == 1
        assert subgraph[0][1][0] is None
        assert len(subgraph[0][2]) == 1
        assert subgraph[0][2][0] in ["0", 1, 2]
        # - The following case should be [[["root"], [None], []]]
        subgraph = bfw.run(nodes=["root"], n=1, in_size=[1], out_size=[0])
        assert len(subgraph) == 1
        assert len(subgraph[0]) == 3
        assert len(subgraph[0][2]) == 0
        # - The following case should be [[["root"], [], []]]
        subgraph = bfw.run(nodes=["root"], n=1, in_size=[0], out_size=[0])
        assert len(subgraph) == 1
        assert len(subgraph[0]) == 3
        assert len(subgraph[0][1]) == 0
        # - The following case should be [[["root"], [None, None, None], [child, child, child, child]]]
        subgraph = bfw.run(nodes=["root"], n=1, in_size=[3], out_size=[4])
        assert len(subgraph) == 1
        assert len(subgraph[0]) == 3
        assert len(subgraph[0][1]) == 3
        for child in subgraph[0][1]:
            assert child is None
        assert len(subgraph[0][2]) == 4
        for child in subgraph[0][2]:
            assert child in ["0", 1, 2]

    def test_two_hops(self):
        g = create_simple_graph()
        bfw = DirectedBreadthFirstNeighbours(g)
        # - The following case should be [[["root"], [None], [child*2], [None], [None*2], ["root"*2], [grandchild*4]]]
        in_size = [1, 1]
        out_size = [2, 2]
        subgraph = bfw.run(nodes=["root"], n=1, in_size=in_size, out_size=out_size)
        assert len(subgraph) == 1
        assert len(subgraph[0]) == 7
        assert len(subgraph[0][1]) == in_size[0]
        assert subgraph[0][1][0] is None
        assert len(subgraph[0][2]) == out_size[0]
        for child in subgraph[0][2]:
            assert child in ["0", 1, 2]
        assert len(subgraph[0][3]) == in_size[0] * in_size[1]
        assert subgraph[0][3][0] is None
        assert len(subgraph[0][4]) == in_size[0] * out_size[1]
        for child in subgraph[0][4]:
            assert child is None
        assert len(subgraph[0][5]) == out_size[0] * in_size[1]
        for parent in subgraph[0][5]:
            assert parent == "root"
        assert len(subgraph[0][6]) == out_size[0] * out_size[1]
        for grandchild in subgraph[0][6]:
            assert grandchild in [None, "c1.1", "c2.1", "c2.2"]
        for idx, child in enumerate(subgraph[0][2]):
            grandchildren = subgraph[0][6][(2 * idx) : (2 * idx + 2)]
            if child == "0":
                for grandchild in grandchildren:
                    assert grandchild is None
            elif child == 1:
                for grandchild in grandchildren:
                    assert grandchild == "c1.1"
            else:  # child == 2
                for grandchild in grandchildren:
                    assert grandchild in ["c2.1", "c2.2"]
        # - Check structure size for multiple start nodes
        # - For each start node, should be [[[node], [in], [out], [in.in], [in.out], [out.in], [out.out]]]
        nodes = list(g.nodes())
        in_size = [2, 3]
        out_size = [4, 5]
        subgraph = bfw.run(nodes=nodes, n=1, in_size=in_size, out_size=out_size)
        assert len(subgraph) == len(nodes)
        for node_graph in subgraph:
            assert len(node_graph) == 7
            assert len(node_graph[0]) == 1  # 1 start node
            assert len(node_graph[1]) == in_size[0]
            assert len(node_graph[2]) == out_size[0]
            assert len(node_graph[3]) == in_size[0] * in_size[1]
            assert len(node_graph[4]) == in_size[0] * out_size[1]
            assert len(node_graph[5]) == out_size[0] * in_size[1]
            assert len(node_graph[6]) == out_size[0] * out_size[1]

    def test_three_hops(self):
        g = create_test_graph()
        bfw = DirectedBreadthFirstNeighbours(g)
        graph_nodes = list(g.nodes())
        for _ in range(50):
            node = random.choice(graph_nodes)
            in_size = [random.randint(0, 2) for _ in range(3)]
            out_size = [random.randint(0, 2) for _ in range(3)]
            subgraph = bfw.run(nodes=[node], n=1, in_size=in_size, out_size=out_size)
            assert len(subgraph) == 1
            assert len(subgraph[0]) == 15  # 2**(num_hops+1) - 1 = 15
            assert len(subgraph[0][0]) == 1
            assert len(subgraph[0][1]) == in_size[0]
            assert len(subgraph[0][2]) == out_size[0]
            assert len(subgraph[0][3]) == in_size[0] * in_size[1]
            assert len(subgraph[0][4]) == in_size[0] * out_size[1]
            assert len(subgraph[0][5]) == out_size[0] * in_size[1]
            assert len(subgraph[0][6]) == out_size[0] * out_size[1]
            assert len(subgraph[0][7]) == in_size[0] * in_size[1] * in_size[2]
            assert len(subgraph[0][8]) == in_size[0] * in_size[1] * out_size[2]
            assert len(subgraph[0][9]) == in_size[0] * out_size[1] * in_size[2]
            assert len(subgraph[0][10]) == in_size[0] * out_size[1] * out_size[2]
            assert len(subgraph[0][11]) == out_size[0] * in_size[1] * in_size[2]
            assert len(subgraph[0][12]) == out_size[0] * in_size[1] * out_size[2]
            assert len(subgraph[0][13]) == out_size[0] * out_size[1] * in_size[2]
            assert len(subgraph[0][14]) == out_size[0] * out_size[1] * out_size[2]

    def test_benchmark_bfs_walk(self, benchmark):
        g = create_test_graph()
        bfw = DirectedBreadthFirstNeighbours(g)

        nodes = ["0"]
        n = 5
        in_size = [5, 5]
        out_size = [5, 5]

        benchmark(lambda: bfw.run(nodes=nodes, n=n, in_size=in_size, out_size=out_size))
