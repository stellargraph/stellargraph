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
from stellar.data.explorer import BreadthFirstWalk


def create_test_graph():
    """
    Creates a simple graph for testing the BreadthFirstWalk class. The node ids are string or integers.

    :return: A simple graph with 10 nodes and 20 edges (including self loops for all but one of the nodes) in
    networkx format.
    """
    g = nx.Graph()
    edges = [
        ('0', 1),
        ('0', 2),
        (1, 3),
        (1, 4),
        (3, 6),
        (4, 7),
        (4, 8),
        (2, 5),
        (5, 9),
        (5, 10),
        ('0', '0'),
        (1, 1),
        (3, 3),
        (6, 6),
        (4, 4),
        (7, 7),
        (8, 8),
        (2, 2),
        (5, 5),
        (9, 9),
    ]

    g.add_edges_from(edges)

    return g


class TestBreadthFirstWalk(object):

    def test_parameter_checking(self):
        g = create_test_graph()
        bfw = BreadthFirstWalk(g)

        d = 1
        nodes = ['0', 1]
        n = 1
        n_size = 1

        with pytest.raises(ValueError):
            # nodes should be a list of node ids even for a single node
            bfw.run(nodes=None, d=d, n=n, n_size=n_size)
            bfw.run(nodes=0, d=d, n=n, n_size=n_size)
            # only list is acceptable type for nodes
            bfw.run(nodes=(1, 2, ), d=d, n=n, n_size=n_size)
            # d has to be positive integer or 0
            bfw.run(nodes=nodes, d=1.5, n=n, n_size=n_size)
            bfw.run(nodes=nodes, d=-1, n=n, n_size=n_size)
            # n has to be positive integer
            bfw.run(nodes=nodes, d=d, n=-1, n_size=n_size)
            bfw.run(nodes=nodes, d=d, n=10.1, n_size=n_size)
            bfw.run(nodes=nodes, d=d, n=0, n_size=n_size)
            # n_size has to be positive integer
            bfw.run(nodes=nodes, d=d, n=n, n_size=0)
            bfw.run(nodes=nodes, d=d, n=n, n_size=-5)
            bfw.run(nodes=nodes, d=d, n=-1, n_size=2.4)
            bfw.run(nodes=nodes, d=d, n=n, n_size=(1, 2, ))

        # If no root nodes are given, an empty list is returned which is not an error but I thought this method
        # is the best for checking this behaviour.
        nodes = []
        subgraph = bfw.run(nodes=nodes, d=d, n=n, n_size=n_size)
        assert len(subgraph) == 0


    def test_walk_generation_single_root_node(self):

        g = create_test_graph()
        bfw = BreadthFirstWalk(g)

        d = 0
        nodes = ['0']
        n = 1
        n_size = 1

        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs[0]) == 1

        d = 1
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs[0]) == 2

        n_size = 2
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs[0]) == (1-n_size**(d+1))//(1-n_size)

        n_size = 3
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs[0]) == (1-n_size**(d+1))//(1-n_size)

        d = 2
        n_size = 1
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs[0]) == 3

        n_size = 2
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs[0]) == (1-n_size**(d+1))//(1-n_size)

        n_size = 3
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs[0]) == (1-n_size**(d+1))//(1-n_size)

    def test_walk_generation_many_root_nodes(self):

        g = create_test_graph()
        bfw = BreadthFirstWalk(g)

        d = 0
        nodes = ['0', 2]
        n = 1
        n_size = 1

        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == 2
        for i, subgraph in enumerate(subgraphs):
            assert len(subgraph) == 1
            assert subgraph[0] == nodes[i]  # should equal the root node

        d = 1
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == 2
        for subgraph in subgraphs:
            assert len(subgraph) == 2

        n_size = 2
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == 2
        for subgraph in subgraphs:
            assert len(subgraph) == (1-n_size**(d+1))//(1-n_size)

        n_size = 3
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == 2
        for subgraph in subgraphs:
            assert len(subgraph) == (1-n_size**(d+1))//(1-n_size)

        n_size = 1
        d = 2
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == 2
        for subgraph in subgraphs:
            assert len(subgraph) == 3

        n_size = 2
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == 2
        for subgraph in subgraphs:
            assert len(subgraph) == (1-n_size**(d+1))//(1-n_size)

        n_size = 3
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == 2
        for subgraph in subgraphs:
            assert len(subgraph) == (1-n_size**(d+1))//(1-n_size)

    def test_walk_generation_number_of_walks_per_root_nodes(self):

        g = create_test_graph()
        bfw = BreadthFirstWalk(g)

        d = 0
        nodes = [1]
        n = 2
        n_size = 1

        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == 2
        for i, subgraph in enumerate(subgraphs):
            assert len(subgraph) == 1
            assert subgraph[0] == nodes[0]  # should equal the root node

        d = 1
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == 2
        for subgraph in subgraphs:
            assert len(subgraph) == 2

        n_size = 2
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == 2
        for subgraph in subgraphs:
            assert len(subgraph) == (1-n_size**(d+1))//(1-n_size)

        n_size = 3
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == 2
        for subgraph in subgraphs:
            assert len(subgraph) == (1-n_size**(d+1))//(1-n_size)

        #############################################################
        nodes = [1, 5]
        n_size = 1
        n = 2
        d = 1

        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == n*len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) == 2

        n_size = 2
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == n*len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) == (1-n_size**(d+1))//(1-n_size)

        n_size = 3
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == n*len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) == (1-n_size**(d+1))//(1-n_size)

        #############################################################
        nodes = [1, 5]
        n_size = 2
        n = 3
        d = 2

        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == n*len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) == (1-n_size**(d+1))//(1-n_size)

        n_size = 3
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == n*len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) == (1-n_size**(d+1))//(1-n_size)

        n_size = 4
        subgraphs = bfw.run(nodes=nodes, n=n, d=d, n_size=n_size)
        assert len(subgraphs) == n*len(nodes)
        for subgraph in subgraphs:
            assert len(subgraph) == (1-n_size**(d+1))//(1-n_size)
