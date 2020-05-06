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

import pandas as pd
import numpy as np
import pytest
from stellargraph.data.explorer import UniformRandomMetaPathWalk
from stellargraph.core.graph import StellarGraph
from ..test_utils.graphs import example_graph_random


# FIXME (#535): Consider using graph fixtures
def create_test_graph():
    """
    Creates a simple graph for testing the BreadthFirstWalk class. The node ids are string or integers. Each node
    also has a label based on the type of its id such that nodes with string ids and those with integer ids have
    labels 's' and 'n' respectively.

    Returns:
        A simple graph with 13 nodes and 24 edges (including self loops for all but two of the nodes) in
        networkx format.

    """
    nodes = {
        "s": pd.DataFrame(index=["0", "5", "7", "self loner", "loner"]),
        "n": pd.DataFrame(index=[1, 2, 3, 4, 6, 8, 9, 10]),
    }
    edges = pd.DataFrame(
        [
            ("0", 1),
            ("0", 2),
            (1, 3),
            (1, 4),
            (3, 6),
            (4, "7"),
            (4, 8),
            (2, "5"),
            ("5", 9),
            ("5", 10),
            ("0", "0"),
            (1, 1),
            (3, 3),
            (6, 6),
            (4, 4),
            ("7", "7"),
            (8, 8),
            (2, 2),
            ("5", "5"),
            (9, 9),
            (
                "self loner",
                "self loner",
            ),  # node that is not connected with any other nodes but has self loop
        ],
        columns=["source", "target"],
    )

    return StellarGraph(nodes, edges)


class TestMetaPathWalk(object):
    def test_parameter_checking(self):
        g = create_test_graph()
        mrw = UniformRandomMetaPathWalk(g)

        nodes = [1]
        n = 1
        length = 2
        seed = None
        metapaths = [["n", "s", "n"]]

        # nodes should be a list of node ids even for a single node
        with pytest.raises(ValueError):
            mrw.run(nodes=None, n=n, length=length, metapaths=metapaths, seed=seed)
        with pytest.raises(ValueError):
            mrw.run(nodes=0, n=n, length=length, metapaths=metapaths, seed=seed)
        # n has to be positive integer
        with pytest.raises(ValueError):
            mrw.run(nodes=nodes, n=-1, length=length, metapaths=metapaths, seed=seed)
        with pytest.raises(TypeError):
            mrw.run(nodes=nodes, n=11.4, length=length, metapaths=metapaths, seed=seed)
        with pytest.raises(ValueError):
            mrw.run(nodes=nodes, n=0, length=length, metapaths=metapaths, seed=seed)
        # length has to be positive integer
        with pytest.raises(ValueError):
            mrw.run(nodes=nodes, n=n, length=-3, metapaths=metapaths, seed=seed)
        with pytest.raises(ValueError):
            mrw.run(nodes=nodes, n=n, length=0, metapaths=metapaths, seed=seed)
        with pytest.raises(TypeError):
            mrw.run(nodes=nodes, n=n, length=4.6, metapaths=metapaths, seed=seed)
        with pytest.raises(TypeError):
            mrw.run(nodes=nodes, n=n, length=1.0000001, metapaths=metapaths, seed=seed)
        # metapaths have to start and end with the same node type
        with pytest.raises(ValueError):
            mrw.run(nodes=nodes, n=n, length=length, metapaths=[["s", "n"]], seed=seed)
        with pytest.raises(ValueError):
            mrw.run(
                nodes=nodes,
                n=n,
                length=length,
                metapaths=[["s", "n", "s"], ["n", "s"]],
                seed=seed,
            )
        # metapaths have to have minimum length of two
        with pytest.raises(ValueError):
            mrw.run(nodes=nodes, n=n, length=length, metapaths=[["s"]], seed=seed)
        # metapaths has to be a list of lists of strings denoting the node labels
        with pytest.raises(ValueError):
            mrw.run(nodes=nodes, n=n, length=length, metapaths=["n", "s"], seed=seed)
        with pytest.raises(ValueError):
            mrw.run(nodes=nodes, n=n, length=length, metapaths=[[1, 2]], seed=seed)
        with pytest.raises(ValueError):
            mrw.run(
                nodes=nodes, n=n, length=length, metapaths=[["n", "s"], []], seed=seed
            )
        with pytest.raises(ValueError):
            mrw.run(
                nodes=nodes,
                n=n,
                length=length,
                metapaths=[["n", "s"], ["s", 1]],
                seed=seed,
            )
        with pytest.raises(ValueError):
            mrw.run(nodes=nodes, n=n, length=length, metapaths=[("n", "s")], seed=seed)
        with pytest.raises(ValueError):
            mrw.run(
                nodes=nodes,
                n=n,
                length=length,
                metapaths=(["n", "s"], ["s", "n", "s"]),
                seed=seed,
            )
        # seed has to be integer or None
        with pytest.raises(ValueError):
            mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=-1)
        with pytest.raises(TypeError):
            mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=1000.345)

        # If no root nodes are given, an empty list is returned which is not an error but I thought this method
        # is the best for checking this behaviour.
        walks = mrw.run(nodes=[], n=n, length=length, metapaths=metapaths, seed=seed)
        assert len(walks) == 0

    def test_walk_generation_single_root_node_loner(self):
        g = create_test_graph()
        mrw = UniformRandomMetaPathWalk(g)

        seed = None
        nodes = ["loner"]  # has no edges, not even to itself
        n = 1
        length = 5
        metapaths = [["s", "n", "s"]]

        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert len(walks) == n
        assert len(walks[0]) == 1

        n = 5
        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert len(walks) == n
        for walk in walks:
            assert len(walk) == 1

    def test_walk_generation_single_root_node_self_loner(self):
        g = create_test_graph()
        mrw = UniformRandomMetaPathWalk(g)

        seed = None
        nodes = ["self loner"]  # this node has self edges but not other edges
        n = 1
        length = 10
        metapaths = [["s", "n", "n", "s"]]

        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert len(walks) == n
        assert (
            len(walks[0]) == 1
        )  # for the ['s', 'n', 'n', 's'] metapath only the starting node is returned

        metapaths = [["s", "s"]]
        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert len(walks) == n
        assert len(walks[0]) == length  # the node is repeated length times
        for node in walks[0]:
            assert node == "self loner"

    def test_walk_generation_single_root_node(self):

        g = create_test_graph()
        mrw = UniformRandomMetaPathWalk(g)

        nodes = ["0"]
        n = 1
        length = 15
        metapaths = [["s", "n", "n", "s"]]
        seed = 42

        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert len(walks) == n
        assert len(walks[0]) <= length  # test against maximum walk length

        n = 5
        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert len(walks) == n
        assert len(walks[0]) <= length  # test against maximum walk length

        metapaths = [["s", "n", "s"], ["s", "n", "n", "s"]]
        n = 1
        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert len(walks) == n * len(metapaths)
        for walk in walks:
            assert len(walk) <= length  # test against maximum walk length

        metapaths = [["s", "n", "s"], ["s", "n", "n", "s"]]
        n = 5
        length = 100
        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert len(walks) == n * len(metapaths)
        for walk in walks:
            assert len(walk) <= length  # test against maximum walk length

        nodes = [8]
        metapaths = [["s", "n", "s"], ["s", "n", "n", "s"]]
        n = 5
        length = 100
        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert (
            len(walks) == 0
        )  # metapaths start with a node of type 's' but starting node is type 'n' so an empty list is returned

    def test_walk_generation_many_root_nodes(self):

        g = create_test_graph()
        mrw = UniformRandomMetaPathWalk(g)

        nodes = ["0", 2]
        n = 1
        length = 15
        metapaths = [["s", "n", "n", "s"]]
        seed = 42

        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert (
            len(walks) == 1
        )  # the starting node 2 should not generate a walk because it is of type 'n' not 's'
        assert len(walks[0]) <= length  # test against maximum walk length

        metapaths = [["s", "n", "n", "s"], ["n", "n", "s", "n"]]

        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert (
            len(walks) == 2
        )  # each starting node will generate one walk from each metapath
        for walk in walks:
            assert len(walk) <= length  # test against maximum walk length

        n = 2
        nodes = ["0", "5"]
        metapaths = [["s", "n", "n", "s"]]
        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert len(walks) == n * len(
            nodes
        )  # each starting node will generate one walk from each metapath
        for walk in walks:
            assert len(walk) <= length  # test against maximum walk length

        n = 2
        nodes = ["0", "5", 1, 6]
        metapaths = [["s", "n", "n", "s"]]
        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert (
            len(walks) == n * 2
        )  # the first two starting node will generate one walk from each metapath
        for walk in walks:
            assert len(walk) <= length  # test against maximum walk length

        n = 5
        nodes = ["0", "5", 1, 6]
        metapaths = [["s", "n", "n", "s"], ["n", "s", "n"], ["n", "n"]]
        walks = mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed)
        assert len(walks) == n * 6
        for walk in walks:
            assert len(walk) <= length  # test against maximum walk length

    def test_init_parameters(self):
        g = create_test_graph()
        n = 2
        length = 15
        metapaths = [["s", "n", "n", "s"]]
        seed = 42
        nodes = ["0", "5"]

        mrw = UniformRandomMetaPathWalk(
            g, n=n, length=length, metapaths=metapaths, seed=seed
        )
        mrw_no_params = UniformRandomMetaPathWalk(g)

        run_1 = mrw.run(nodes=nodes)
        run_2 = mrw_no_params.run(
            nodes=nodes, n=n, length=length, metapaths=metapaths, seed=seed
        )
        assert len(run_1) == len(run_2)
        assert all(np.array_equal(w1, w2) for w1, w2 in zip(run_1, run_2))

    def test_benchmark_uniformrandommetapathwalk(self, benchmark):
        g = example_graph_random(n_nodes=50, n_edges=500, node_types=2)
        mrw = UniformRandomMetaPathWalk(g)

        # this should be made larger to be more realistic, when it is fast enough
        nodes = np.arange(0, 5)
        n = 5
        length = 5
        metapaths = [
            ["n-0", "n-1", "n-1", "n-0"],
            ["n-0", "n-1", "n-0"],
            ["n-0", "n-0"],
        ]

        benchmark(lambda: mrw.run(nodes=nodes, n=n, length=length, metapaths=metapaths))
