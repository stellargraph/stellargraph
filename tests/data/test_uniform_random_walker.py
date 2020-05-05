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
import numpy as np
from stellargraph.data.explorer import UniformRandomWalk
from ..test_utils.graphs import create_test_graph, example_graph_random


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
        # n has to be positive integer
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=0, length=length, seed=seed)
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=-121, length=length, seed=seed)
        with pytest.raises(TypeError):
            urw.run(nodes=nodes, n=21.4, length=length, seed=seed)
        with pytest.raises(TypeError):
            urw.run(nodes=nodes, n=-0.5, length=length, seed=seed)
        with pytest.raises(TypeError):
            urw.run(nodes=nodes, n=0.0001, length=length, seed=seed)
        with pytest.raises(TypeError):
            urw.run(nodes=nodes, n="2", length=length, seed=seed)

        # length has to be positive integer
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=n, length=0, seed=seed)
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=n, length=-5, seed=seed)
        with pytest.raises(TypeError):
            urw.run(nodes=nodes, n=n, length=11.9, seed=seed)
        with pytest.raises(TypeError):
            urw.run(nodes=nodes, n=n, length=-9.9, seed=seed)
        with pytest.raises(TypeError):
            urw.run(nodes=nodes, n=n, length="10", seed=seed)

        # seed has to be None, 0,  or positive integer
        with pytest.raises(ValueError):
            urw.run(nodes=nodes, n=n, length=length, seed=-1)
        with pytest.raises(TypeError):
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

    def test_walk_generation_loner_root_node(self):

        g = create_test_graph()
        urw = UniformRandomWalk(g)

        nodes = ["loner"]  # this node has no edges including itself
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

    def test_walk_generation_self_loner_root_node(self):

        g = create_test_graph()
        urw = UniformRandomWalk(g)

        nodes = ["self loner"]  # this node has link to self but no other edges
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
                assert node == "self loner"  # all nodes should be the same node

        n = 1
        length = 99
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length
            for node in subgraph:
                assert node == "self loner"  # all nodes should be the same node

        n = 10
        length = 10
        subgraphs = urw.run(nodes=nodes, n=n, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length
            for node in subgraph:
                assert node == "self loner"  # all nodes should be the same node

    def test_init_parameters(self):
        g = create_test_graph()

        nodes = ["0", 2]
        n = 1
        length = 2
        seed = 0

        urw = UniformRandomWalk(g, n=n, length=length, seed=seed)
        urw_no_params = UniformRandomWalk(g)

        run_1 = urw.run(nodes=nodes)
        run_2 = urw_no_params.run(nodes=nodes, n=n, length=length, seed=seed)
        assert np.array_equal(run_1, run_2)

    def test_benchmark_uniformrandomwalk(self, benchmark):
        g = example_graph_random(n_nodes=100, n_edges=500)
        urw = UniformRandomWalk(g)

        nodes = np.arange(0, 50)
        n = 2
        n = 5
        length = 5

        benchmark(lambda: urw.run(nodes=nodes, n=n, length=length))
