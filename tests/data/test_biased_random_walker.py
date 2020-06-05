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

import numpy as np
import pandas as pd
import pytest
import networkx as nx
from stellargraph.data.explorer import BiasedRandomWalk
from stellargraph.core.graph import StellarGraph
from ..test_utils.graphs import create_test_graph, example_graph_random


# FIXME (#535): Consider using graph fixtures
def create_test_weighted_graph(is_multigraph=False):
    """
    Creates a simple graph for testing the weighted biased random walk class. The node ids are string or integers.

    :return: .
    """
    nodes = pd.DataFrame(
        index=["0", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "self loner", "loner"]
    )
    edges = pd.DataFrame(
        [
            ("0", 1, 3),
            ("0", 2, 4),
            (1, 3, 1),
            (1, 4, 7),
            (3, 6, 9),
            (4, 7, 2),
            (4, 8, 5),
            (2, 5, 7),
            (5, 9, 5),
            (5, 10, 6),
            ("0", "0", 7),
            (1, 1, 8),
            (3, 3, 8),
            (6, 6, 9),
            (4, 4, 1),
            (7, 7, 2),
            (8, 8, 3),
            (2, 2, 4),
            (5, 5, 5),
            (9, 9, 6),
            ("self loner", "self loner", 0),  # an isolated node with a self link
        ],
        columns=["source", "target", "weight"],
    )

    return StellarGraph(nodes, edges)


def weighted(a, b, c, d):
    nodes = pd.DataFrame(index=[1, 2, 3, 4])
    edges = pd.DataFrame(
        [(1, 2, a), (2, 3, b), (3, 4, c), (4, 1, d)],
        columns=["source", "target", "weight"],
    )
    return StellarGraph(nodes, edges)


class TestBiasedWeightedRandomWalk(object):
    def test_parameter_checking(self):
        g = create_test_weighted_graph()
        biasedrw = BiasedRandomWalk(g)

        nodes = ["0"]
        n = 1
        length = 2
        p = 1.0
        q = 1.0
        seed = None

        with pytest.raises(ValueError):
            # weighted must be a Boolean.
            biasedrw.run(
                nodes=nodes, n=n, p=p, q=q, length=length, seed=seed, weighted="unknown"
            )

    def test_init_parameters(self):
        g = weighted(1, 2, 3, 4)
        nodes = list(g.nodes())
        n = 4
        length = 4
        seed = 42
        p = 1.0
        q = 1.0

        rw = BiasedRandomWalk(g, n=n, p=p, q=q, length=length, seed=seed, weighted=True)
        rw_no_params = BiasedRandomWalk(g)

        run_1 = rw.run(nodes=nodes)
        run_2 = rw_no_params.run(
            nodes=nodes, n=n, p=p, q=q, length=length, seed=seed, weighted=True
        )
        assert np.array_equal(run_1, run_2)

    def test_identity_unweighted_weighted_1_walks(self):

        # graph with all edge weights = 1
        g = weighted(1, 1, 1, 1)
        nodes = g.nodes()
        n = 4
        length = 4
        seed = 42
        p = 1.0
        q = 1.0

        biasedrw = BiasedRandomWalk(g)
        run_1 = biasedrw.run(
            nodes=nodes, n=n, p=p, q=q, length=length, seed=seed, weighted=True
        )
        run_2 = biasedrw.run(
            nodes=nodes, n=n, p=p, q=q, length=length, seed=seed, weighted=True
        )
        assert np.array_equal(run_1, run_2)

    def test_weighted_walks(self):

        # all positive walks
        g = weighted(1, 2, 3, 4)

        nodes = list(g.nodes())
        n = 1
        length = 1
        seed = None
        p = 1.0
        q = 1.0

        biasedrw = BiasedRandomWalk(g)
        assert (
            len(
                biasedrw.run(
                    nodes=nodes, n=n, p=p, q=q, length=length, seed=seed, weighted=True
                )
            )
            == 4
        )

        # negative edge
        g = weighted(1, -2, 3, 4)

        biasedrw = BiasedRandomWalk(g)

        neg_message = r"graph: expected all edge weights to be non-negative and finite, found some negative or infinite: 1 to 2 \(weight = -2\)"

        with pytest.raises(ValueError, match=neg_message):
            BiasedRandomWalk(g, weighted=True)

        with pytest.raises(ValueError, match=neg_message):
            biasedrw.run(
                nodes=nodes, n=n, p=p, q=q, length=length, seed=seed, weighted=True
            )

        # edge with weight infinity
        g = weighted(1, np.inf, 3, 4)

        biasedrw = BiasedRandomWalk(g)

        inf_message = r"graph: expected all edge weights to be non-negative and finite, found some negative or infinite: 1 to 2 \(weight = inf\)"
        with pytest.raises(ValueError, match=inf_message):
            BiasedRandomWalk(g, weighted=True)

        with pytest.raises(ValueError, match=inf_message):
            biasedrw.run(
                nodes=nodes, n=n, p=p, q=q, length=length, seed=seed, weighted=True
            )

    def test_weighted_graph_label(self):

        g = nx.Graph()
        edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
        g.add_edges_from(edges)
        g[1][2]["w"] = 1
        g[2][3]["w"] = 2
        g[3][4]["w"] = 3
        g[4][1]["w"] = 4

        g = StellarGraph.from_networkx(g, edge_weight_attr="w")

        nodes = list(g.nodes())
        n = 1
        length = 1
        seed = None
        p = 1.0
        q = 1.0

        biasedrw = BiasedRandomWalk(g)

        assert (
            len(
                biasedrw.run(
                    nodes=nodes, n=n, p=p, q=q, length=length, seed=seed, weighted=True
                )
            )
            == 4
        )

        g = nx.Graph()
        edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
        g.add_edges_from(edges)
        g[1][2]["wt"] = 1
        g[2][3]["wt"] = 2
        g[3][4]["wt"] = 3
        g[4][1]["wt"] = 4

    def test_benchmark_biasedweightedrandomwalk(self, benchmark):
        g = example_graph_random(n_nodes=100, n_edges=500)
        biasedrw = BiasedRandomWalk(g)

        nodes = np.arange(0, 50)
        n = 20
        p = 2
        q = 3
        length = 10

        benchmark(
            lambda: biasedrw.run(
                nodes=nodes, n=n, p=p, q=q, length=length, weighted=True
            )
        )


class TestBiasedRandomWalk(object):
    def test_parameter_checking(self):
        g = create_test_graph()
        biasedrw = BiasedRandomWalk(g)

        nodes = ["0"]
        n = 1
        length = 2
        p = 1.0
        q = 1.0
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
        with pytest.raises(TypeError):
            biasedrw.run(nodes=nodes, n=21.4, p=p, q=q, length=length, seed=seed)
        with pytest.raises(TypeError):
            biasedrw.run(nodes=nodes, n=-0.5, p=p, q=q, length=length, seed=seed)
        with pytest.raises(TypeError):
            biasedrw.run(nodes=nodes, n=0.0001, p=p, q=q, length=length, seed=seed)
        with pytest.raises(TypeError):
            biasedrw.run(nodes=nodes, n="2", p=p, q=q, length=length, seed=seed)

        # p has to be > 0.
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=0.0, q=q, length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=-0.25, q=q, length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=-1, q=q, length=length, seed=seed)

        # q has to be > 0.
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=0.0, length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=-0.9, length=length, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=-75, length=length, seed=seed)

        # length has to be positive integer
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=0, seed=seed)
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=-5, seed=seed)
        with pytest.raises(TypeError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=11.9, seed=seed)
        with pytest.raises(TypeError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=-9.9, seed=seed)
        with pytest.raises(TypeError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=q, length="10", seed=seed)

        # seed has to be None, 0,  or positive integer
        with pytest.raises(ValueError):
            biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=-1)
        with pytest.raises(TypeError):
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
        p = 1.0
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

    def test_walk_generation_loner_root_node(self):

        g = create_test_graph()
        biasedrw = BiasedRandomWalk(g)

        nodes = ["loner"]  # this node has no edges including itself
        n = 1
        length = 1
        seed = None
        p = 0.5
        q = 1.0

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

    def test_walk_generation_self_loner_root_node(self):

        g = create_test_graph()
        biasedrw = BiasedRandomWalk(g)

        nodes = ["self loner"]  # this node has link to self but no other edges
        n = 1
        length = 1
        seed = None
        p = 1.0
        q = 1.0

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
                assert node == "self loner"  # all nodes should be the same node

        n = 1
        length = 99
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length
            for node in subgraph:
                assert node == "self loner"  # all nodes should be the same node

        n = 10
        length = 10
        subgraphs = biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length, seed=seed)
        assert len(subgraphs) == n
        for subgraph in subgraphs:
            assert len(subgraph) == length
            for node in subgraph:
                assert node == "self loner"  # all nodes should be the same node

    def test_walk_biases(self):
        # a square with a triangle:
        #   0-3
        #  /| |
        # 1-2-4
        nodes = pd.DataFrame(index=range(5))
        edges = pd.DataFrame(
            [(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)],
            columns=["source", "target"],
        )
        graph = StellarGraph(nodes, edges)
        biasedrw = BiasedRandomWalk(graph)

        # there's 18 total walks of length 4 starting at 0 in `graph`,
        # and the non-tiny transition probabilities are always equal
        # so with a large enough sample, all the possible paths for a
        # given p, q should come up.
        nodes = [0]
        n = 1000
        seed = None
        length = 4

        always = 1e-20
        never = 1e20

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

    def test_benchmark_biasedrandomwalk(self, benchmark):
        g = example_graph_random(n_nodes=100, n_edges=500)
        biasedrw = BiasedRandomWalk(g)

        nodes = np.arange(0, 50)
        n = 2
        p = 2
        q = 3
        length = 5

        benchmark(lambda: biasedrw.run(nodes=nodes, n=n, p=p, q=q, length=length))
