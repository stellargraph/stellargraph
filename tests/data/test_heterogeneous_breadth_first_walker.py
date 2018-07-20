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
from stellar.data.explorer import SampledHeterogeneousBreadthFirstWalk
from stellar.data.stellargraph import StellarGraph


def create_simple_test_graph():
    """
    Creates a simple graph for testing the SampledHeterogeneousBreadthFirstWalk class. The node ids are string or integers.

    :return: A multi graph with 8 nodes and 9 edges (with no self loops but 1 with only a self loop and 1 isolated node)
    in StellarGraph format.
    """

    g = StellarGraph()

    g.add_nodes_from([0, 1, "5", 4, 7], label="user")
    g.add_nodes_from([2, 3, 6], label="movie")

    g.add_edges_from([(1, 2), (1, 3), ("5", 6), ("5", 3), (4, 2)], label="rating")
    g.add_edges_from([("5", 4), (1, 4), (1, "5")], label="friend")
    g.add_edges_from(
        [(7, 7)], label="friend"
    )  # isolated node with only a link back to itself

    return g


def create_multi_test_graph():
    """
    Creates a multi graph for testing the SampledHeterogeneousBreadthFirstWalk class. The node ids are string or
    integers.

    :return: A multi graph with 8 nodes and 9 edges (with no self loops but 1 with only a self loop and 1 isolated node)
    in StellarGraph format.
    """

    g = StellarGraph()

    g.add_nodes_from([0, 1, "5", 4, 7], label="user")
    g.add_nodes_from([2, 3, 6], label="movie")

    g.add_edges_from([(1, 2), (1, 3), ("5", 6), ("5", 3), (4, 2)], label="rating")
    g.add_edges_from([("5", 4), (1, 4), (1, "5")], label="friend")
    g.add_edges_from([(1, 4)], label="colleague")

    return g


class TestSampledHeterogeneousBreadthFirstWalk(object):
    def test_parameter_checking(self):
        g = create_simple_test_graph()

        bfw = SampledHeterogeneousBreadthFirstWalk(g)
        graph_schema = g.create_graph_schema(create_type_maps=True)

        nodes = [0, 1]
        n = 1
        n_size = [1]
        seed = 1001

        with pytest.raises(ValueError):
            # nodes should be a list of node ids even for a single node
            bfw.run(
                nodes=None, n=n, n_size=n_size, graph_schema=graph_schema, seed=seed
            )
        with pytest.raises(ValueError):
            bfw.run(nodes=0, n=n, n_size=n_size, graph_schema=graph_schema, seed=seed)
        with pytest.raises(ValueError):
            # only list is acceptable type for nodes
            bfw.run(
                nodes=(1, 2), n=n, n_size=n_size, graph_schema=graph_schema, seed=seed
            )
            # n has to be positive integer
        with pytest.raises(ValueError):
            bfw.run(
                nodes=nodes, n=-1, n_size=n_size, graph_schema=graph_schema, seed=seed
            )
        with pytest.raises(ValueError):
            bfw.run(
                nodes=nodes, n=10.1, n_size=n_size, graph_schema=graph_schema, seed=seed
            )
        with pytest.raises(ValueError):
            bfw.run(
                nodes=nodes, n=0, n_size=n_size, graph_schema=graph_schema, seed=seed
            )
            # n_size has to be list of positive integers
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, n_size=0, graph_schema=graph_schema, seed=seed)
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, n_size=[-5], graph_schema=graph_schema, seed=seed)
        with pytest.raises(ValueError):
            bfw.run(
                nodes=nodes, n=-1, n_size=[2.4], graph_schema=graph_schema, seed=seed
            )
        with pytest.raises(ValueError):
            bfw.run(
                nodes=nodes, n=n, n_size=(1, 2), graph_schema=graph_schema, seed=seed
            )
            # graph_schema must be None or GraphSchema type
        with pytest.raises(ValueError):
            bfw.run(
                nodes=nodes, n=n, n_size=n_size, graph_schema="graph schema", seed=seed
            )
        with pytest.raises(ValueError):
            bfw.run(nodes=nodes, n=n, n_size=n_size, graph_schema=9092, seed=seed)
            # seed must be positive integer or 0
        with pytest.raises(ValueError):
            bfw.run(
                nodes=nodes, n=n, n_size=n_size, graph_schema=graph_schema, seed=-1235
            )
        with pytest.raises(ValueError):
            bfw.run(
                nodes=nodes,
                n=n,
                n_size=n_size,
                graph_schema=graph_schema,
                seed=10.987665,
            )
        with pytest.raises(ValueError):
            bfw.run(
                nodes=nodes,
                n=n,
                n_size=n_size,
                graph_schema=graph_schema,
                seed=-982.4746,
            )
        with pytest.raises(ValueError):
            bfw.run(
                nodes=nodes,
                n=n,
                n_size=n_size,
                graph_schema=graph_schema,
                seed="don't be random",
            )

        # If no root nodes are given, an empty list is returned which is not an error but I thought this method
        # is the best for checking this behaviour.
        nodes = []
        subgraph = bfw.run(
            nodes=nodes, n=n, n_size=n_size, graph_schema=graph_schema, seed=seed
        )
        assert len(subgraph) == 0

    def test_walk_generation_single_root_node_loner(self):
        """
        Tests that the sampler behaves correctly when a root node is isolated with no self loop
        Returns:

        """
        g = create_simple_test_graph()
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        nodes = [0]  # this is an isolated user node with no self loop
        n = 1
        n_size = [0]

        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
        assert len(subgraphs) == 1
        assert len(subgraphs[0]) == 3
        assert subgraphs[0][0][0] == 0  # this should be the root node id
        # node 0 is of type 'user' and for the simple test graph it has 2 types of edges, rating, and friend,
        # so 2 empty subgraphs should be returned
        assert len(subgraphs[0][1]) == 0  # this should be empty list
        assert len(subgraphs[0][2]) == 0  # this should be the empty list

        # These test should return the same result as the one before regardless of the value of n_size
        n_size = [2, 3]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
        assert len(subgraphs) == 1
        assert len(subgraphs[0]) == 3
        assert subgraphs[0][0][0] == 0  # this should be the root node id
        # node 0 is of type 'user' and for the simple test graph it has 2 types of edges, rating, and friend,
        # so 2 empty subgraphs should be returned
        assert len(subgraphs[0][1]) == 0  # this should be empty list
        assert len(subgraphs[0][2]) == 0  # this should be empty

    def test_walk_generation_single_root_node_self_loner(self):
        g = create_simple_test_graph()
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        root_node_id = 7
        nodes = [
            root_node_id
        ]  # this node is only connected with itself with an edge of type "friend"
        n = 1

        n_size = [0]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
        assert len(subgraphs) == 1
        assert len(subgraphs[0]) == 3
        assert subgraphs[0][0][0] == root_node_id  # this should be the root node id
        # node 0 is of type 'user' and for the simple test graph it has 2 types of edges, rating, and friend,
        # so 2 empty subgraphs should be returned
        assert len(subgraphs[0][1]) == 0  # this should be empty list
        assert len(subgraphs[0][2]) == 0  # this should be the empty list

        n_size = [1]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
        assert len(subgraphs) == 1
        assert len(subgraphs[0]) == 3
        assert subgraphs[0][0][0] == root_node_id  # this should be the root node id
        # node 0 is of type 'user' and for the simple test graph it has 2 types of edges, rating, and friend,
        # so 1 subgraph with the root id corresponding to friend type edge and 1 empty subgraph should be returned
        assert subgraphs[0][1][0] == root_node_id  # this should be the root node id
        assert len(subgraphs[0][2]) == 0  # this should be the empty list

        n_size = [2, 2]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
        # The correct result should be [[[7], [7, 7], [], [7, 7], [], [7, 7], []]] so check
        assert len(subgraphs) == 1
        assert len(subgraphs[0]) == 7
        for level in subgraphs[0]:
            assert type(level) == list
            if len(level) > 0:
                for value in level:
                    assert value == root_node_id

        n_size = [2, 2, 3]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size)
        # The correct result should be
        # [[[7], [7, 7], [], [7, 7], [], [7, 7], [], [7, 7, 7], [], [7, 7, 7], [], [7, 7, 7], [], [7, 7, 7], []]]
        # so check
        assert len(subgraphs) == 1
        assert len(subgraphs[0]) == 15
        for level in subgraphs[0]:
            assert type(level) == list
            if len(level) > 0:
                for value in level:
                    assert value == root_node_id

    def test_walk_generation_single_root_node(self):

        g = create_simple_test_graph()
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        nodes = [3]
        n = 1
        n_size = [2]

        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=42)
        assert len(subgraphs) == n
        # should return [[[3], ['5', 1]]]
        assert subgraphs == [[[3], ["5", 1]]]

        n_size = [3]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=42)
        assert len(subgraphs) == n
        # should return [[[3], ['5', 1, 1]]]
        assert subgraphs == [[[3], ["5", 1, 1]]]

        n_size = [1, 1]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=42)
        assert len(subgraphs) == n
        # should return [[[3], ['5'], [4], [6]]]
        assert subgraphs == [[[3], ["5"], [4], [6]]]

        nodes = ["5"]
        n_size = [2, 3]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=42)
        assert len(subgraphs) == n
        # should return
        # [['5'], [1, 4], [6, 6], ['5', '5', '5'], [2, 2, 2], ['5', 1, '5'], [2, 2, 2], ['5', '5', '5'],
        # ['5', '5', '5']]]
        assert subgraphs == [
            [
                ["5"],
                [1, 4],
                [6, 6],
                ["5", "5", "5"],
                [2, 2, 2],
                ["5", 1, "5"],
                [2, 2, 2],
                ["5", "5", "5"],
                ["5", "5", "5"],
            ]
        ]

        nodes = ["5"]
        n_size = [2, 3]
        n = 3
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=42)
        assert len(subgraphs) == n
        # should return
        # [[['5'], [1, 4], [6, 6], ['5', '5', '5'], [2, 2, 2], ['5', 1, '5'], [2, 2, 2], ['5', '5', '5'],
        # ['5', '5', '5']], [['5'], [4, 4], [3, 6], ['5', '5', 1], [2, 2, 2], [1, 1, '5'], [2, 2, 2], ['5', '5', '5'],
        # ['5', '5', '5']], [['5'], [4, 4], [6, 6], [1, '5', '5'], [2, 2, 2], [1, 1, '5'], [2, 2, 2], ['5', '5', '5'],
        # ['5', '5', '5']]]
        valid_result = [
            [
                ["5"],
                [1, 4],
                [6, 6],
                ["5", "5", "5"],
                [2, 2, 2],
                ["5", 1, "5"],
                [2, 2, 2],
                ["5", "5", "5"],
                ["5", "5", "5"],
            ],
            [
                ["5"],
                [4, 4],
                [3, 6],
                ["5", "5", 1],
                [2, 2, 2],
                [1, 1, "5"],
                [2, 2, 2],
                ["5", "5", "5"],
                ["5", "5", "5"],
            ],
            [
                ["5"],
                [4, 4],
                [6, 6],
                [1, "5", "5"],
                [2, 2, 2],
                [1, 1, "5"],
                [2, 2, 2],
                ["5", "5", "5"],
                ["5", "5", "5"],
            ],
        ]
        for a, b in zip(subgraphs, valid_result):
            assert len(a) == len(b)
            assert a == b
        #
        # Test with multi-graph
        #
        g = create_multi_test_graph()
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        nodes = [1]
        n = 1
        n_size = [2]

        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=19893839)
        assert len(subgraphs) == n
        # should return [[[1], [4, 4], [4, 4], [2, 2]]]
        assert subgraphs == [[[1], [4, 4], [4, 4], [2, 2]]]

        n_size = [2, 3]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=19893839)
        assert len(subgraphs) == n
        #
        valid_result = [
            [
                [1],
                [4, 4],
                [4, 4],
                [2, 2],
                [1, 1, 1],
                [1, "5", "5"],
                [2, 2, 2],
                [1, 1, 1],
                ["5", 1, "5"],
                [2, 2, 2],
                [1, 1, 1],
                [1, "5", 1],
                [2, 2, 2],
                [1, 1, 1],
                [1, 1, "5"],
                [2, 2, 2],
                [4, 1, 1],
                [4, 1, 4],
            ]
        ]
        for a, b in zip(subgraphs, valid_result):
            assert len(a) == len(b)
            assert a == b

        nodes = [1]
        n_size = [2, 0]
        n = 2
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=19893839)
        assert len(subgraphs) == n
        valid_result = [
            [
                [1],
                [4, 4],
                [4, 4],
                [2, 2],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ],
            [
                [1],
                [4, 4],
                ["5", "5"],
                [2, 2],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ],
        ]
        for a, b in zip(subgraphs, valid_result):
            assert len(a) == len(b)
            assert a == b

    def test_walk_generation_many_root_nodes(self):

        g = create_simple_test_graph()
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        nodes = [0, 7]  # both nodes are type user
        n = 1
        n_size = [0]

        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=999)
        assert len(subgraphs) == len(nodes) * n
        for i, subgraph in enumerate(subgraphs):
            assert len(subgraph) == 3
            assert subgraph[0][0] == nodes[i]  # should equal the root node

        n_size = [1]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=999)
        assert len(subgraphs) == len(nodes) * n
        for i, subgraph in enumerate(subgraphs):
            assert len(subgraph) == 3

        n_size = [2]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=999)
        assert len(subgraphs) == len(nodes) * n
        for i, subgraph in enumerate(subgraphs):
            assert len(subgraph) == 3
        valid_result = [[[0], [], []], [[7], [7, 7], []]]
        for a, b in zip(subgraphs, valid_result):
            assert a == b

        n_size = [2, 2]
        nodes = [0, 4]
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=999)
        assert len(subgraphs) == len(nodes) * n
        assert subgraphs[0][0][0] == nodes[0]
        valid_result = [
            [[0], [], []],
            [[4], [1, "5"], [2, 2], [4, 4], [3, 2], [1, 1], [6, 3], [1, 1], [4, 4]],
        ]
        for a, b in zip(subgraphs, valid_result):
            assert a == b

        n_size = [2, 3]
        nodes = [1, 6]  # a user and a movie node respectively
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=999)
        assert len(subgraphs) == len(nodes) * n
        assert subgraphs[0][0][0] == nodes[0]
        valid_result = [
            [
                [1],
                ["5", 4],
                [3, 3],
                [4, 4, 1],
                [6, 3, 3],
                ["5", 1, "5"],
                [2, 2, 2],
                [1, "5", "5"],
                [1, "5", 1],
            ],
            [[6], ["5", "5"], [1, 1, 4], [3, 6, 3], [1, 1, 4], [3, 3, 6]],
        ]
        for a, b in zip(subgraphs, valid_result):
            assert a == b

        n = 5
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=999)
        assert len(subgraphs) == len(nodes) * n

        #
        # Test with multi-graph
        #
        g = create_multi_test_graph()
        bfw = SampledHeterogeneousBreadthFirstWalk(g)

        nodes = [1, 6]
        n = 1
        n_size = [2]

        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=999)
        assert len(subgraphs) == n * len(nodes)
        valid_result = [[[1], [4, 4], ["5", "5"], [2, 2]], [[6], ["5", "5"]]]
        assert subgraphs == valid_result

        n = 1
        n_size = [2, 3]

        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=999)
        assert len(subgraphs) == n * len(nodes)

        nodes = [4, "5", 0]
        n = 1
        n_size = [3, 3, 1]

        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=999)
        assert len(subgraphs) == n * len(nodes)

        n = 99
        subgraphs = bfw.run(nodes=nodes, n=n, n_size=n_size, seed=999)
        assert len(subgraphs) == n * len(nodes)
