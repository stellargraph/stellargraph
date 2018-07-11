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

import networkx as nx
import random


class GraphWalk(object):
    """
    Base class for exploring graphs.
    """

    def __init__(self, graph):
        self.graph = graph

    def neighbors(self, graph, node):
        return list(nx.neighbors(graph, node))

    def run(self, **kwargs):
        """
        To be overridden by subclasses. It is the main entry point for performing random walks on the given
        graph.
        It should return the sequences of nodes in each random walk.

        Args:
            **kwargs:

        Returns:

        """


class UniformRandomWalk(GraphWalk):
    """
    Performs uniform random walks on the given graph
    """

    def run(self, nodes=None, n=None, length=None, seed=None):
        """

        Args:
            nodes: <list> The root nodes as a list of node IDs
            n: <int> Total number of random walks per root node
            length: <int> Maximum length of each random walk
            seed: <int> Random number generator seed; default is None

        Returns:
            <list> List of lists of nodes ids for each of the random walks

        """
        self._check_parameter_values(nodes=nodes, n=n, length=length, seed=seed)

        random.seed(seed)  # seed the random umber generator

        walks = []
        for node in nodes:  # iterate over root nodes
            for walk_number in range(n):  # generate n walks per root node
                walk = list()
                current_node = node
                for _ in range(length):
                    walk.extend([current_node])
                    neighbours = self.neighbors(self.graph, current_node)
                    if (
                        len(neighbours) == 0
                    ):  # for whatever reason this node has no neighbours so stop
                        break
                    else:
                        random.shuffle(neighbours)  # shuffles the list in place
                        current_node = neighbours[0]  # select the first node to follow

                walks.append(walk)

        return walks

    def _check_parameter_values(self, nodes, n, length, seed):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            nodes: <list> A list of root node ids such that from each node a uniform random walk of up to length l
            will be generated.
            n: <int> Number of walks per node id.
            length: <int> Maximum length of walk measured as the number of edges followed from root node.
            seed: <int> Random number generator seed

        """
        if nodes is None:
            raise ValueError("A list of root node IDs was not provided.")
        if type(nodes) != list:
            raise ValueError("nodes parameter should be a list of node IDs.")
        if (
            len(nodes) == 0
        ):  # this is not an error but maybe a warning should be printed to inform the caller
            print(
                "WARNING: No root node IDs given. An empty list will be returned as a result."
            )

        if n <= 0:
            raise ValueError(
                "The number of walks per root node, n, should be a positive integer."
            )
        if type(n) != int:
            raise ValueError(
                "The number of walks per root node, n, should be integer type."
            )

        if length <= 0:
            raise ValueError("The walk length, length, should be positive integer.")
        if type(length) != int:
            raise ValueError("The walk length, length, should be integer type.")

        if seed is not None:
            if seed < 0:
                raise ValueError(
                    "The random number generator seed value, seed, should be positive integer or None."
                )
            if type(seed) != int:
                raise ValueError(
                    "The random number generator seed value, seed, should be integer type or None."
                )


class BiasedRandomWalk(GraphWalk):
    """
    Performs biased second order random walks (like Node2Vec random walks)
    """

    def run(self, **kwargs):
        """

        :param p: p parameter in Node2Vec
        :param q: q parameter in Node2Vec
        :param n: Number of random walks
        :param l: Length of random walks
        :param e_types: List of edge types that the random walk is allowed to traverse. Set to None for homogeneous
        graphs with a single edge type.
        :return:
        """
        pass


class MetaPathWalk(GraphWalk):
    """
    For heterogeneous graphs, it performs walks based on given metapaths.
    """

    def run(self, **kwargs):
        """

        :param n: Number of walks for each given metapath
        :param l: Length of random walks
        :param mp: List of metapaths to drive the random walks
        :return:
        """
        pass


class DepthFirstWalk(GraphWalk):
    """
    Depth First Walk that generates all paths from a starting node to a given depth.
    It can be used to extract, in a memory efficient way, a sub-graph starting from a node and up to a given depth.
    """

    def run(self, **kwargs):
        """

        :param n: Number of walks. If it is equal to the number of nodes in the graph, then it generates all the
        sub-graphs with every node as the root and up to the given depth, d.
        :param d: Depth of walk as in distance (number of edges) from starting node.
        :return: (The return value might differ when compared to the other walk types with exception to
        BreadthFirstWalk defined next)
        """
        pass


class BreadthFirstWalk(GraphWalk):
    """
    Breadth First Walk that generates all paths from a starting node to a given depth.
    It can be used to extract a sub-graph starting from a node and up to a given depth.
    """

    pass


class SampledBreadthFirstWalk(GraphWalk):
    """
    Breadth First Walk that generates a sampled number of paths from a starting node.
    It can be used to extract a random sub-graph starting from a set of initial nodes.
    """

    def run(self, nodes=None, n=1, n_size=None):
        """

        Args:
            nodes:  <list> A list of root node ids such that from each node n BFWs will be generated up to the
            given depth d.
            n: <int> Number of walks per node id.
            n_size: <list> The number of neighbouring nodes to expand at each depth of the walk. Sampling of
            neighbours with replacement is always used regardless of the node degree and number of neighbours
            requested.

        Returns:
            A list of lists such that each list element is a sequence of ids corresponding to a BFW.
        """
        self._check_parameter_values(nodes=nodes, n=n, n_size=n_size)

        walks = []
        d = len(n_size)  # depth of search

        for node in nodes:  # iterate over root nodes
            for _ in range(n):  # do n bounded breadth first walks from each root node
                q = list()  # the queue of neighbours
                walk = list()  # the list of nodes in the subgraph of node
                q.extend(
                    [(node, 0)]
                )  # extend() needs iterable as parameter; we use list of tuples (node id, depth)

                while len(q) > 0:
                    # remove the top element in the queue and
                    frontier = q.pop(
                        0
                    )  # index 0 pop the item from the front of the list
                    depth = frontier[1] + 1  # the depth of the neighbouring nodes
                    walk.extend([frontier[0]])  # add to the walk
                    if (
                        depth <= d
                    ):  # consider the subgraph up to and including depth d from root node
                        neighbours = self.neighbors(self.graph, frontier[0])
                        if len(neighbours) == 0:
                            # Oops, this node has no neighbours and it doesn't have a self link.
                            # We can't handle this so raise an exception.
                            raise ValueError(
                                "Node with id {} has no neighbours and no self link. I don't know what to do!".format(
                                    frontier[0]
                                )
                            )
                        else:  # sample with replacement
                            neighbours = random.choices(neighbours, k=n_size[depth - 1])

                        # add them to the back of the queue
                        q.extend([(sampled_node, depth) for sampled_node in neighbours])

                # finished i-th walk from node so add it to the list of walks as a list
                walks.append(walk)

        return walks

    def _check_parameter_values(self, nodes, n, n_size):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            nodes: <list> A list of root node ids such that from each node n BFWs will be generated up to the
            given depth d.
            n: <int> Number of walks per node id.
            n_size: <list> The number of neighbouring nodes to expand at each depth of the walk.

        """
        if nodes is None:
            raise ValueError("A list of root node IDs was not provided.")
        if type(nodes) != list:
            raise ValueError("nodes parameter should be a list of node IDs.")
        if (
            len(nodes) == 0
        ):  # this is not an error but maybe a warning should be printed to inform the caller
            print(
                "WARNING: No root node IDs given. An empty list will be returned as a result."
            )

        if n <= 0:
            raise ValueError(
                "The number of walks per root node, n, should be a positive integer."
            )
        if type(n) != int:
            raise ValueError(
                "The number of walks per root node, n, should be integer type."
            )

        if n_size is None:
            raise ValueError(
                "The neighbourhood size, n_size, must be a list of integers not None"
            )
        if type(n_size) != list:
            raise ValueError(
                "The neighbourhood size, n_size, must be a list of integers"
            )

        if len(n_size) == 0:
            raise ValueError(
                "The neighbourhood size, n_size, should not be empty list."
            )

        for d in n_size:
            if type(d) != int:
                raise ValueError(
                    "The neighbourhood size, n_size, must be list of integers."
                )
