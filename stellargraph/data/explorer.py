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

__all__ = [
    "UniformRandomWalk",
    "BiasedRandomWalk",
    "UniformRandomMetaPathWalk",
    "DepthFirstWalk",
    "BreadthFirstWalk",
    "SampledBreadthFirstWalk",
    "SampledHeterogeneousBreadthFirstWalk",
]


import networkx as nx
import numpy as np
import random
from collections import defaultdict

from ..core.schema import GraphSchema
from ..core.graph import StellarGraphBase
from ..core.utils import is_real_iterable


class GraphWalk(object):
    """
    Base class for exploring graphs.
    """

    def __init__(self, graph, graph_schema=None, seed=None):
        self.graph = graph

        # Initialize the random state
        self._random_state = random.Random(seed)

        # Initialize a numpy random state (for numpy random methods)
        self._np_random_state = np.random.RandomState(seed=seed)

        # We require a StellarGraph for this
        if not isinstance(graph, StellarGraphBase):
            raise TypeError(
                "Graph must be a StellarGraph or StellarDiGraph to use heterogeneous sampling."
            )

        if not graph_schema:
            self.graph_schema = self.graph.create_graph_schema(create_type_maps=True)
        else:
            self.graph_schema = graph_schema

        if self.graph_schema is not None and type(self.graph_schema) is not GraphSchema:
            raise ValueError(
                "({}) The parameter graph_schema should be either None or of type GraphSchema.".format(
                    type(self).__name__
                )
            )

        # Create a dict of adjacency lists per edge type, for faster neighbour sampling from graph in SampledHeteroBFS:
        # TODO: this could be better placed inside StellarGraph class
        edge_types = self.graph_schema.edge_types
        self.adj = dict()
        for et in edge_types:
            self.adj.update({et: defaultdict(lambda: [None])})

        for n1, nbrdict in graph.adjacency():
            for et in edge_types:
                neigh_et = [
                    n2
                    for n2, nkeys in nbrdict.items()
                    for k in iter(nkeys)
                    if self.graph_schema.is_of_edge_type((n1, n2, k), et)
                ]
                # Create adjacency list in lexographical order
                # Otherwise sampling methods will not be deterministic
                # even when the seed is set.
                self.adj[et][n1] = sorted(neigh_et, key=str)

    def neighbors(self, graph, node):
        if node not in graph:
            print("node {} not in graph".format(node))
            print("Graph nodes {}".format(graph.nodes()))
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
        Perform a random walk starting from the root nodes.

        Args:
            nodes: <list> The root nodes as a list of node IDs
            n: <int> Total number of random walks per root node
            length: <int> Maximum length of each random walk
            seed: <int> Random number generator seed; default is None

        Returns:
            <list> List of lists of nodes ids for each of the random walks

        """
        self._check_parameter_values(nodes=nodes, n=n, length=length, seed=seed)

        if seed:
            # seed the random number generator
            rs = random.Random(seed)
        else:
            # Restore the random state
            rs = self._random_state

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
                        rs.shuffle(neighbours)  # shuffles the list in place
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
            raise ValueError(
                "({}) A list of root node IDs was not provided.".format(
                    type(self).__name__
                )
            )
        if not is_real_iterable(nodes):
            raise ValueError("nodes parameter should be an iterable of node IDs.")
        if (
            len(nodes) == 0
        ):  # this is not an error but maybe a warning should be printed to inform the caller
            print(
                "WARNING: ({}) No root node IDs given. An empty list will be returned as a result.".format(
                    type(self).__name__
                )
            )

        if type(n) != int:
            raise ValueError(
                "({}) The number of walks per root node, n, should be integer type.".format(
                    type(self).__name__
                )
            )
        if n <= 0:
            raise ValueError(
                "({}) The number of walks per root node, n, should be a positive integer.".format(
                    type(self).__name__
                )
            )

        if type(length) != int:
            raise ValueError(
                "({}) The walk length, length, should be integer type.".format(
                    type(self).__name__
                )
            )
        if length <= 0:
            raise ValueError(
                "({}) The walk length, length, should be positive integer.".format(
                    type(self).__name__
                )
            )

        if seed is not None:
            if type(seed) != int:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be integer type or None.".format(
                        type(self).__name__
                    )
                )
            if seed < 0:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be positive integer or None.".format(
                        type(self).__name__
                    )
                )


def naive_weighted_choices(rs, weights):
    """
    Select an index at random, weighted by the iterator `weights` of
    arbitrary (non-negative) floats. That is, `x` will be returned
    with probability `weights[x]/sum(weights)`.

    For doing a single sample with arbitrary weights, this is much (5x
    or more) faster than numpy.random.choice, because the latter
    requires a lot of preprocessing (normalized probabilties), and
    does a lot of conversions/checks/preprocessing internally.
    """

    # divide the interval [0, sum(weights)) into len(weights)
    # subintervals [x_i, x_{i+1}), where the width x_{i+1} - x_i ==
    # weights[i]
    subinterval_ends = []
    running_total = 0
    for w in weights:
        assert w >= 0
        running_total += w
        subinterval_ends.append(running_total)

    # pick a place in the overall interval
    x = rs.random() * running_total

    # find the subinterval that contains the place, by looking for the
    # first subinterval where the end is (strictly) after it
    for idx, end in enumerate(subinterval_ends):
        if x < end:
            break

    return idx


class BiasedRandomWalk(GraphWalk):
    """
    Performs biased second order random walks (like those used in Node2Vec algorithm
    https://snap.stanford.edu/node2vec/) controlled by the values of two parameters p and q.
    """

    def __init__(self, graph, graph_schema=None, seed=None):
        # TODO: add doc string
        super().__init__(graph, graph_schema=None, seed=None)

    #        if (graph.weighted):
    #           print("its a weighted graph!")

    def run(
        self,
        nodes=None,
        n=None,
        p=1.0,
        q=1.0,
        length=None,
        seed=None,
        weighted=False,
        weight="weight",
    ):

        """
        Perform a random walk starting from the root nodes.

        Args:
            nodes: <list> The root nodes as a list of node IDs
            n: <int> Total number of random walks per root node
            p: <float> Defines probability, 1/p, of returning to source node
            q: <float> Defines probability, 1/q, for moving to a node away from the source node
            length: <int> Maximum length of each random walk
            seed: <int> Random number generator seed; default is None
            weighted: <False or True> Indicates whether the walk is unweighted or weighted
            weight: <string> Label of the edge.

        Returns:
            <list> List of lists of nodes ids for each of the random walks

        """
        self._check_parameter_values(
            nodes=nodes,
            n=n,
            p=p,
            q=q,
            length=length,
            seed=seed,
            weighted=weighted,
            weight=weight,
        )

        if seed:
            # seed a new random number generator
            rs = random.Random(seed)
        else:
            # Restore the random state
            rs = self._random_state

        if weighted:
            for node in self.graph.nodes():
                for neighbor in self.neighbors(self.graph, node):
                    edges = self.graph[node][neighbor]
                    wts = set()
                    for k, v in edges.items():
                        if (
                            v.get(weight) == None
                            or np.isnan(v.get(weight))
                            or v.get(weight) == np.inf
                        ):
                            raise ValueError(
                                "Missing edge weight between ({}) and ({}).  Cannot perform a weighted random walk on the graph.".format(
                                    node, neighbor
                                )
                            )
                        if not isinstance(v.get(weight), (int, float)):
                            raise ValueError(
                                "Edge weight between ({}) and ({}) is ({}). It should be a numberic value.".format(
                                    node, neighbor, v.get(weight)
                                )
                            )
                        if v.get(weight) < 0:  # check if edge has a negative weight
                            raise ValueError(
                                "An edges between ({}) and ({}) has negative weight of ({}). Cannot perform a meaningful random walk on the graph.".format(
                                    node, neighbor, v.get(weight)
                                )
                            )

                        wts.add(v.get(weight))
                    if (
                        len(wts) > 1
                    ):  # multigraph with different weights on edges between same pair of nodes
                        raise ValueError(
                            "({}) and ({}) have multiple edges with weights ({}). Ambigous to choose an edge for the random walk.".format(
                                node, neighbor, list(wts)
                            )
                        )

        ip = 1.0 / p
        iq = 1.0 / q

        walks = []
        for node in nodes:  # iterate over root nodes
            for walk_number in range(n):  # generate n walks per root node
                # the walk starts at the root
                walk = [node]

                neighbours = self.neighbors(self.graph, node)

                previous_node = node
                previous_node_neighbours = neighbours

                # calculate the appropriate unnormalized transition
                # probability, given the history of the walk
                def transition_probability(nn, current_node, weighted):

                    if weighted:
                        weight_cn = self.graph[current_node][nn][0].get(weight)
                        if nn == previous_node:  # d_tx = 0
                            return ip * weight_cn
                        elif nn in previous_node_neighbours:  # d_tx = 1
                            return 1.0 * weight_cn
                        else:  # d_tx = 2
                            return iq * weight_cn
                    else:
                        if nn == previous_node:  # d_tx = 0
                            return ip
                        elif nn in previous_node_neighbours:  # d_tx = 1
                            return 1.0
                        else:  # d_tx = 2
                            return iq

                if neighbours:
                    current_node = rs.choice(neighbours)
                    for _ in range(length - 1):
                        walk.append(current_node)
                        neighbours = self.neighbors(self.graph, current_node)

                        if not neighbours:
                            break

                        # select one of the neighbours using the
                        # appropriate transition probabilities
                        choice = naive_weighted_choices(
                            rs,
                            (
                                transition_probability(nn, current_node, weighted)
                                for nn in neighbours
                            ),
                        )

                        previous_node = current_node
                        previous_node_neighbours = neighbours
                        current_node = neighbours[choice]

                walks.append(walk)

        return walks

    def _check_parameter_values(self, nodes, n, p, q, length, seed, weighted, weight):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            nodes: <list> A list of root node ids such that from each node a uniform random walk of up to length l
            will be generated.
            n: <int> Number of walks per node id.
            p: <float>
            q: <float>
            length: <int> Maximum length of walk measured as the number of edges followed from root node.
            seed: <int> Random number generator seed.
            weighted: <False or True> Indicates whether the walk is unweighted or weighted.
            weight: <string> Label of the edge.

        """
        if nodes is None:
            raise ValueError(
                "({}) A list of root node IDs was not provided.".format(
                    type(self).__name__
                )
            )
        if not is_real_iterable(nodes):
            raise ValueError("nodes parameter should be an iterableof node IDs.")
        if (
            len(nodes) == 0
        ):  # this is not an error but maybe a warning should be printed to inform the caller
            print(
                "WARNING: ({}) No root node IDs given. An empty list will be returned as a result.".format(
                    type(self).__name__
                )
            )

        if type(n) != int:
            raise ValueError(
                "({}) The number of walks per root node, n, should be integer type.".format(
                    type(self).__name__
                )
            )

        if n <= 0:
            raise ValueError(
                "({}) The number of walks per root node, n, should be a positive integer.".format(
                    type(self).__name__
                )
            )

        if p <= 0.0:
            raise ValueError(
                "({}) Parameter p should be greater than 0.".format(type(self).__name__)
            )

        if q <= 0.0:
            raise ValueError(
                "({}) Parameter q should be greater than 0.".format(type(self).__name__)
            )

        if type(length) != int:
            raise ValueError(
                "({}) The walk length, length, should be integer type.".format(
                    type(self).__name__
                )
            )

        if length <= 0:
            raise ValueError(
                "({}) The walk length, length, should be positive integer.".format(
                    type(self).__name__
                )
            )

        if seed is not None:
            if seed < 0:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be positive integer or None.".format(
                        type(self).__name__
                    )
                )
            if type(seed) != int:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be integer type or None.".format(
                        type(self).__name__
                    )
                )

        if type(weighted) != bool:
            raise ValueError(
                "({}) The weighted indicator has to be either False (unweighted random walks) or True (weighted random walks).".format(
                    type(self).__name__
                )
            )

        if not weight.isalpha:
            raise ValueError(
                "({}) The label of the edge weight has to be string or none".format(
                    type(self).__name__
                )
            )


class UniformRandomMetaPathWalk(GraphWalk):
    """
    For heterogeneous graphs, it performs uniform random walks based on given metapaths.
    """

    def run(
        self,
        nodes=None,
        n=None,
        length=None,
        metapaths=None,
        node_type_attribute="label",
        seed=None,
    ):
        """
        Performs metapath-driven uniform random walks on heterogeneous graphs.

        Args:
            nodes: <list> The root nodes as a list of node IDs
            n: <int> Total number of random walks per root node
            length: <int> Maximum length of each random walk
            metapaths: <list> List of lists of node labels that specify a metapath schema, e.g.,
            [['Author', 'Paper', 'Author'], ['Author, 'Paper', 'Venue', 'Paper', 'Author']] specifies two metapath
            schemas of length 3 and 5 respectively.
            node_type_attribute: <str> The node attribute name that stores the node's type
            seed: <int> Random number generator seed; default is None

        Returns:
            <list> List of lists of nodes ids for each of the random walks generated
        """
        self._check_parameter_values(
            nodes=nodes,
            n=n,
            length=length,
            metapaths=metapaths,
            node_type_attribute=node_type_attribute,
            seed=seed,
        )

        if seed:
            # seed the random number generator
            rs = random.Random(seed)
        else:
            # Restore the random state
            rs = self._random_state

        walks = []

        for node in nodes:
            # retrieve node type
            label = self.graph.node[node][node_type_attribute]
            filtered_metapaths = [
                metapath
                for metapath in metapaths
                if len(metapath) > 0 and metapath[0] == label
            ]

            for metapath in filtered_metapaths:
                # augment metapath to be length long
                # if (
                #     len(metapath) == 1
                # ):  # special case for random walks like in a homogeneous graphs
                #     metapath = metapath * length
                # else:
                metapath = metapath[1:] * ((length // (len(metapath) - 1)) + 1)
                for _ in range(n):
                    walk = (
                        []
                    )  # holds the walk data for this walk; first node is the starting node
                    current_node = node
                    for d in range(length):
                        walk.append(current_node)
                        # d+1 can also be used to index metapath to retrieve the node type for the next step in the walk
                        neighbours = nx.neighbors(self.graph, node)
                        # filter these by node type
                        neighbours = [
                            node
                            for node in neighbours
                            if self.graph.node[node][node_type_attribute] == metapath[d]
                        ]
                        if len(neighbours) == 0:
                            # if no neighbours of the required type as dictated by the metapath exist, then stop.
                            break
                        # select one of the neighbours uniformly at random
                        current_node = rs.choice(
                            neighbours
                        )  # the next node in the walk

                    walks.append(walk)  # store the walk

        return walks

    def _check_parameter_values(
        self, nodes, n, length, metapaths, node_type_attribute, seed
    ):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            nodes: <list> The starting nodes as a list of node IDs.
            n: <int> Number of walks per node id.
            length: <int> Maximum length of of each random walk
            metapaths: <list> List of lists of node labels that specify a metapath schema, e.g.,
            [['Author', 'Paper', 'Author'], ['Author, 'Paper', 'Venue', 'Paper', 'Author']] specifies two metapath
            schemas of length 3 and 5 respectively.
            node_type_attribute: <str> The node attribute name that stores the node's type
            seed: <int> Random number generator seed

        """
        if nodes is None:
            raise ValueError(
                "({}) A list of starting node IDs was not provided (parameter nodes is None).".format(
                    type(self).__name__
                )
            )
        if not is_real_iterable(nodes):
            raise ValueError(
                "({}) The nodes parameter should be an iterable of node IDs.".format(
                    type(self).__name__
                )
            )
        if (
            len(nodes) == 0
        ):  # this is not an error but maybe a warning should be printed to inform the caller
            print(
                "WARNING: ({}) No starting node IDs given. An empty list will be returned as a result.".format(
                    type(self).__name__
                )
            )
        if n <= 0:
            raise ValueError(
                "({}) The number of walks per starting node, n, should be a positive integer.".format(
                    type(self).__name__
                )
            )
        if type(n) != int:
            raise ValueError(
                "({}) The number of walks per starting node, n, should be integer type.".format(
                    type(self).__name__
                )
            )

        if length <= 0:
            raise ValueError(
                "({}) The walk length parameter, length, should be positive integer.".format(
                    type(self).__name__
                )
            )
        if type(length) != int:
            raise ValueError(
                "({}) The walk length parameter, length, should be integer type.".format(
                    type(self).__name__
                )
            )

        if type(metapaths) != list:
            raise ValueError(
                "({}) The metapaths parameter must be a list of lists.".format(
                    type(self).__name__
                )
            )
        for metapath in metapaths:
            if type(metapath) != list:
                raise ValueError(
                    "({}) Each metapath must be list type of node labels".format(
                        type(self).__name__
                    )
                )
            if len(metapath) < 2:
                raise ValueError(
                    "({}) Each metapath must specify at least two node types".format(
                        type(self).__name__
                    )
                )

            for node_label in metapath:
                if type(node_label) != str:
                    raise ValueError(
                        "({}) Node labels in metapaths must be string type.".format(
                            type(self).__name__
                        )
                    )
            if metapath[0] != metapath[-1]:
                raise ValueError(
                    "({} The first and last node type in a metapath should be the same.".format(
                        type(self).__name__
                    )
                )

        if type(node_type_attribute) != str:
            raise ValueError(
                "({}) The parameter label should be string type not {} as given".format(
                    type(self).__name__, type(node_type_attribute).__name__
                )
            )

        if seed is not None:
            if seed < 0:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be positive integer or None.".format(
                        type(self).__name__
                    )
                )
            if type(seed) != int:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be integer type or None.".format(
                        type(self).__name__
                    )
                )


class DepthFirstWalk(GraphWalk):
    """
    Depth First Walk that generates all paths from a starting node to a given depth.
    It can be used to extract, in a memory efficient way, a sub-graph starting from a node and up to a given depth.
    """

    # TODO: Implement the run method
    pass


class BreadthFirstWalk(GraphWalk):
    """
    Breadth First Walk that generates all paths from a starting node to a given depth.
    It can be used to extract a sub-graph starting from a node and up to a given depth.
    """

    # TODO: Implement the run method
    pass


class SampledBreadthFirstWalk(GraphWalk):
    """
    Breadth First Walk that generates a sampled number of paths from a starting node.
    It can be used to extract a random sub-graph starting from a set of initial nodes.
    """

    def run(self, nodes=None, n=1, n_size=None, seed=None):
        """
        Performs a sampled breadth-first walk starting from the root nodes.

        Args:
            nodes:  <list> A list of root node ids such that from each node n BFWs will be generated up to the
            given depth d.
            n: <int> Number of walks per node id.
            n_size: <list> The number of neighbouring nodes to expand at each depth of the walk. Sampling of
            neighbours with replacement is always used regardless of the node degree and number of neighbours
            requested.
            seed: <int> Random number generator seed; default is None

        Returns:
            A list of lists such that each list element is a sequence of ids corresponding to a BFW.
        """
        self._check_parameter_values(nodes=nodes, n=n, n_size=n_size, seed=seed)

        walks = []
        d = len(n_size)  # depth of search

        if seed:
            # seed the random number generator
            rs = random.Random(seed)
        else:
            # Restore the random state
            rs = self._random_state

        for node in nodes:  # iterate over root nodes
            for _ in range(n):  # do n bounded breadth first walks from each root node
                q = list()  # the queue of neighbours
                walk = list()  # the list of nodes in the subgraph of node
                # extend() needs iterable as parameter; we use list of tuples (node id, depth)
                q.extend([(node, 0)])

                while len(q) > 0:
                    # remove the top element in the queue
                    # index 0 pop the item from the front of the list
                    frontier = q.pop(0)
                    depth = frontier[1] + 1  # the depth of the neighbouring nodes
                    walk.extend([frontier[0]])  # add to the walk

                    # consider the subgraph up to and including depth d from root node
                    if depth <= d:
                        neighbours = self.neighbors(self.graph, frontier[0])
                        if len(neighbours) == 0:
                            break
                        else:
                            # sample with replacement
                            neighbours = [
                                rs.choice(neighbours) for _ in range(n_size[depth - 1])
                            ]

                        # add them to the back of the queue
                        q.extend([(sampled_node, depth) for sampled_node in neighbours])

                # finished i-th walk from node so add it to the list of walks as a list
                walks.append(walk)

        return walks

    def _check_parameter_values(self, nodes, n, n_size, seed):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            nodes: <list> A list of root node ids such that from each node n BFWs will be generated up to the
            given depth d.
            n: <int> Number of walks per node id.
            n_size: <list> The number of neighbouring nodes to expand at each depth of the walk.
            seed: <int> Random number generator seed; default is None

        """
        if nodes is None:
            raise ValueError(
                "({}) A list of root node IDs was not provided (nodes parameter is None).".format(
                    type(self).__name__
                )
            )
        if not is_real_iterable(nodes):
            raise ValueError(
                "({}) The nodes parameter should be an iterable of node IDs.".format(
                    type(self).__name__
                )
            )
        if (
            len(nodes) == 0
        ):  # this is not an error but maybe a warning should be printed to inform the caller
            print(
                "WARNING: ({}) No root node IDs given. An empty list will be returned as a result.".format(
                    type(self).__name__
                )
            )

        if type(n) != int:
            raise ValueError(
                "({}) The number of walks per root node, n, should be integer type.".format(
                    type(self).__name__
                )
            )

        if n <= 0:
            raise ValueError(
                "({}) The number of walks per root node, n, should be a positive integer.".format(
                    type(self).__name__
                )
            )

        if n_size is None:
            raise ValueError(
                "({}) The neighbourhood size, n_size, must be a list of integers not None.".format(
                    type(self).__name__
                )
            )
        if type(n_size) != list:
            raise ValueError(
                "({}) The neighbourhood size, n_size, must be a list of integers.".format(
                    type(self).__name__
                )
            )

        if len(n_size) == 0:
            raise ValueError(
                "({}) The neighbourhood size, n_size, should not be empty list.".format(
                    type(self).__name__
                )
            )

        for d in n_size:
            if type(d) != int:
                raise ValueError(
                    "({}) The neighbourhood size, n_size, must be list of positive integers or 0.".format(
                        type(self).__name__
                    )
                )
            if d < 0:
                raise ValueError(
                    "({}) The neighbourhood size, n_size, must be list of positive integers or 0.".format(
                        type(self).__name__
                    )
                )

        if seed is not None:
            if type(seed) != int:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be integer type or None.".format(
                        type(self).__name__
                    )
                )
            if seed < 0:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be positive integer or None.".format(
                        type(self).__name__
                    )
                )


class SampledHeterogeneousBreadthFirstWalk(GraphWalk):
    """
    Breadth First Walk for heterogeneous graphs that generates a sampled number of paths from a starting node.
    It can be used to extract a random sub-graph starting from a set of initial nodes.
    """

    def run(self, nodes=None, n=1, n_size=None, seed=None):
        """
        Performs a sampled breadth-first walk starting from the root nodes.

        Args:
            nodes:  <list> A list of root node ids such that from each node n BFWs will be generated
                with the number of samples per hop specified in n_size.
            n: <int> Number of walks per node id.
            n_size: <list> The number of neighbouring nodes to expand at each depth of the walk. Sampling of
            neighbours with replacement is always used regardless of the node degree and number of neighbours
            requested.
            graph_schema: <GraphSchema> If None then the graph schema is extracted from self.graph
            seed: <int> Random number generator seed; default is None

        Returns:
            A list of lists such that each list element is a sequence of ids corresponding to a sampled Heterogeneous
            BFW.
        """
        self._check_parameter_values(
            nodes=nodes, n=n, n_size=n_size, graph_schema=self.graph_schema, seed=seed
        )

        walks = []
        d = len(n_size)  # depth of search

        if seed:
            # seed the random number generator
            rs = random.Random(seed)
        else:
            # Restore the random state
            rs = self._random_state

        for node in nodes:  # iterate over root nodes
            for _ in range(n):  # do n bounded breadth first walks from each root node
                q = list()  # the queue of neighbours
                walk = list()  # the list of nodes in the subgraph of node

                # Start the walk by adding the head node, and node type to the frontier list q
                node_type = self.graph_schema.get_node_type(node)
                q.extend([(node, node_type, 0)])

                # add the root node to the walks
                walk.append([node])
                while len(q) > 0:
                    # remove the top element in the queue and pop the item from the front of the list
                    frontier = q.pop(0)
                    current_node, current_node_type, depth = frontier
                    depth = depth + 1  # the depth of the neighbouring nodes

                    # consider the subgraph up to and including depth d from root node
                    if depth <= d:
                        # Find edge types for current node type
                        current_edge_types = self.graph_schema.schema[current_node_type]

                        # Create samples of neigbhours for all edge types
                        for et in current_edge_types:
                            neigh_et = self.adj[et][current_node]

                            # If there are no neighbours of this type then we return None
                            # in the place of the nodes that would have been sampled
                            # YT update: with the new way to get neigh_et from self.adj[et][current_node], len(neigh_et) is always > 0.
                            # In case of no neighbours of the current node for et, neigh_et == [None],
                            # and samples automatically becomes [None]*n_size[depth-1]
                            if len(neigh_et) > 0:
                                samples = [
                                    rs.choice(neigh_et)
                                    for _ in range(n_size[depth - 1])
                                ]
                                # Choices limits us to Python 3.6+
                                # samples = random.choices(neigh_et, k=n_size[depth - 1])
                            else:  # this doesn't happen anymore, see the comment above
                                samples = [None] * n_size[depth - 1]

                            walk.append(samples)
                            q.extend(
                                [
                                    (sampled_node, et.n2, depth)
                                    for sampled_node in samples
                                ]
                            )

                # finished i-th walk from node so add it to the list of walks as a list
                walks.append(walk)

        return walks

    def _check_parameter_values(self, nodes, n, n_size, graph_schema, seed):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            nodes: <list> A list of root node ids such that from each node n BFWs will be generated up to the
            given depth d.
            n: <int> Number of walks per node id.
            n_size: <list> The number of neighbouring nodes to expand at each depth of the walk.
            graph_schema: <GraphSchema> None or a stellargraph graph schema object
            seed: <int> Random number generator seed; default is None

        """
        if nodes is None:
            raise ValueError(
                "({}) A list of root node IDs was not provided (nodes parameter is None).".format(
                    type(self).__name__
                )
            )
        if not is_real_iterable(nodes):
            raise ValueError(
                "({}) The nodes parameter should be an iterable of node IDs.".format(
                    type(self).__name__
                )
            )
        if (
            len(nodes) == 0
        ):  # this is not an error but maybe a warning should be printed to inform the caller
            print(
                "WARNING: ({}) No root node IDs given. An empty list will be returned as a result.".format(
                    type(self).__name__
                )
            )

        if type(n) != int:
            raise ValueError(
                "({}) The number of walks per root node, n, should be integer type.".format(
                    type(self).__name__
                )
            )

        if n <= 0:
            raise ValueError(
                "({}) The number of walks per root node, n, should be a positive integer.".format(
                    type(self).__name__
                )
            )

        if n_size is None:
            raise ValueError(
                "({}) The neighbourhood size, n_size, must be a list of integers not None.".format(
                    type(self).__name__
                )
            )
        if type(n_size) != list:
            raise ValueError(
                "({}) The neighbourhood size, n_size, must be a list of integers.".format(
                    type(self).__name__
                )
            )

        if len(n_size) == 0:
            raise ValueError(
                "({}) The neighbourhood size, n_size, should not be empty list.".format(
                    type(self).__name__
                )
            )

        for d in n_size:
            if type(d) != int:
                raise ValueError(
                    "({}) The neighbourhood size, n_size, must be list of integers.".format(
                        type(self).__name__
                    )
                )
            if d < 0:
                raise ValueError(
                    "({}) n_sie should be positive integer or 0.".format(
                        type(self).__name__
                    )
                )

        if graph_schema is not None and type(graph_schema) is not GraphSchema:
            raise ValueError(
                "({}) The parameter graph_schema should be either None or of type GraphSchema.".format(
                    type(self).__name__
                )
            )

        if seed is not None:
            if type(seed) != int:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be integer type or None.".format(
                        type(self).__name__
                    )
                )
            if seed < 0:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be positive integer or None.".format(
                        type(self).__name__
                    )
                )
