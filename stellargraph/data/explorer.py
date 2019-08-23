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

__all__ = [
    "UniformRandomWalk",
    "BiasedRandomWalk",
    "UniformRandomMetaPathWalk",
    "SampledBreadthFirstWalk",
    "SampledHeterogeneousBreadthFirstWalk",
]


import networkx as nx
import numpy as np
import random
from collections import defaultdict, deque

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
        self._check_seed(seed)
        self._random_state = random.Random(seed)

        # Initialize a numpy random state (for numpy random methods)
        self._np_random_state = np.random.RandomState(seed=seed)

        # We require a StellarGraph for this
        if not isinstance(graph, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph or StellarDiGraph.")

        if not graph_schema:
            self.graph_schema = self.graph.create_graph_schema(create_type_maps=True)
        else:
            self.graph_schema = graph_schema

        if type(self.graph_schema) is not GraphSchema:
            self._raise_error(
                "The parameter graph_schema should be either None or of type GraphSchema."
            )

    def get_adjacency_types(self):
        # Allow additional info for heterogeneous graphs.
        adj = getattr(self, "adj_types", None)
        if not adj:
            # Create a dict of adjacency lists per edge type, for faster neighbour sampling from graph in SampledHeteroBFS:
            # TODO: this could be better placed inside StellarGraph class
            edge_types = self.graph_schema.edge_types
            adj = {et: defaultdict(lambda: [None]) for et in edge_types}

            for n1, nbrdict in self.graph.adjacency():
                for et in edge_types:
                    neigh_et = [
                        n2
                        for n2, nkeys in nbrdict.items()
                        for k in nkeys
                        if self.graph_schema.is_of_edge_type((n1, n2, k), et)
                    ]
                    # Create adjacency list in lexographical order
                    # Otherwise sampling methods will not be deterministic
                    # even when the seed is set.
                    adj[et][n1] = sorted(neigh_et, key=str)
            self.adj_types = adj
        return adj

    def _check_seed(self, seed):
        if seed is not None:
            if type(seed) != int:
                self._raise_error(
                    "The random number generator seed value, seed, should be integer type or None."
                )
            if seed < 0:
                self._raise_error(
                    "The random number generator seed value, seed, should be non-negative integer or None."
                )

    def _get_random_state(self, seed):
        """
        Args:
            seed: The optional seed value for a given run.

        Returns:
            The random state as determined by the seed.
        """
        if seed is None:
            # Restore the random state
            return self._random_state
        # seed the random number generator
        return random.Random(seed)

    def neighbors(self, node):
        if node not in self.graph:
            self._raise_error("node {} not in graph".format(node))
        return list(nx.neighbors(self.graph, node))

    def run(self, **kwargs):
        """
        To be overridden by subclasses. It is the main entry point for performing random walks on the given
        graph.
        It should return the sequences of nodes in each random walk.

        Args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError

    def _raise_error(self, msg):
        raise ValueError("({}) {}".format(type(self).__name__, msg))

    def _check_common_parameters(self, nodes, n, length, seed):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            nodes: <list> A list of root node ids from which to commence the random walks.
            n: <int> Number of walks per node id.
            length: <int> Maximum length of each walk.
            seed: <int> Random number generator seed.
        """
        self._check_nodes(nodes)
        self._check_repetitions(n)
        self._check_length(length)
        self._check_seed(seed)

    def _check_nodes(self, nodes):
        if nodes is None:
            self._raise_error("A list of root node IDs was not provided.")
        if not is_real_iterable(nodes):
            self._raise_error("Nodes parameter should be an iterable of node IDs.")
        if (
            len(nodes) == 0
        ):  # this is not an error but maybe a warning should be printed to inform the caller
            print(
                "({}) WARNING: No root node IDs given. An empty list will be returned as a result.".format(
                    type(self).__name__
                )
            )

    def _check_repetitions(self, n):
        if type(n) != int:
            self._raise_error(
                "The number of walks per root node, n, should be integer type."
            )
        if n <= 0:
            self._raise_error(
                "The number of walks per root node, n, should be a positive integer."
            )

    def _check_length(self, length):
        if type(length) != int:
            self._raise_error("The walk length, length, should be integer type.")
        if length <= 0:
            # Technically, length 0 should be okay, but by consensus is invalid.
            self._raise_error("The walk length, length, should be a positive integer.")

    # For neighbourhood sampling
    def _check_sizes(self, n_size):
        err_msg = "The neighbourhood size must be a list of non-negative integers."
        if not isinstance(n_size, list):
            self._raise_error(err_msg)
        if len(n_size) == 0:
            # Technically, length 0 should be okay, but by consensus it is invalid.
            self._raise_error("The neighbourhood size list should not be empty.")
        for d in n_size:
            if type(d) != int or d < 0:
                self._raise_error(err_msg)


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
        self._check_common_parameters(nodes, n, length, seed)
        rs = self._get_random_state(seed)

        walks = []
        for node in nodes:  # iterate over root nodes
            for walk_number in range(n):  # generate n walks per root node
                walk = list()
                current_node = node
                for _ in range(length):
                    walk.extend([current_node])
                    neighbours = self.neighbors(current_node)
                    if (
                        len(neighbours) == 0
                    ):  # for whatever reason this node has no neighbours so stop
                        break
                    else:
                        rs.shuffle(neighbours)  # shuffles the list in place
                        current_node = neighbours[0]  # select the first node to follow

                walks.append(walk)

        return walks


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
        if w < 0:
            raise ValueError("Detected negative weight: {}".format(w))
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

    def run(
        self,
        nodes=None,
        n=None,
        p=1.0,
        q=1.0,
        length=None,
        seed=None,
        weighted=False,
        edge_weight_label="weight",
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
            edge_weight_label: <string> Label of the edge weight property.

        Returns:
            <list> List of lists of nodes ids for each of the random walks

        """
        self._check_common_parameters(nodes, n, length, seed)
        self._check_weights(p, q, weighted, edge_weight_label)
        rs = self._get_random_state(seed)

        if weighted:
            # Check that all edge weights are greater than or equal to 0.
            # Also, if the given graph is a MultiGraph, then check that there are no two edges between
            # the same two nodes with different weights.
            for node in self.graph.nodes():
                for neighbor in self.neighbors(node):

                    wts = set()
                    for k, v in self.graph[node][neighbor].items():
                        weight = v.get(edge_weight_label)
                        if weight is None or np.isnan(weight) or weight == np.inf:
                            self._raise_error(
                                "Missing or invalid edge weight ({}) between ({}) and ({}).".format(
                                    weight, node, neighbor
                                )
                            )
                        if not isinstance(weight, (int, float)):
                            self._raise_error(
                                "Edge weight between nodes ({}) and ({}) is not numeric ({}).".format(
                                    node, neighbor, weight
                                )
                            )
                        if weight < 0:  # check if edge has a negative weight
                            self._raise_error(
                                "An edge weight between nodes ({}) and ({}) is negative ({}).".format(
                                    node, neighbor, weight
                                )
                            )

                        wts.add(weight)
                    if (
                        len(wts) > 1
                    ):  # multigraph with different weights on edges between same pair of nodes
                        self._raise_error(
                            "({}) and ({}) have multiple edges with weights ({}). Ambiguous to choose an edge for the random walk.".format(
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

                neighbours = self.neighbors(node)

                previous_node = node
                previous_node_neighbours = neighbours

                # calculate the appropriate unnormalised transition
                # probability, given the history of the walk
                def transition_probability(
                    nn, current_node, weighted, edge_weight_label
                ):

                    if weighted:
                        weight_cn = self.graph[current_node][nn][0].get(
                            edge_weight_label
                        )
                    else:
                        weight_cn = 1.0

                    if nn == previous_node:  # d_tx = 0
                        return ip * weight_cn
                    elif nn in previous_node_neighbours:  # d_tx = 1
                        return 1.0 * weight_cn
                    else:  # d_tx = 2
                        return iq * weight_cn

                if neighbours:
                    current_node = rs.choice(neighbours)
                    for _ in range(length - 1):
                        walk.append(current_node)
                        neighbours = self.neighbors(current_node)

                        if not neighbours:
                            break

                        # select one of the neighbours using the
                        # appropriate transition probabilities
                        choice = naive_weighted_choices(
                            rs,
                            (
                                transition_probability(
                                    nn, current_node, weighted, edge_weight_label
                                )
                                for nn in neighbours
                            ),
                        )

                        previous_node = current_node
                        previous_node_neighbours = neighbours
                        current_node = neighbours[choice]

                walks.append(walk)

        return walks

    def _check_weights(self, p, q, weighted, edge_weight_label):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            p: <float> The backward walk 'penalty' factor.
            q: <float> The forward walk 'penalty' factor.
            weighted: <False or True> Indicates whether the walk is unweighted or weighted.
            edge_weight_label: <string> Label of the edge weight property.
       """
        if p <= 0.0:
            self._raise_error("Parameter p should be greater than 0.")

        if q <= 0.0:
            self._raise_error("Parameter q should be greater than 0.")

        if type(weighted) != bool:
            self._raise_error(
                "Parameter weighted has to be either False (unweighted random walks) or True (weighted random walks)."
            )

        if not isinstance(edge_weight_label, str):
            self._raise_error("The edge weight property label has to be of type string")


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
        self._check_common_parameters(nodes, n, length, seed)
        self._check_metapath_values(metapaths, node_type_attribute)
        rs = self._get_random_state(seed)

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
                        neighbours = self.neighbors(current_node)
                        # filter these by node type
                        neighbours = [
                            n_node
                            for n_node in neighbours
                            if self.graph.node[n_node][node_type_attribute]
                            == metapath[d]
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

    def _check_metapath_values(self, metapaths, node_type_attribute):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            metapaths: <list> List of lists of node labels that specify a metapath schema, e.g.,
                [['Author', 'Paper', 'Author'], ['Author, 'Paper', 'Venue', 'Paper', 'Author']] specifies two metapath
                schemas of length 3 and 5 respectively.
            node_type_attribute: <str> The node attribute name that stores the node's type
        """
        if type(metapaths) != list:
            self._raise_error("The metapaths parameter must be a list of lists.")
        for metapath in metapaths:
            if type(metapath) != list:
                self._raise_error("Each metapath must be list type of node labels")
            if len(metapath) < 2:
                self._raise_error("Each metapath must specify at least two node types")

            for node_label in metapath:
                if type(node_label) != str:
                    self._raise_error("Node labels in metapaths must be string type.")
            if metapath[0] != metapath[-1]:
                self._raise_error(
                    "The first and last node type in a metapath should be the same."
                )

        if type(node_type_attribute) != str:
            self._raise_error(
                "The parameter label should be string type not {} as given".format(
                    type(node_type_attribute).__name__
                )
            )


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
        self._check_sizes(n_size)
        self._check_common_parameters(nodes, n, len(n_size), seed)
        rs = self._get_random_state(seed)

        walks = []
        max_hops = len(n_size)  # depth of search

        for node in nodes:  # iterate over root nodes
            for _ in range(n):  # do n bounded breadth first walks from each root node
                q = deque()  # the queue of neighbours
                walk = list()  # the list of nodes in the subgraph of node
                # extend() needs iterable as parameter; we use list of tuples (node id, depth)
                q.append((node, 0))

                while len(q) > 0:
                    # remove the top element in the queue
                    # index 0 pop the item from the front of the list
                    cur_node, cur_depth = q.popleft()
                    depth = cur_depth + 1  # the depth of the neighbouring nodes
                    walk.append(cur_node)  # add to the walk

                    # consider the subgraph up to and including max_hops from root node
                    if depth > max_hops:
                        continue
                    neighbours = (
                        self.neighbors(cur_node) if cur_node is not None else []
                    )
                    if len(neighbours) == 0:
                        # Either node is unconnected or is in directed graph with no out-nodes.
                        neighbours = [None] * n_size[cur_depth]
                    else:
                        # sample with replacement
                        neighbours = [
                            rs.choice(neighbours) for _ in range(n_size[cur_depth])
                        ]

                    # add them to the back of the queue
                    q.extend((sampled_node, depth) for sampled_node in neighbours)

                # finished i-th walk from node so add it to the list of walks as a list
                walks.append(walk)

        return walks


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
        self._check_sizes(n_size)
        self._check_common_parameters(nodes, n, len(n_size), seed)
        rs = self._get_random_state(seed)

        adj = self.get_adjacency_types()

        walks = []
        d = len(n_size)  # depth of search

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
                            neigh_et = adj[et][current_node]

                            # If there are no neighbours of this type then we return None
                            # in the place of the nodes that would have been sampled
                            # YT update: with the new way to get neigh_et from adj[et][current_node], len(neigh_et) is always > 0.
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


class DirectedBreadthFirstNeighbours(GraphWalk):
    """
    Breadth First sampler that generates the composite of a number of sampled paths from a starting node.
    It can be used to extract a random sub-graph starting from a set of initial nodes.
    """

    def __init__(self, graph, graph_schema=None, seed=None):
        super().__init__(graph, graph_schema, seed)
        if not graph.is_directed():
            self._raise_error("Graph must be directed")

    def run(self, nodes=None, n=1, in_size=None, out_size=None, seed=None):
        """
        Performs a sampled breadth-first walk starting from the root nodes.

        Args:
            nodes:  <list> A list of root node ids such that from each node n BFWs will be generated up to the
            given depth d.
            n: <int> Number of walks per node id.
            in_size: <list> The number of in-directed nodes to sample with replacement at each depth of the walk.
            out_size: <list> The number of out-directed nodes to sample with replacement at each depth of the walk.
            seed: <int> Random number generator seed; default is None


        Returns:
            A list of multi-hop neighbourhood samples. Each sample expresses multiple undirected walks, but the in-node
            neighbours and out-node neighbours are sampled separately. Each sample has the format:

                [[node]
                 [in_1...in_n]  [out_1...out_m]
                 [in_1.in_1...in_n.in_p] [in_1.out_1...in_n.out_q]
                    [out_1.in_1...out_m.in_p] [out_1.out_1...out_m.out_q]
                 [in_1.in_1.in_1...in_n.in_p.in_r] [in_1.in_1.out_1...in_n.in_p.out_s] ...
                 ...]

            where a single, undirected walk might be, for example:

                [node out_i  out_i.in_j  out_i.in_j.in_k ...]
        """
        self._check_neighbourhood_sizes(in_size, out_size)
        self._check_common_parameters(nodes, n, len(in_size), seed)
        rs = self._get_random_state(seed)

        max_hops = len(in_size)
        # A binary tree is a graph of nodes; however, we wish to avoid overusing the term 'node'.
        # Consider that each binary tree node carries some information.
        # We uniquely and deterministically number every node in the tree, so we
        # can represent the information stored in the tree via a flattened list of 'slots'.
        # Each slot (and corresponding binary tree node) now has a unique index in the flattened list.
        max_slots = 2 ** (max_hops + 1) - 1

        samples = []

        for node in nodes:  # iterate over root nodes
            for _ in range(n):  # do n bounded breadth first walks from each root node
                q = list()  # the queue of neighbours
                # the list of sampled node-lists:
                sample = [[] for _ in range(max_slots)]
                # Add node to queue as (node, depth, slot)
                q.append((node, 0, 0))

                while len(q) > 0:
                    # remove the top element in the queue
                    # index 0 pop the item from the front of the list
                    cur_node, cur_depth, cur_slot = q.pop(0)
                    sample[cur_slot].append(cur_node)  # add to the walk
                    depth = cur_depth + 1  # the depth of the neighbouring nodes

                    # consider the subgraph up to and including max_hops from root node
                    if depth > max_hops:
                        continue
                    # get in-nodes
                    neighbours = self._sample_neighbours(
                        rs, cur_node, 0, in_size[cur_depth]
                    )
                    # add them to the back of the queue
                    slot = 2 * cur_slot + 1
                    q.extend(
                        [(sampled_node, depth, slot) for sampled_node in neighbours]
                    )
                    # get out-nodes
                    neighbours = self._sample_neighbours(
                        rs, cur_node, 1, out_size[cur_depth]
                    )
                    # add them to the back of the queue
                    slot = slot + 1
                    q.extend(
                        [(sampled_node, depth, slot) for sampled_node in neighbours]
                    )

                # finished multi-hop neighbourhood sampling
                samples.append(sample)

        return samples

    def _sample_neighbours(self, rs, node, idx, size):
        """
        Samples (with replacement) the specified number of nodes
        from the directed neighbourhood of the given starting node.
        If the neighbourhood is empty, then the result will contain
        only None values.
        Args:
            rs: The random state used for sampling.
            node: The starting node.
            idx: <int> The index specifying the direction of the
                neighbourhood to be sampled: 0 => in-nodes;
                1 => out-nodes.
            size: <int> The number of nodes to sample.
        Returns:
            The fixed-length list of neighbouring nodes (or None values
            if the neighbourhood is empty).
        """
        if node is None:
            # Non-node, e.g. previously sampled from empty neighbourhood
            return [None] * size
        fn = self.graph.in_edges if idx == 0 else self.graph.out_edges
        neighbours = [n[idx] for n in list(fn(node))]
        if len(neighbours) == 0:
            # Sampling from empty neighbourhood
            return [None] * size
        # Sample with replacement
        return [rs.choice(neighbours) for _ in range(size)]

    def _check_neighbourhood_sizes(self, in_size, out_size):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            nodes: <list> A list of root node ids such that from each node n BFWs will be generated up to the
            given depth d.
            n_size: <list> The number of neighbouring nodes to expand at each depth of the walk.
            seed: <int> Random number generator seed; default is None
        """
        self._check_sizes(in_size)
        self._check_sizes(out_size)
        if len(in_size) != len(out_size):
            self._raise_error(
                "The number of hops for the in and out neighbourhoods must be the same."
            )
