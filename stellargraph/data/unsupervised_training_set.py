#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:43:04 2019

@author: hab031
"""


import networkx as nx
import numpy as np
import pandas as pd
import random
import os
from collections import defaultdict

from stellargraph.core.schema import GraphSchema
from stellargraph.core.graph import StellarGraphBase
from stellargraph.core.utils import is_real_iterable
import stellargraph as sg


# Base class for BiasedRandomWalk class. Copied from explorer.py. Did not modify.


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


## copied this method from explorer.py. No modification done.


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


# Modiefied the run method in the following way. For each walk, (target, context) +ive and -ive samples are accumulated after each walk.
# If a batch size for samples is specified, the method returns the (target,context) pairs of the batch size. In the next call of the run method, the next batch is returned and so on.
# If no batch size is specified, all the (target,context) pairs  generated for all the walks are returned.


class BiasedRandomWalk(GraphWalk):
    """
    Performs biased second order random walks (like those used in Node2Vec algorithm
    https://snap.stanford.edu/node2vec/) controlled by the values of two parameters p and q.
    """

    # calculate the appropriate unnormalised transition
    # probability, given the history of the walk
    def transition_probability(
        self,
        nn,
        current_node,
        previous_node,
        previous_node_neighbours,
        ip,
        iq,
        weighted,
        edge_weight_label,
    ):

        if weighted:
            weight_cn = self.graph[current_node][nn][0].get(edge_weight_label)
        else:
            weight_cn = 1.0

            if nn == previous_node:  # d_tx = 0
                return ip * weight_cn
            elif nn in previous_node_neighbours:  # d_tx = 1
                return 1.0 * weight_cn
            else:  # d_tx = 2
                return iq * weight_cn

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
        sample_size=None,
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
            edge_weight_label: <string> Label of the edge weight property
            sample_size: <int> the number of +ive and -ive training (target, context) pairs to return


        Returns:
            train_edge_ids, train_edge_labels

        """
        self._check_parameter_values(
            nodes=nodes,
            n=n,
            p=p,
            q=q,
            length=length,
            seed=seed,
            weighted=weighted,
            edge_weight_label=edge_weight_label,
            sample_size=sample_size,
        )

        if seed:
            # seed a new random number generator
            rs = random.Random(seed)
        else:
            # Restore the random state
            rs = self._random_state

        if weighted:
            # Check that all edge weights are greater than or equal to 0.
            # Also, check that if the given graph is a MultiGraph, then check that there are no two edges between
            # the same two nodes with different weights.
            for node in self.graph.nodes():
                for neighbor in self.neighbors(self.graph, node):

                    wts = set()
                    for k, v in self.graph[node][neighbor].items():
                        if (
                            v.get(edge_weight_label) == None
                            or np.isnan(v.get(edge_weight_label))
                            or v.get(edge_weight_label) == np.inf
                        ):
                            raise ValueError(
                                "Missing edge weight between ({}) and ({}).".format(
                                    node, neighbor
                                )
                            )
                        if not isinstance(v.get(edge_weight_label), (int, float)):
                            raise ValueError(
                                "Edge weight between nodes ({}) and ({}) is not numeric ({}).".format(
                                    node, neighbor, v.get(edge_weight_label)
                                )
                            )
                        if (
                            v.get(edge_weight_label) < 0
                        ):  # check if edge has a negative weight
                            raise ValueError(
                                "An edge weight between nodes ({}) and ({}) is negative ({}).".format(
                                    node, neighbor, v.get(edge_weight_label)
                                )
                            )

                        wts.add(v.get(edge_weight_label))
                    if (
                        len(wts) > 1
                    ):  # multigraph with different weights on edges between same pair of nodes
                        raise ValueError(
                            "({}) and ({}) have multiple edges with weights ({}). Ambiguous to choose an edge for the random walk.".format(
                                node, neighbor, list(wts)
                            )
                        )

        positive_pairs = list()
        negative_pairs = list()

        positive_samples_counter = 0
        negative_samples_counter = 0

        all_nodes = list(self.graph.nodes())

        # Use the sampling distribution as per node2vec
        degrees = self.graph.degree()
        sampling_distribution = [degrees[n] ** 0.75 for n in all_nodes]

        ip = 1.0 / p
        iq = 1.0 / q

        print("walking...")
        walks = []
        for node in nodes:  # iterate over root nodes
            for walk_number in range(
                n
            ):  # generate n walks per root node. The walk starts at the root.
                walk = [node]

                neighbours = self.neighbors(self.graph, node)

                previous_node = node
                previous_node_neighbours = neighbours

                if neighbours:
                    current_node = rs.choice(neighbours)
                    for _ in range(length - 1):
                        walk.append(current_node)
                        neighbours = self.neighbors(self.graph, current_node)

                        if not neighbours:
                            break

                        # select one of the neighbours using the appropriate transition probabilities
                        choice = naive_weighted_choices(
                            rs,
                            (
                                self.transition_probability(
                                    nn,
                                    current_node,
                                    previous_node,
                                    previous_node_neighbours,
                                    ip,
                                    iq,
                                    weighted,
                                    edge_weight_label,
                                )
                                for nn in neighbours
                            ),
                        )

                        previous_node = current_node
                        previous_node_neighbours = neighbours
                        current_node = neighbours[choice]

                print(walk)
                if len(walk) > 1:
                    target = walk[0]
                    context_window = walk[1:]

                    for context in context_window:
                        # Don't add self pairs
                        if context != target:
                            positive_pairs.append((target, context))
                            positive_samples_counter += 1
                            # For each positive sample, add a negative sample.
                            # Negative samples are contexts not in the current walk with respect to the current target(start node of the walk).
                            while negative_samples_counter < positive_samples_counter:
                                random_sample = random.choices(
                                    all_nodes, weights=sampling_distribution
                                )
                                if not random_sample in context_window:
                                    negative_pairs.append((target, *random_sample))
                                    negative_samples_counter = (
                                        negative_samples_counter + 1
                                    )

                        # If the batch_size number of samples are accumulated, yield.
                        if positive_samples_counter == sample_size:

                            all_pairs = positive_pairs + negative_pairs
                            all_targets = [1] * len(positive_pairs) + [0] * len(
                                negative_pairs
                            )
                            edge_ids_labels = list(zip(all_pairs, all_targets))
                            random.shuffle(edge_ids_labels)

                            print(
                                "Sampled {} positive edges".format(len(positive_pairs))
                            )
                            print(
                                "Sampled {} negative edges".format(len(negative_pairs))
                            )
                            positive_pairs.clear()
                            negative_pairs.clear()
                            positive_samples_counter = 0
                            negative_samples_counter = 0

                            yield edge_ids_labels

                walks.append(walk)
        # Return samples when either all the samples are to be pregenerated, i.e. batch_size isn't specified or this is the last batch and may not have batch_size number of samples left.
        all_pairs = positive_pairs + negative_pairs
        all_targets = [1] * len(positive_pairs) + [0] * len(negative_pairs)
        edge_ids_labels = list(zip(all_pairs, all_targets))
        random.shuffle(edge_ids_labels)
        print("all walks done!")
        yield edge_ids_labels

    def _check_parameter_values(
        self, nodes, n, p, q, length, seed, weighted, edge_weight_label, sample_size
    ):
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
            edge_weight_label: <string> Label of the edge weight property.
            sample_size: <int> the number of +ive and -ive training (target, context) pairs to return.

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
                "({}) Parameter weighted has to be either False (unweighted random walks) or True (weighted random walks).".format(
                    type(self).__name__
                )
            )

        if not isinstance(edge_weight_label, str):
            raise ValueError(
                "({}) The edge weight property label has to be of type string".format(
                    type(self).__name__
                )
            )

        if sample_size is not None:
            if sample_size < 0:
                raise ValueError(
                    "({}) The training sample size, sample_size, should be positive integer or None.".format(
                        type(self).__name__
                    )
                )
            if type(sample_size) != int:
                raise ValueError(
                    "({}) The training sample size, sample_size, should be positive integer or None.".format(
                        type(self).__name__
                    )
                )


##############################################################################


def main():
    print("Testing unsupervised GraphSAGE")

    # read/create a StellarGraph. using cora dataset for testing

    data_dir = "~/data/cora"

    edgelist = pd.read_table(
        os.path.join(data_dir, "cora.cites"), header=None, names=["source", "target"]
    )
    edgelist["label"] = "cites"  # set the edge type

    Gnx = nx.from_pandas_edgelist(edgelist, edge_attr="label")

    nx.set_node_attributes(Gnx, "paper", "label")

    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    column_names = feature_names + ["subject"]
    node_data = pd.read_table(
        os.path.join(data_dir, "cora.content"), header=None, names=column_names
    )

    node_features = node_data[feature_names]

    G = sg.StellarGraph(Gnx, node_features=node_features)

    rw = BiasedRandomWalk(G)

    walks = rw.run(
        nodes=G.nodes(),  # root nodes
        length=5,  # maximum length of a random walk
        n=1,  # number of random walks per root node
        p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=False,
        seed=42,
        sample_size=500,
    )

    training_samples = next(walks)
    edge_ids_train, edge_labels_train = [
        [z[i] for z in training_samples] for i in (0, 1)
    ]

    ####### The GraphSAGELinkGenerator
    num_samples = [10, 5]
    batch_size = 50
    train_gen = GraphSAGELinkGenerator(G, batch_size, num_samples).flow(
        edge_ids_train, edge_labels_train
    )


#  train_gen = GraphSAGELinkGenerator(G, batch_size, num_samples).flow(ut)
#  ut.next_batch(G)

#  train_gen_2 = GraphSAGELinkGenerator(G, batch_size, num_samples).flow(ut)


if __name__ == "__main__":
    main()
