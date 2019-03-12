#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["UnsupervisedSampler"]


import random

from stellargraph.core.utils import is_real_iterable
from stellargraph.core.graph import StellarGraphBase
from stellargraph.data.explorer import UniformRandomWalk


class UnsupervisedSampler:
    """
        The UnsupervisedSampler is responsible for sampling walks in the given graph
        and returning positive and negative samples w.r.t. those walks, on demand.

        The positive samples are all the (target, context) pairs from the walks and the negative
        samples are contexts generated for each target based on a sampling distribtution.

        Currently only uniform random walks are performed, other walk strategies (such as
        second order walks) will be enabled in the future.

        Args:
            G (StellarGraph): A stellargraph with features.
            nodes (optional, iterable) The root nodes from which individual walks start.
                If not provided, all nodes in the graph are used.
            length (int): An integer giving the length of the walks. Length must be at least 2.
            number_of_walks (int): Number of walks from each root node.
    """

    def __init__(self, G, nodes=None, length=2, number_of_walks=1, seed=None):
        if not isinstance(G, StellarGraphBase):
            raise ValueError(
                "({}) Graph must be a StellarGraph object.".format(type(self).__name__)
            )
        else:
            self.graph = G

        # Instantiate the walker class used to generate random walks in the graph
        self.walker = UniformRandomWalk(G, seed=seed)

        # This code will enable alternative walker classes
        # TODO: Enable this code, but figure out how to pass required options to run
        # if walker is not None:
        #     if not isinstance(
        #         walker, UniformRandomWalk
        #     ):  # only work with Uniform Random Walker at the moment
        #         raise TypeError(
        #             "({}) Only Uniform Random Walks are possible".format(
        #                 type(self).__name__
        #             )
        #         )
        #     else:
        #         self.walker = walker(G, seed=seed)
        # else:
        #         self.walker = UniformRandomWalk(G, seed=seed)

        # Define the root nodes for the walks
        # if no root nodes are provided for sampling defaulting to using all nodes as root nodes.
        if nodes is None:
            self.nodes = list(G.nodes())
        elif is_real_iterable(nodes):  # check whether the nodes provided are valid.
            self.nodes = list(nodes)
        else:
            raise ValueError("nodes parameter should be an iterableof node IDs.")

        # Require walks of at lease length two because to create a sample pair we need at least two nodes.
        if length < 2:
            raise ValueError(
                "({}) For generating (target,context) samples, walk length has to be at least 2".format(
                    type(self).__name__
                )
            )
        else:
            self.length = length

        if number_of_walks < 1:
            raise ValueError(
                "({}) At least 1 walk from each head node has to be done".format(
                    type(self).__name__
                )
            )
        else:
            self.number_of_walks = number_of_walks

        # Setup an interal random state with the given seed
        self.random = random.Random(seed)

    def generator(self, batch_size):

        """
        This method yields a batch_size number of positive and negative samples from the graph.
        This method generates one walk at a time of a given length from each root node and returns
        the positive pairs from the walks and the same number of negative pairs from a global
        node sampling distribution.

        Currently the global node sampling distribution for the negative pairs is the degree
        distribution to the 3/4 power. This is the same used in node2vec
        (https://snap.stanford.edu/node2vec/).

        Args:
             batch_size (int): The number of samples to generate for each batch.
                This must be an even number.

        Returns:
            Tuple of lists of target/context pairs and labels – 0 for a negative and 1 for a
             positive pair: ([[target, context] ,... ], [label, ...])
        """
        self._check_parameter_values(batch_size)

        positive_pairs = list()
        negative_pairs = list()

        sample_counter = 0

        all_nodes = list(self.graph.nodes())

        # Use the sampling distribution as per node2vec
        degrees = self.graph.degree()
        sampling_distribution = [degrees[n] ** 0.75 for n in all_nodes]

        done = False
        while not done:
            self.random.shuffle(self.nodes)
            for node in self.nodes:  # iterate over root nodes
                # Get 1 walk at a time. For now its assumed that its a uniform random walker
                walk = self.walker.run(
                    nodes=[node],  # root nodes
                    length=self.length,  # maximum length of a random walk
                    n=1,  # number of random walks per root node
                )

                # (target,contect) pair sampling - GraphSAGE way
                target = walk[0][0]
                context_window = walk[0][1:]
                for context in context_window:
                    # Don't add self pairs
                    if context == target:
                        continue

                    positive_pairs.append((target, context))
                    sample_counter += 1

                    # For each positive sample, add a negative sample.
                    random_sample = self.random.choices(
                        all_nodes, weights=sampling_distribution, k=1
                    )
                    negative_pairs.append((target, *random_sample))
                    sample_counter += 1

                    # If the batch_size number of samples are accumulated, yield.
                    if sample_counter == batch_size:
                        all_pairs = positive_pairs + negative_pairs
                        all_targets = [1] * len(positive_pairs) + [0] * len(
                            negative_pairs
                        )

                        positive_pairs.clear()
                        negative_pairs.clear()
                        sample_counter = 0

                        edge_ids_labels = list(zip(all_pairs, all_targets))
                        self.random.shuffle(edge_ids_labels)
                        edge_ids, edge_labels = [
                            [z[i] for z in edge_ids_labels] for i in (0, 1)
                        ]

                        yield edge_ids, edge_labels

    def _check_parameter_values(self, batch_size):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            batch_size: <int> number of samples to generate in each call of generator

        """

        if (
            batch_size is None
        ):  # must provide a batch size since this is an indicator of how many samples to return
            raise ValueError(
                "({}) The batch_size must be provided to generate samples for each batch in the epoch".format(
                    type(self).__name__
                )
            )

        if type(batch_size) != int:  # must be an integer
            raise TypeError(
                "({}) The batch_size must be positive integer.".format(
                    type(self).__name__
                )
            )

        if batch_size < 1:  # must be greater than 0
            raise ValueError(
                "({}) The batch_size must be positive integer.".format(
                    type(self).__name__
                )
            )

        if (
            batch_size % 2 != 0
        ):  # should be even since we generate 1 negative sample for each positive one.
            raise ValueError(
                "({}) The batch_size must be an even integer since equal number of positive and negative samples are generated in each batch.".format(
                    type(self).__name__
                )
            )
