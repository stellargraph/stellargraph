#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:00:01 2019

@author: hab031
"""


__all__ = ["UnsupervisedSampler"]


import random
import numpy as np

from stellargraph.core.utils import is_real_iterable
from stellargraph.core.graph import StellarGraphBase
from stellargraph.data.explorer import UniformRandomWalk


class UnsupervisedSampler:
    def __init__(
        self, G, walker=None, nodes=None, length=2, number_of_walks=1, seed=None
    ):

        # Initialize the random state
        self._random_state = random.Random(seed)

        # Initialize a numpy random state (for numpy random methods)
        self._np_random_state = np.random.RandomState(seed=seed)

        if not isinstance(G, StellarGraphBase):
            raise ValueError(
                "({}) Graph must be a StellarGraph object.".format(type(self).__name__)
            )
        else:
            self.graph = G

        # walker
        if walker is not None:
            if not isinstance(
                walker, UniformRandomWalk
            ):  # only work with Uniform Random Walker at the moment
                raise TypeError(
                    "({}) Only Uniform Random Walks are possible".format(
                        type(self).__name__
                    )
                )
            else:
                self.walker = UniformRandomWalk(G)
        else:
            self.walker = UniformRandomWalk(G)

        if nodes is None:
            self.nodes = (
                G.nodes()
            )  # if no root nodes are provided for sampling defaulting to using all nodes as root nodes.
        elif not is_real_iterable(nodes):  # check whether the nodes provided are valid.
            raise ValueError("nodes parameter should be an iterableof node IDs.")
        else:
            self.nodes = nodes

        if (
            length < 2
        ):  # for a  sample pair need at least one neighbor beyond the root node.
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

        if seed is not None:
            self.seed = seed

    def generator(self, batch_size):

        self._check_parameter_values(batch_size=batch_size)

        if batch_size % 2 != 0:
            batch_size += 1

        positive_pairs = list()
        negative_pairs = list()

        positive_samples_counter = 0
        negative_samples_counter = 0

        all_nodes = list(self.graph.nodes())

        # Use the sampling distribution as per node2vec
        degrees = self.graph.degree()
        sampling_distribution = [degrees[n] ** 0.75 for n in all_nodes]

        done = False
        while not done:
            for node in self.nodes:  # iterate over root nodes
                # Get 1 walk at a time. For now its assumed that its a uniform random walker
                walk = self.walker.run(
                    nodes=[node],  # root nodes
                    length=self.length,  # maximum length of a random walk
                    n=1,  # number of random walks per root node
                    seed=None,
                )
                # (target,contect) pair sampling - GraphSAGE way
                target = walk[0][0]
                context_window = walk[0][1:]
                if len(context_window) >= 1:
                    for context in context_window:
                        # Don't add self pairs
                        if context != target:
                            positive_pairs.append((target, context))
                            positive_samples_counter += 1

                            # For each positive sample, add a negative sample.
                            random_sample = random.choices(
                                all_nodes, weights=sampling_distribution, k=1
                            )
                            negative_pairs.append((target, *random_sample))
                            negative_samples_counter = negative_samples_counter + 1

                            # If the batch_size number of samples are accumulated, yield.
                            if (
                                positive_samples_counter + negative_samples_counter
                            ) >= batch_size:
                                all_pairs = positive_pairs + negative_pairs
                                all_targets = [1] * len(positive_pairs) + [0] * len(
                                    negative_pairs
                                )
                                edge_ids_labels = list(zip(all_pairs, all_targets))
                                random.shuffle(edge_ids_labels)
                                edge_ids, edge_labels = [
                                    [z[i] for z in edge_ids_labels] for i in (0, 1)
                                ]

                                positive_pairs.clear()
                                negative_pairs.clear()
                                positive_samples_counter = 0
                                negative_samples_counter = 0
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

        if batch_size < 1:  # must be greater than 0
            raise ValueError(
                "({}) The batch_size must be positive integer.".format(
                    type(self).__name__
                )
            )
