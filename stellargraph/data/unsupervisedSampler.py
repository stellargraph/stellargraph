#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:00:01 2019

@author: hab031
"""


__all__ = ["UnsupervisedSampler"]


import random

from stellargraph.core.utils import is_real_iterable
from stellargraph.core.graph import StellarGraphBase
from stellargraph.data.explorer import UniformRandomWalk


class UnsupervisedSampler:
    def __init__(
        self,
        G,
        nodes=None,
        batch_size=None,
        walker=None,
        length=1,
        number_of_walks=1,
        p=0.5,
        q=2.0,
        weighted=False,
        seed=None,
    ):

        if not isinstance(G, StellarGraphBase):
            raise ValueError(
                "({}) Graph must be a StellarGraph object.".format(type(self).__name__)
            )
        else:
            self.graph = G

        # walker
        if walker is None:
            walker = UniformRandomWalk(G)
        elif not isinstance(walker, UniformRandomWalk):
            raise TypeError(
                "({}) Only Uniform Random Walks are possible".format(
                    type(self).__name__
                )
            )

        else:
            self.walker = walker

        if nodes is None:
            self.nodes = G.nodes()
        elif not is_real_iterable(nodes):
            raise ValueError("nodes parameter should be an iterableof node IDs.")
        else:
            self.nodes = nodes

        if batch_size is None:
            raise ValueError(
                "({}) Batch size must be provided to determine how many samples to generate in an epoch.".format(
                    type(self).__name__
                )
            )
        elif batch_size is not None:
            if type(batch_size) != int:
                raise ValueError(
                    "({}) The training sample size, sample_size, should be positive integer or None.".format(
                        type(self).__name__
                    )
                )
            elif batch_size < 0:
                raise ValueError(
                    "({}) The training sample size, sample_size, should be positive integer or None.".format(
                        type(self).__name__
                    )
                )
        else:
            self.batch_size = batch_size

        self.length = length
        self.number_of_walks = number_of_walks
        self.p = p
        self.q = q
        self.weighted = weighted
        self.seed = seed

    def generator(
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

        self._check_parameter_values(
            nodes=nodes,
            n=n,
            p=p,
            q=q,
            length=length,
            seed=seed,
            weighted=weighted,
            edge_weight_label=edge_weight_label,
        )

        if seed:
            # seed a new random number generator
            rs = random.Random(seed)
        else:
            # Restore the random state
            rs = self._random_state

        positive_pairs = list()
        negative_pairs = list()

        positive_samples_counter = 0
        negative_samples_counter = 0

        all_nodes = list(self.graph.nodes())

        # Use the sampling distribution as per node2vec
        degrees = self.graph.degree()
        sampling_distribution = [degrees[n] ** 0.75 for n in all_nodes]

        if self.batch_size is None:
            walks = self.walker.run(
                nodes=self.nodes,
                n=self.number_of_walks,
                length=self.length,
                p=self.p,
                q=self.q,
                weighted=self.weighted,
                seed=self.seed,
            )

            positive_samples_counter = 0
            negative_samples_counter = 0

            # (target,contect) pair sampling - GraphSAGE way
            for w in walks:
                target = w[0]
                context_window = w[1:]

                positive_samples_counter = 0

                for context in context_window:
                    # Don't add self pairs
                    if context != target:
                        positive_pairs.append((target, context))
                        positive_samples_counter += 1

                negative_samples_counter = 0
                while negative_samples_counter < positive_samples_counter:
                    random_samples = random.choices(
                        all_nodes,
                        weights=sampling_distribution,
                        k=positive_samples_counter - negative_samples_counter,
                    )
                    for context in random_samples:
                        if not context in context_window:
                            negative_pairs.append((target, context))
                            negative_samples_counter = negative_samples_counter + 1

            all_pairs = positive_pairs + negative_pairs
            all_targets = [1] * len(positive_pairs) + [0] * len(negative_pairs)
            edge_ids_labels = list(zip(all_pairs, all_targets))
            random.shuffle(edge_ids_labels)
            edge_ids, edge_labels = [[z[i] for z in edge_ids_labels] for i in (0, 1)]
            yield edge_ids, edge_labels

        else:
            done = False
            while not done:
                for node in self.nodes:  # iterate over root nodes
                    # Get 1 walk at a time. For now its assumed that its a biased random walker
                    walk = self.walker.run(
                        nodes=[node],  # root nodes
                        length=self.length,  # maximum length of a random walk
                        n=1,  # number of random walks per root node
                        p=self.p,  # Defines (unormalised) probability, 1/p, of returning to source node
                        q=self.q,  # Defines (unormalised) probability, 1/q, for moving away from source node
                        weighted=self.weighted,
                        seed=self.seed,
                    )
                    # (target,contect) pair sampling - GraphSAGE way
                    target = walk[0][0]
                    context_window = walk[0][1:]
                    for context in context_window:
                        # Don't add self pairs
                        if context != target:
                            positive_pairs.append((target, context))
                            positive_samples_counter += 1
                            # For each positive sample, add a negative sample.
                            # Negative samples are contexts not in the current walk with respect to the current target(start node of the walk).
                            while negative_samples_counter < positive_samples_counter:
                                random_sample = random.choices(
                                    all_nodes, weights=sampling_distribution, k=1
                                )
                                if not random_sample in context_window:
                                    negative_pairs.append((target, *random_sample))
                                    negative_samples_counter = (
                                        negative_samples_counter + 1
                                    )
                                    # If the batch_size number of samples are accumulated, yield.
                                    if (
                                        positive_samples_counter
                                        + negative_samples_counter
                                    ) == self.batch_size:
                                        all_pairs = positive_pairs + negative_pairs
                                        all_targets = [1] * len(positive_pairs) + [
                                            0
                                        ] * len(negative_pairs)
                                        edge_ids_labels = list(
                                            zip(all_pairs, all_targets)
                                        )
                                        random.shuffle(edge_ids_labels)
                                        edge_ids, edge_labels = [
                                            [z[i] for z in edge_ids_labels]
                                            for i in (0, 1)
                                        ]

                                        positive_pairs.clear()
                                        negative_pairs.clear()
                                        positive_samples_counter = 0
                                        negative_samples_counter = 0
                                        yield edge_ids, edge_labels

    def _check_parameter_values(
        self,
        nodes,
        number_of_walks,
        p,
        q,
        length,
        seed,
        weighted,
        edge_weight_label,
        batch_size,
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

        if type(number_of_walks) != int:
            raise ValueError(
                "({}) The number of walks per root node, n, should be integer type.".format(
                    type(self).__name__
                )
            )

        if number_of_walks <= 0:
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

        if batch_size is not None:
            if batch_size < 0:
                raise ValueError(
                    "({}) The training sample size, sample_size, should be positive integer or None.".format(
                        type(self).__name__
                    )
                )
            if type(batch_size) != int:
                raise ValueError(
                    "({}) The training sample size, sample_size, should be positive integer or None.".format(
                        type(self).__name__
                    )
                )
