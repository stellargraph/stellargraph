# -*- coding: utf-8 -*-
#
# Copyright 2018-2019 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generators that create batches of data from a machine-learnign ready graph
for link prediction/link attribute inference problems using GraphSAGE and HinSAGE.

"""
__all__ = ["GraphSAGELinkGenerator", "HinSAGELinkGenerator"]

import random
import operator
import numpy as np
import itertools as it
import operator
import collections
from functools import reduce
from tensorflow import keras

from ..core.graph import StellarGraphBase
from ..data import (
    SampledBreadthFirstWalk,
    SampledHeterogeneousBreadthFirstWalk,
    UniformRandomWalk,
    UnsupervisedSampler,
)
from ..core.utils import is_real_iterable
from .sequences import LinkSequence, OnDemandLinkSequence


class GraphSAGELinkGenerator:
    """
    A data generator for link prediction with Homogeneous GraphSAGE models

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`.flow` method supplying the nodes and (optionally) targets,
    or an UnsupervisedSampler instance that generates node samples on demand,
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = GraphSageLinkGenerator(G, 50, [10,10])
        train_data_gen = G_generator.flow(edge_ids)

    Args:
        G (StellarGraph): A machine-learning ready graph.
        batch_size (int): Size of batch of links to return.
        num_samples (list): List of number of neighbour node samples per GraphSAGE layer (hop) to take.
        seed (int or str), optional: Random seed for the sampling methods.
    """

    def __init__(self, G, batch_size, num_samples, seed=None):
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        G.check_graph_for_ml(features=True)

        self.graph = G
        self.num_samples = num_samples
        self.batch_size = batch_size

        # This is a link generator and requries a model with two root nodes per query
        self.multiplicity = 2

        # We need a schema for compatibility with HinSAGE
        self.schema = G.create_graph_schema(create_type_maps=True)

        # The sampler used to generate random samples of neighbours
        self.sampler = SampledBreadthFirstWalk(G, graph_schema=self.schema, seed=seed)

    def sample_features(self, head_links, sampling_schema):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_links: An iterable of edges to perform sampling for.
            sampling_schema: The sampling schema for the model

        Returns:
            A list of the same length as ``num_samples`` of collected features from
            the sampled nodes of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        node_type = sampling_schema[0][0][0]
        head_size = len(head_links)

        # The number of samples for each head node (not including itself)
        num_full_samples = np.sum(np.cumprod(self.num_samples))

        # Reshape node samples to sensible format
        def get_levels(loc, lsize, samples_per_hop, walks):
            end_loc = loc + lsize
            walks_at_level = list(it.chain(*[w[loc:end_loc] for w in walks]))
            if len(samples_per_hop) < 1:
                return [walks_at_level]
            return [walks_at_level] + get_levels(
                end_loc, lsize * samples_per_hop[0], samples_per_hop[1:], walks
            )

        # Get sampled nodes for the subgraphs for the edges where each edge is a tuple
        # of 2 nodes, so we are extracting 2 head nodes per edge
        batch_feats = []
        for hns in zip(*head_links):
            node_samples = self.sampler.run(nodes=hns, n=1, n_size=self.num_samples)

            nodes_per_hop = get_levels(0, 1, self.num_samples, node_samples)

            # Get features for the sampled nodes
            batch_feats.append(
                [
                    self.graph.get_feature_for_nodes(layer_nodes, node_type)
                    for layer_nodes in nodes_per_hop
                ]
            )

        # Resize features to (batch_size, n_neighbours, feature_size)
        # and re-pack features into a list where source, target feats alternate
        # This matches the GraphSAGE link model with (node_src, node_dst) input sockets:
        batch_feats = [
            np.reshape(feats, (head_size, -1, feats.shape[1]))
            for ab in zip(*batch_feats)
            for feats in ab
        ]
        return batch_feats

    def flow(self, link_ids, targets=None, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied edge IDs and numeric targets.

        The edge IDs are the edges to train or inference on. They are
        expected to by tuples of (source_id, destination_id).

        The targets are an array of numeric targets corresponding to the
        supplied link_ids to be used by the downstream task. They should
        be given in the same order as the list of link IDs.
        If they are not specified (for example, for use in prediction),
        the targets will not be available to the downsteam task.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            link_ids (list or UnsupervisedSampler): an iterable of (src_id, dst_id) tuples
                specifying the edges or an UnsupervisedSampler object that has a generator
                method to generate samples on the fly.
            targets (optional, array): a 2D array of numeric targets with shape
                `(len(link_ids), target_size)`
            shuffle (optional, bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A LinkSequence or OnDemandLinkSequence object to use with the GraphSAGE model
            methods :meth:`fit_generator`, :meth:`evaluate_generator`, and :meth:`predict_generator`
        """
        # Pass sampler to on-demand link sequence generation
        if isinstance(link_ids, UnsupervisedSampler):
            return OnDemandLinkSequence(self, link_ids)

        # Otherwise pass iterable (check?) to standard LinkSequence
        elif isinstance(link_ids, collections.Iterable):
            return LinkSequence(self, link_ids, targets, shuffle)

        else:
            raise TypeError(
                "Argument to .flow not recognised. "
                "Please pass a list of samples or a UnsupervisedSampler object."
            )


class HinSAGELinkGenerator:
    """
    A data generator for link prediction with Heterogeneous HinSAGE models

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    The generator should be given the (src,dst) node types usng

    * It's possible to do link prediction on a graph where that link type is completely removed from the graph
      (e.g., "same_as" links in ER)


    Args:
        g (StellarGraph): A machine-learning ready graph.
        batch_size (int): Size of batch of links to return.
        num_samples (list): List of number of neighbour node samples per GraphSAGE layer (hop) to take.
        head_node_types (list): List of the types (str) of the two head nodes forming the node pair.
        seed (int or str, optional): Random seed for the sampling methods.

    Example::

        G_generator = HinSAGELinkGenerator(G, 50, [10,10])
        data_gen = G_generator.flow(edge_ids)
    """

    def __init__(self, G, batch_size, num_samples, head_node_types, seed=None):
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        G.check_graph_for_ml(features=True)

        self.graph = G
        self.num_samples = num_samples
        self.batch_size = batch_size

        # This is a node generator and requries a model with one root nodes per query
        self.head_node_types = head_node_types
        self.multiplicity = 2

        # We need a schema for compatibility with HinSAGE
        self.schema = G.create_graph_schema(create_type_maps=True)

        # The sampler used to generate random samples of neighbours
        self.sampler = SampledHeterogeneousBreadthFirstWalk(
            G, graph_schema=self.schema, seed=seed
        )

    def _get_features(self, node_samples, head_size):
        """
        Collect features from sampled nodes.
        Args:
            node_samples: A list of lists of node IDs
            head_size: The number of head nodes (typically the batch size).

        Returns:
            A list of numpy arrays that store the features for each head
            node.
        """
        # Note the if there are no samples for a node a zero array is returned.
        # Resize features to (batch_size, n_neighbours, feature_size)
        # for each node type (note that we can have different feature size for each node type)
        batch_feats = [
            self.graph.get_feature_for_nodes(layer_nodes, nt)
            for nt, layer_nodes in node_samples
        ]

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [np.reshape(a, (head_size, -1, a.shape[1])) for a in batch_feats]

        return batch_feats

    def sample_features(self, head_links, sampling_schema):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_links (list): An iterable of edges to perform sampling for.
            sampling_schema (dict): The sampling schema for the model

        Returns:
            A list of the same length as `num_samples` of collected features from
            the sampled nodes of shape: `(len(head_nodes), num_sampled_at_layer, feature_size)`
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        nodes_by_type = []
        for ii in range(2):
            # Extract head nodes from edges: each edge is a tuple of 2 nodes, so we are extracting 2 head nodes per edge
            head_nodes = [e[ii] for e in head_links]

            # Get sampled nodes for the subgraphs starting from the (src, dst) head nodes
            # nodes_samples is list of two lists: [[samples for src], [samples for dst]]
            node_samples = self.sampler.run(
                nodes=head_nodes, n=1, n_size=self.num_samples
            )

            # Reshape node samples to the required format for the HinSAGE model
            # This requires grouping the sampled nodes by edge type and in order
            nodes_by_type.append(
                [
                    (
                        nt,
                        reduce(
                            operator.concat,
                            (samples[ks] for samples in node_samples for ks in indices),
                            [],
                        ),
                    )
                    for nt, indices in sampling_schema[ii]
                ]
            )

        # Interlace the two lists, nodes_by_type[0] (for src head nodes) and nodes_by_type[1] (for dst head nodes)
        nodes_by_type = [
            tuple((ab[0][0], reduce(operator.concat, (ab[0][1], ab[1][1]))))
            for ab in zip(nodes_by_type[0], nodes_by_type[1])
        ]

        batch_feats = self._get_features(nodes_by_type, len(head_links))

        return batch_feats

    def flow(self, link_ids, targets=None, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied edge IDs and numeric targets.

        The edge IDs are the edges to train or inference on. They are
        expected to by tuples of (source_id, destination_id).

        The targets are an array of numeric targets corresponding to the
        supplied link_ids to be used by the downstream task. They should
        be given in the same order as the list of link IDs.
        If they are not specified (for example, for use in prediction),
        the targets will not be available to the downsteam task.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            link_ids: an iterable of (src_id, dst_id) tuples specifying the
                edges.
            targets: a 2D array of numeric targets with shape
                ``(len(link_ids), target_size)``
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A LinkSequence object to use with the GraphSAGE model
            methods :meth:`fit_generator`, :meth:`evaluate_generator`, and :meth:`predict_generator`
        """
        if not isinstance(link_ids, collections.Iterable):
            raise TypeError(
                "Argument to .flow not recognised. "
                "Please pass a list of samples or a UnsupervisedSampler object."
            )

        return LinkSequence(self, link_ids, targets, shuffle)
