# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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
    "Neo4jGraphSAGENodeGenerator",
    "Neo4jDirectedGraphSAGENodeGenerator",
]

import numpy as np
import random

from .sampler import (
    Neo4jDirectedBreadthFirstNeighbors,
    Neo4jSampledBreadthFirstWalk,
)

from ...core.graph import GraphSchema
from ...mapper import NodeSequence
from ...mapper.sampled_node_generators import BatchedNodeGenerator
from ...core.experimental import experimental
from .graph import Neo4jStellarGraph


@experimental(reason="the class is not fully tested")
class Neo4jBatchedNodeGenerator:
    """
    Abstract base class for graph data generators from Neo4j.

    The supplied graph should be a StellarGraph object with node features.

    Do not use this base class: use a subclass specific to the method.

    Args:
        graph (Neo4jStellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        schema (GraphSchema): [Optional] Schema for the graph, for heterogeneous graphs.
    """

    def __init__(self, graph, batch_size, schema=None):
        self.graph = graph
        self.batch_size = batch_size

        # This is a node generator and requries a model with one root nodes per query
        self.multiplicity = 1

        # FIXME: Neo4jStellarGraph must support creating schema in order to extend this for HinSAGE
        self.schema = schema

        self.head_node_types = None
        self.sampler = None

    def flow(self, node_ids, targets=None, shuffle=False, seed=None):

        return NodeSequence(
            self.sample_features,
            self.batch_size,
            node_ids,
            targets,
            shuffle=shuffle,
            seed=seed,
        )

    flow.__doc__ = BatchedNodeGenerator.flow.__doc__


@experimental(reason="the class is not fully tested")
class Neo4jGraphSAGENodeGenerator(Neo4jBatchedNodeGenerator):
    """
    A data generator for node prediction with Homogeneous GraphSAGE models

    At minimum, supply the Neo4jStellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a Neo4jStellarGraph object with node features.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = GraphSAGENodeGenerator(G, 50, [10,10])
        train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
        test_data_gen = G_generator.flow(test_node_ids)

    Args:
        graph (Neo4jStellarGraph): Neo4jStellarGraph object
        batch_size (int): Size of batch to return.
        num_samples (list): The number of samples per layer (hop) to take.
        name (int, optional): Optional name for the generator.
    """

    def __init__(self, graph, batch_size, num_samples, name=None):
        super().__init__(graph, batch_size)

        self.num_samples = num_samples
        self.name = name

        # check that there is only a single node type for GraphSAGE

        self.sampler = Neo4jSampledBreadthFirstWalk(graph)

    def sample_features(self, head_nodes, batch_num):
        """
        Collect the features of the nodes sampled from Neo4j,
        and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            batch_num: Ignored, because this is not reproducible.

        Returns:
            A list of the same length as ``num_samples`` of collected features from
            the sampled nodes of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        nodes_per_hop = self.sampler.run(nodes=head_nodes, n=1, n_size=self.num_samples)

        batch_nodes = np.concatenate(nodes_per_hop)
        batch_features = self.graph.node_features(batch_nodes)

        features = []
        idx = 0
        for nodes in nodes_per_hop:
            features_for_slot = batch_features[idx : idx + len(nodes)]
            resize = -1 if np.size(features_for_slot) > 0 else 0
            features.append(
                np.reshape(
                    features_for_slot,
                    (len(head_nodes), resize, features_for_slot.shape[1]),
                )
            )
            idx += len(nodes)

        return features


@experimental(reason="the class is not fully tested")
class Neo4jDirectedGraphSAGENodeGenerator(Neo4jBatchedNodeGenerator):
    """
    A data generator for node prediction with homogeneous GraphSAGE models
    on directed graphs.

    At minimum, supply the StellarDiGraph, the batch size, and the number of
    node samples (separately for in-nodes and out-nodes)
    for each layer of the GraphSAGE model.

    The supplied graph should be a StellarDiGraph object with node features.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = DirectedGraphSAGENodeGenerator(G, 50, [10,5], [5,1])
        train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
        test_data_gen = G_generator.flow(test_node_ids)

    Args:
        graph (Neo4jStellarDiGraph): Neo4jStellarGraph object
        batch_size (int): Size of batch to return.
        in_samples (list): The number of in-node samples per layer (hop) to take.
        out_samples (list): The number of out-node samples per layer (hop) to take.
        name (string, optional): Optional name for the generator
    """

    def __init__(
        self, graph, batch_size, in_samples, out_samples, name=None,
    ):
        super().__init__(graph, batch_size)

        # TODO Add checks for in- and out-nodes sizes
        self.in_samples = in_samples
        self.out_samples = out_samples
        self.name = name

        # Create sampler for GraphSAGE
        self.sampler = Neo4jDirectedBreadthFirstNeighbors(graph)

    def sample_features(self, head_nodes, batch_num):
        """
        Collect the features of the sampled nodes from Neo4j,
        and return these as a list of feature arrays for the GraphSAGE algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            batch_num: Ignored, because this is not reproducible.

        Returns:
            A list of feature tensors from the sampled nodes at each layer, each of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the total number (cumulative product)
            of nodes sampled at the given number of hops from each head node,
            given the sequence of in/out directions.
        """
        node_samples = self.sampler.run(
            nodes=head_nodes, n=1, in_size=self.in_samples, out_size=self.out_samples,
        )

        # Reshape node samples to sensible format
        # Each 'slot' represents the list of nodes sampled from some neighbourhood, and will have a corresponding
        # NN input layer. Every hop potentially generates both in-nodes and out-nodes, held separately,
        # and thus the slot (or directed hop sequence) structure forms a binary tree.

        max_hops = len(self.in_samples)
        max_slots = 2 ** (max_hops + 1) - 1
        features = [None] * max_slots  # flattened binary tree

        batch_nodes = np.concatenate(node_samples)
        batch_features = self.graph.node_features(batch_nodes)

        idx = 0
        for slot in range(max_slots):
            features_for_slot = batch_features[idx : idx + len(node_samples[slot])]
            resize = -1 if np.size(features_for_slot) > 0 else 0
            features[slot] = np.reshape(
                features_for_slot, (len(head_nodes), resize, features_for_slot.shape[1])
            )
            idx += len(node_samples[slot])

        return features
