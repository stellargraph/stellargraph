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
    "Neo4JGraphSAGENodeGenerator",
    "Neo4JDirectedGraphSAGENodeGenerator",
]

import warnings
import numpy as np
import itertools as it

from .sampler import (
    Neo4JDirectedBreadthFirstNeighbors,
    Neo4JSampledBreadthFirstWalk,
)

from ...core.graph import StellarGraph, GraphSchema
from ...mapper import NodeSequence
from ...mapper.sampled_node_generators import BatchedNodeGenerator
from ...core.experimental import experimental


@experimental(reason="the class is not fully tested")
class Neo4JBatchedNodeGenerator(BatchedNodeGenerator):
    """
    Abstract base class for graph data generators from Neo4j.

    The supplied graph should be a StellarGraph object with node features.

    Do not use this base class: use a subclass specific to the method.

    Args:
        G (StellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        neo4j_graphdb (py2neo.Graph): the Neo4j Graph Database object
        schema (GraphSchema): [Optional] Schema for the graph, for heterogeneous graphs.
    """

    def __init__(self, G, batch_size, neo4j_graphdb, schema=None):

        super().__init__(G, batch_size, schema)

        try:
            import py2neo
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e.msg}. StellarGraph can only connect to Neo4j using the 'py2neo' module; please install it",
                name=e.name,
                path=e.path,
            ) from None

        if not isinstance(neo4j_graphdb, py2neo.Graph):
            raise TypeError(
                f"neo4j_graphdb: expected py2neo.Graph, found {type(neo4j_graphdb)}"
            )
        # Create Neo4j graph database object
        self.neo4j_graphdb = neo4j_graphdb

    def flow(self, node_ids, targets=None, shuffle=False, seed=None):

        if self.head_node_types is not None:
            expected_node_type = self.head_node_types[0]
        else:
            expected_node_type = None

        # Check all IDs are actually in the graph and are of expected type
        for n in node_ids:
            try:
                node_type = self.graph.node_type(n)
            except KeyError:
                raise KeyError(f"Node ID {n} supplied to generator not found in graph")

            if expected_node_type is not None and (node_type != expected_node_type):
                raise ValueError(
                    f"Node ID {n} not of expected type {expected_node_type}"
                )

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
class Neo4JGraphSAGENodeGenerator(Neo4JBatchedNodeGenerator):
    """
    A data generator for node prediction with Homogeneous GraphSAGE models

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a StellarGraph object with node features.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = GraphSAGENodeGenerator(G, 50, [10,10], neo4j_graphdb)
        train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
        test_data_gen = G_generator.flow(test_node_ids)

    Args:
        G (StellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        num_samples (list): The number of samples per layer (hop) to take.
        neo4j_graphdb (py2neo.Graph): the Neo4j Graph Database object.
        seed (int, optional): Random seed for the node sampler.
    """

    def __init__(self, G, batch_size, num_samples, neo4j_graphdb, seed=None, name=None):
        super().__init__(G, batch_size, neo4j_graphdb)

        self.num_samples = num_samples
        self.head_node_types = self.schema.node_types
        self.name = name

        # check that there is only a single node type for GraphSAGE

        if len(self.head_node_types) > 1:
            warnings.warn(
                "running homogeneous GraphSAGE on a graph with multiple node types",
                RuntimeWarning,
                stacklevel=2,
            )

        self.sampler = Neo4JSampledBreadthFirstWalk(
            G, graph_schema=self.schema, seed=seed
        )

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
        nodes_per_hop = self.sampler.run(
            self.neo4j_graphdb, nodes=head_nodes, n=1, n_size=self.num_samples
        )
        node_type = self.head_node_types[0]

        # Get features for sampled nodes
        batch_feats = [
            self.graph.node_features(layer_nodes, node_type)
            for layer_nodes in nodes_per_hop
        ]

        # Resize features for sampled nodes
        batch_feats = [
            np.reshape(a, (len(head_nodes), -1 if np.size(a) > 0 else 0, a.shape[1]))
            for a in batch_feats
        ]
        return batch_feats


@experimental(reason="the class is not fully tested")
class Neo4JDirectedGraphSAGENodeGenerator(Neo4JBatchedNodeGenerator):
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

        G_generator = DirectedGraphSAGENodeGenerator(G, 50, [10,5], [5,1], neo4j_graphdb)
        train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
        test_data_gen = G_generator.flow(test_node_ids)

    Args:
        G (StellarDiGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        in_samples (list): The number of in-node samples per layer (hop) to take.
        out_samples (list): The number of out-node samples per layer (hop) to take.
        neo4j_graphdb (py2neo.Graph): The Neo4j Graph database object
        seed (int, optional) Random seed for the node sampler.
    """

    def __init__(
        self,
        G,
        batch_size,
        in_samples,
        out_samples,
        neo4j_graphdb,
        seed=None,
        name=None,
    ):
        super().__init__(G, batch_size, neo4j_graphdb)

        # TODO Add checks for in- and out-nodes sizes
        self.in_samples = in_samples
        self.out_samples = out_samples
        self.head_node_types = self.schema.node_types
        self.name = name

        # Check that there is only a single node type for GraphSAGE
        if len(self.head_node_types) > 1:
            warnings.warn(
                "running homogeneous GraphSAGE on a graph with multiple node types",
                RuntimeWarning,
                stacklevel=2,
            )

        # Create sampler for GraphSAGE
        self.sampler = Neo4JDirectedBreadthFirstNeighbors(
            G, graph_schema=self.schema, seed=seed
        )

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
            self.neo4j_graphdb,
            nodes=head_nodes,
            n=1,
            in_size=self.in_samples,
            out_size=self.out_samples,
        )

        # Reshape node samples to sensible format
        # Each 'slot' represents the list of nodes sampled from some neighbourhood, and will have a corresponding
        # NN input layer. Every hop potentially generates both in-nodes and out-nodes, held separately,
        # and thus the slot (or directed hop sequence) structure forms a binary tree.

        node_type = self.head_node_types[0]

        max_hops = len(self.in_samples)
        max_slots = 2 ** (max_hops + 1) - 1
        features = [None] * max_slots  # flattened binary tree

        for slot in range(max_slots):
            nodes_in_slot = node_samples[slot]
            features_for_slot = self.graph.node_features(nodes_in_slot, node_type)
            resize = -1 if np.size(features_for_slot) > 0 else 0
            features[slot] = np.reshape(
                features_for_slot, (len(head_nodes), resize, features_for_slot.shape[1])
            )

        return features
