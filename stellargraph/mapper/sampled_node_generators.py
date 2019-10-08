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
Mappers to provide input data for the graph models in layers.

"""
__all__ = [
    "GraphSAGENodeGenerator",
    "HinSAGENodeGenerator",
    "DirectedGraphSAGENodeGenerator",
]

import warnings
import operator
import random
import numpy as np
import itertools as it
import networkx as nx
import scipy.sparse as sps
from tensorflow.keras import backend as K
from functools import reduce
from tensorflow.keras.utils import Sequence

from ..data import (
    SampledBreadthFirstWalk,
    SampledHeterogeneousBreadthFirstWalk,
    DirectedBreadthFirstNeighbours,
)
from ..core.graph import StellarGraphBase, GraphSchema, StellarDiGraph
from ..core.utils import is_real_iterable
from .sequences import NodeSequence


class GraphSAGENodeGenerator:
    """
    A data generator for node prediction with Homogeneous GraphSAGE models

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = GraphSAGENodeGenerator(G, 50, [10,10])
        train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
        test_data_gen = G_generator.flow(test_node_ids)

    Args:
        G (StellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        num_samples (list): The number of samples per layer (hop) to take.
        schema (GraphSchema): [Optional] Graph schema for G.
        seed (int): [Optional] Random seed for the node sampler.
    """

    def __init__(self, G, batch_size, num_samples, schema=None, seed=None):
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        self.graph = G
        self.num_samples = num_samples
        self.batch_size = batch_size

        # This is a node generator and requries a model with one root nodes per query
        self.multiplicity = 1

        # Check if the graph has features
        G.check_graph_for_ml()

        # We need a schema for compatibility with HinSAGE
        if schema is None:
            self.schema = G.create_graph_schema(create_type_maps=True)
        elif isinstance(schema, GraphSchema):
            self.schema = schema
        else:
            raise TypeError("Schema must be a GraphSchema object")

        # Check that there is only a single node type for GraphSAGE
        if len(self.schema.node_types) > 1:
            print(
                "Warning: running homogeneous GraphSAGE on a graph with multiple node types"
            )

        # Create sampler for GraphSAGE
        self.sampler = SampledBreadthFirstWalk(G, graph_schema=self.schema, seed=seed)

    def sample_features(self, head_nodes, sampling_schema):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            sampling_schema: The sampling schema for the model

        Returns:
            A list of the same length as ``num_samples`` of collected features from
            the sampled nodes of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        node_samples = self.sampler.run(nodes=head_nodes, n=1, n_size=self.num_samples)

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

        nodes_per_hop = get_levels(0, 1, self.num_samples, node_samples)
        node_type = sampling_schema[0][0][0]

        # Get features for sampled nodes
        batch_feats = [
            self.graph.get_feature_for_nodes(layer_nodes, node_type)
            for layer_nodes in nodes_per_hop
        ]

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [
            np.reshape(a, (len(head_nodes), -1 if np.size(a) > 0 else 0, a.shape[1]))
            for a in batch_feats
        ]
        return batch_feats

    def flow(self, node_ids, targets=None, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        The node IDs are the nodes to train or inference on: the embeddings
        calculated for these nodes are passed to the downstream task. These
        are a subset of the nodes in the graph.

        The targets are an array of numeric targets corresponding to the
        supplied node_ids to be used by the downstream task. They should
        be given in the same order as the list of node IDs.
        If they are not specified (for example, for use in prediction),
        the targets will not be available to the downsteam task.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            node_ids: an iterable of node IDs
            targets: a 2D array of numeric targets with shape
                `(len(node_ids), target_size)`
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with the GraphSAGE model
            in Keras methods ``fit_generator``, ``evaluate_generator``,
            and ``predict_generator``

        """
        return NodeSequence(self, node_ids, targets, shuffle=shuffle)

    def flow_from_dataframe(self, node_targets, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_targets: a Pandas DataFrame of numeric targets indexed
                by the node ID for that target.
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with the GraphSAGE model
            in Keras methods ``fit_generator``, ``evaluate_generator``,
            and ``predict_generator``

        """
        return NodeSequence(
            self, node_targets.index, node_targets.values, shuffle=shuffle
        )


class DirectedGraphSAGENodeGenerator:
    """
    A data generator for node prediction with homogeneous GraphSAGE models
    on directed graphs.

    At minimum, supply the StellarDiGraph, the batch size, and the number of
    node samples (separately for in-nodes and out-nodes)
    for each layer of the GraphSAGE model.

    The supplied graph should be a StellarDiGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = DirectedGraphSAGENodeGenerator(G, 50, [10,5], [5,1])
        train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
        test_data_gen = G_generator.flow(test_node_ids)

    Args:
        G (StellarDiGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        in_samples (list): The number of in-node samples per layer (hop) to take.
        out_samples (list): The number of out-node samples per layer (hop) to take.
        schema (GraphSchema): [Optional] Graph schema for G.
        seed (int): [Optional] Random seed for the node sampler.
    """

    def __init__(self, G, batch_size, in_samples, out_samples, schema=None, seed=None):
        if not isinstance(G, StellarDiGraph):
            raise TypeError("Graph must be a StellarDiGraph object.")
        # TODO Add checks for in- and out-nodes sizes

        self.graph = G
        self.in_samples = in_samples
        self.out_samples = out_samples
        self.batch_size = batch_size

        # This is a node generator and requries a model with one root nodes per query
        self.multiplicity = 1

        # Check if the graph has features
        G.check_graph_for_ml()

        # We need a schema for compatibility with HinSAGE
        if schema is None:
            self.schema = G.create_graph_schema(create_type_maps=True)
        elif isinstance(schema, GraphSchema):
            self.schema = schema
        else:
            raise TypeError("Schema must be a GraphSchema object")

        # Check that there is only a single node type for GraphSAGE
        if len(self.schema.node_types) > 1:
            print(
                "Warning: running homogeneous GraphSAGE on a graph with multiple node types"
            )

        # Create sampler for GraphSAGE
        self.sampler = DirectedBreadthFirstNeighbours(
            G, graph_schema=self.schema, seed=seed
        )

    def sample_features(self, head_nodes, sampling_schema):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            sampling_schema: The sampling schema for the model

        Returns:
            A list of feature tensors from the sampled nodes at each layer, each of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the total number (cumulative product)
            of nodes sampled at the given number of hops from each head node,
            given the sequence of in/out directions.
        """
        node_samples = self.sampler.run(
            nodes=head_nodes, n=1, in_size=self.in_samples, out_size=self.out_samples
        )

        # Reshape node samples to sensible format
        # Each 'slot' represents the list of nodes sampled from some neighbourhood, and will have a corresponding
        # NN input layer. Every hop potentially generates both in-nodes and out-nodes, held separately,
        # and thus the slot (or directed hop sequence) structure forms a binary tree.

        node_type = sampling_schema[0][0][0]

        max_hops = len(self.in_samples)
        max_slots = 2 ** (max_hops + 1) - 1
        features = [None] * max_slots  # flattened binary tree

        for slot in range(max_slots):
            nodes_in_slot = list(it.chain(*[sample[slot] for sample in node_samples]))
            features_for_slot = self.graph.get_feature_for_nodes(
                nodes_in_slot, node_type
            )
            resize = -1 if np.size(features_for_slot) > 0 else 0
            features[slot] = np.reshape(
                features_for_slot, (len(head_nodes), resize, features_for_slot.shape[1])
            )

        return features

    def flow(self, node_ids, targets=None, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        The node IDs are the nodes to train or inference on: the embeddings
        calculated for these nodes are passed to the downstream task. These
        are a subset of the nodes in the graph.

        The targets are an array of numeric targets corresponding to the
        supplied node_ids to be used by the downstream task. They should
        be given in the same order as the list of node IDs.
        If they are not specified (for example, for use in prediction),
        the targets will not be available to the downsteam task.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            node_ids: an iterable of node IDs
            targets: a 2D array of numeric targets with shape
                `(len(node_ids), target_size)`
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with the GraphSAGE model
            in Keras methods ``fit_generator``, ``evaluate_generator``,
            and ``predict_generator``

        """
        return NodeSequence(self, node_ids, targets, shuffle=shuffle)

    def flow_from_dataframe(self, node_targets, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_targets: a Pandas DataFrame of numeric targets indexed
                by the node ID for that target.
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with the GraphSAGE model
            in Keras methods ``fit_generator``, ``evaluate_generator``,
            and ``predict_generator``

        """
        return NodeSequence(
            self, node_targets.index, node_targets.values, shuffle=shuffle
        )


class HinSAGENodeGenerator:
    """Keras-compatible data mapper for Heterogeneous GraphSAGE (HinSAGE)

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the HinSAGE model.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Note that the shuffle argument should be True for training and
    False for prediction.

    Args:
        G (StellarGraph): The machine-learning ready graph
        batch_size (int): Size of batch to return
        num_samples (list): The number of samples per layer (hop) to take
        head_node_type (str): The node type that will be given to the generator
            using the `flow` method, the model will expect this node type.
        schema (GraphSchema, optional): Graph schema for G.
        seed (int, optional): Random seed for the node sampler

    Example::

         G_generator = HinSAGENodeGenerator(G, 50, [10,10])
         train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
         test_data_gen = G_generator.flow(test_node_ids)

     """

    def __init__(
        self, G, batch_size, num_samples, head_node_type, schema=None, seed=None
    ):
        self.graph = G
        self.num_samples = num_samples
        self.batch_size = batch_size

        # This is a node generator and requries a model with one root nodes per query
        self.multiplicity = 1

        # We require a StellarGraph
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        G.check_graph_for_ml(features=True)

        # Generate schema
        if schema is None:
            self.schema = G.create_graph_schema(create_type_maps=True)
        elif isinstance(schema, GraphSchema):
            self.schema = schema
        else:
            raise TypeError("Schema must be a GraphSchema object")

        # The head node type
        if head_node_type not in self.schema.node_types:
            raise KeyError("Supplied head node type must exist in the graph")
        self.head_node_types = [head_node_type]

        # Create sampler for HinSAGE
        self.sampler = SampledHeterogeneousBreadthFirstWalk(
            G, graph_schema=self.schema, seed=seed
        )

    def sample_features(self, head_nodes, sampling_schema):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            sampling_schema: The node sampling schema for the HinSAGE model,
                this is can be generated by the ``GraphSchema`` object.

        Returns:
            A list of the same length as ``num_samples`` of collected features from
            the sampled nodes of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        # Get sampled nodes
        node_samples = self.sampler.run(nodes=head_nodes, n=1, n_size=self.num_samples)

        # Reshape node samples to the required format for the HinSAGE model
        # This requires grouping the sampled nodes by edge type and in order
        nodes_by_type = [
            (
                nt,
                reduce(
                    operator.concat,
                    (samples[ks] for samples in node_samples for ks in indices),
                    [],
                ),
            )
            for nt, indices in sampling_schema[0]
        ]

        # Get features
        batch_feats = [
            self.graph.get_feature_for_nodes(layer_nodes, nt)
            for nt, layer_nodes in nodes_by_type
        ]

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [
            np.reshape(a, (len(head_nodes), -1 if np.size(a) > 0 else 0, a.shape[1]))
            for a in batch_feats
        ]

        return batch_feats

    def flow(self, node_ids, targets=None, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        The node IDs are the nodes to train or inference on: the embeddings
        calculated for these nodes are passed to the downstream task. These
        are a subset of the nodes in the graph.

        The targets are an array of numeric targets corresponding to the
        supplied node_ids to be used by the downstream task. They should
        be given in the same order as the list of node IDs.
        If they are not specified (for example, for use in prediction),
        the targets will not be available to the downsteam task.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            node_ids (iterable): The head node IDs
            targets (Numpy array): a 2D array of numeric targets with shape
                ``(len(node_ids), target_size)``
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with the GraphSAGE model
            in Keras methods `fit_generator`, `evaluate_generator`,
            and `predict_generator`.

        """
        return NodeSequence(self, node_ids, targets, shuffle=shuffle)

    def flow_from_dataframe(self, node_targets, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            node_targets (DataFrame): Numeric targets indexed
                by the node ID for that target.
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with the GraphSAGE model
            in Keras methods `fit_generator`, `evaluate_generator`,
            and `predict_generator`.
        """
        return NodeSequence(
            self, node_targets.index, node_targets.values, shuffle=shuffle
        )
