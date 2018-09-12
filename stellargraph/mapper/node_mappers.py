# -*- coding: utf-8 -*-
#
# Copyright 2018 Data61, CSIRO
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
__all__ = ["GraphSAGENodeGenerator", "HinSAGENodeGenerator"]

import collections
import operator
from functools import reduce

import networkx as nx
import numpy as np
import itertools as it
from typing import AnyStr, Any, List, Optional
from keras.utils import Sequence

from stellargraph.data.explorer import (
    SampledBreadthFirstWalk,
    SampledHeterogeneousBreadthFirstWalk,
)
from stellargraph.data.stellargraph import StellarGraphBase
from stellargraph.data.utils import is_real_iterable


class NodeSequence(Sequence):
    """Keras-compatible data generator to use with
    Keras methods `fit_generator`, `evaluate_generator`,
    and `predict_generator`

    This class generated data samples for node inference models
    and should be created using the `.flow(...)` method of
    `GraphSAGENodeGenerator` or `HinSAGENodeGenerator`.

    These Generators are classes that capture the graph structure
    and the feature vectors of each node. These generator classes
    are used within the NodeSequence to generate samples of k-hop
    neighbourhoods in the graph and to return to this class the
    features from the sampled neighbourhoods.

    Args:
        generator: GraphSAGENodeGenerator or HinSAGENodeGenerator
            The generator object containing the graph information.
        ids: list
            A list of the node_ids to be used as head-nodes in the
            downstream task.
        targets: list, optional (default=None)
            A list of targets or labels to be used in the downstream
            class.
    """

    def __init__(self, generator, ids, targets=None):
        # Check that ids is an iterable
        if not is_real_iterable(ids):
            raise TypeError("IDs must be an iterable or numpy array of graph node IDs")

        # Check targets is iterable & has the correct length
        if targets is not None:
            if not is_real_iterable(targets):
                raise TypeError("Targets must be None or an iterable or numpy array ")
            if len(ids) != len(targets):
                raise ValueError(
                    "The length of the targets must be the same as the length of the ids"
                )

        # Infer head_node_type
        # TODO: Generalize to multiple head node target types?
        head_node_types = set([generator.schema.get_node_type(n) for n in ids])
        if len(head_node_types) > 1:
            raise ValueError(
                "Only a single head node type is currently supported for HinSAGE models"
            )
        head_node_type = head_node_types.pop()

        # Store the generator to draw samples from graph
        self.generator = generator
        self.ids = list(ids)
        self.targets = targets
        self.data_size = len(self.ids)

        # Save head node type and generate sampling schema
        self.head_node_types = [head_node_type]
        self._sampling_schema = generator.schema.sampling_layout(
            self.head_node_types, generator.num_samples
        )

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(self.data_size / self.generator.batch_size))

    def __getitem__(self, batch_num):
        """
        Generate one batch of data

        Args:
            batch_num (int): number of a batch

        Returns:
            batch_feats (list): Node features for nodes and neighbours sampled from a
                batch of the supplied IDs
            batch_targets (list): Targets/labels for the batch.

        """
        start_idx = self.generator.batch_size * batch_num
        end_idx = start_idx + self.generator.batch_size

        if start_idx >= self.data_size:
            raise IndexError("Mapper: batch_num larger than length of data")
        # print("Fetching batch {} [{}]".format(batch_num, start_idx))

        # Get head nodes
        head_ids = self.ids[start_idx:end_idx]

        # Get targets for nodes
        if self.targets is None:
            batch_targets = None
        else:
            batch_targets = self.targets[start_idx:end_idx]

        # Get sampled nodes
        batch_feats = self.generator.sample_features(head_ids, self._sampling_schema)

        return batch_feats, batch_targets


class GraphSAGENodeGenerator:
    """A data generator for node prediction with Homogeneous GraphSAGE models

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the `.flow(...)` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Example:

    ```
        G_generator = GraphSAGENodeGenerator(G, 50, [10,10])
        train_data_gen = G_generator.flow(node_ids)
    ```

    Args:
        G (StellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        num_samples (list): The number of samples per layer (hop) to take.
        name (str or None): Name of the generator (optional)
    """

    def __init__(self, G, batch_size, num_samples, seed=None, name=None):
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        self.graph = G
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.name = name

        # Check if the graph has features
        G.check_graph_for_ml()

        # Create sampler for GraphSAGE
        self.sampler = SampledBreadthFirstWalk(G, seed=seed)

        # We need a schema for compatibility with HinSAGE
        self.schema = G.create_graph_schema(create_type_maps=True)

        # Check that there is only a single node type for GraphSAGE
        if len(self.schema.node_types) > 1:
            print(
                "Warning: running homogeneous GraphSAGE on a graph with multiple node types"
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
            A list of the same length as `num_samples` of collected features from
            the sampled nodes of shape:
                `(len(head_nodes), num_sampled_at_layer, feature_size)`
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        node_samples = self.sampler.run(nodes=head_nodes, n=1, n_size=self.num_samples)

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

    def flow(self, node_ids, targets=None):
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

        Args:
            node_ids: an iterable of node IDs
            targets: a 2D array of numeric targets with shape
                `(len(node_ids), target_size)`

        Returns:
            A NodeSequence object to use with the GraphSAGE model
                in Keras methods `fit_generator`, `evaluate_generator`,
                and `predict_generator`
        """
        return NodeSequence(self, node_ids, targets)

    def flow_from_dataframe(self, node_targets):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_targets: a Pandas DataFrame of numeric targets indexed
                by the node ID for that target.

        Returns:
            A NodeSequence object to use with the GraphSAGE model
                in Keras methods `fit_generator`, `evaluate_generator`,
                 and `predict_generator`
        """

        return NodeSequence(self, node_targets.index, node_targets.values)


class HinSAGENodeGenerator:
    """Keras-compatible data mapper for Heterogeneour GraphSAGE (HinSAGE)

     At minimum, supply the StellarGraph, the batch size, and the number of
     node samples for each layer of the HinSAGE model.

     The supplied graph should be a StellarGraph object that is ready for
     machine learning. Currently the model requires node features for all
     nodes in the graph.

     Use the `.flow(...)` method supplying the nodes and (optionally) targets
     to get an object that can be used as a Keras data generator.

     Example:

     ```
         G_generator = HinSAGENodeGenerator(G, 50, [10,10])
         data_gen = G_generator.flow(node_ids)
     ```

    Args:
        G (StellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        num_samples (list): The number of samples per layer (hop) to take.
        name (str or None): Name of the generator (optional)
     """

    def __init__(self, G, batch_size, num_samples, seed=None, name=None):

        self.graph = G
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.name = name

        # We require a StellarGraph
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        G.check_graph_for_ml(features=True)

        # Create sampler for GraphSAGE
        self.sampler = SampledHeterogeneousBreadthFirstWalk(G, seed=seed)

        # Generate schema
        self.schema = G.create_graph_schema(create_type_maps=True)

    def sample_features(self, head_nodes, sampling_schema):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            node_sampling_schema: The sampling schema for the HinSAGE model,
                this is can be generated by the `GraphSchema` object.

        Returns:
            A list of the same length as `num_samples` of collected features from
            the sampled nodes of shape:
                `(len(head_nodes), num_sampled_at_layer, feature_size)`
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

    def flow(self, node_ids, targets=None):
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

        Args:
            node_ids (iterable): The head node IDs
            targets (Numpy array): a 2D array of numeric targets with shape
                `(len(node_ids), target_size)`
            node_type (str): [Optional] The target node type, if not given
                the node type will be inferred from the graph.

        Returns:
            A NodeSequence object to use with the GraphSAGE model
                in Keras methods `fit_generator`, `evaluate_generator`,
                and `predict_generator`
        """
        return NodeSequence(self, node_ids, targets)

    def flow_from_dataframe(self, node_targets):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_targets (Pandas DataFrame): Numeric targets indexed
                by the node ID for that target.
            node_type (str): [Optional] The target node type, if not given
                the node type will be inferred from the graph.

        Returns:
            A NodeSequence object to use with the GraphSAGE model
                in Keras methods `fit_generator`, `evaluate_generator`,
                 and `predict_generator`
        """

        return NodeSequence(self, node_targets.index, node_targets.values)
