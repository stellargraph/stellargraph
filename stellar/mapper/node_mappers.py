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
import collections
import operator
from functools import reduce

import networkx as nx
import numpy as np
import itertools as it
from typing import AnyStr, Any, List, Optional
from keras.utils import Sequence

from stellar.data.explorer import (
    SampledBreadthFirstWalk,
    SampledHeterogeneousBreadthFirstWalk,
)
from stellar.data.stellargraph import StellarGraphBase


class GraphSAGENodeMapper(Sequence):
    """Keras-compatible data mapper for Homogeneous GraphSAGE

    At minimum, supply the graph, the head-node IDs, the batch size,
    and the number of node samples for each layer.

    Currently, all graph nodes must have a "feature" attribute that is
    a numpy array to be used as the input features to the graph layers.
    If the feature size is not specified all nodes are looked at to
    and the size of the feature attributes taken to be the feature size.
    Tt is assumed that the features for all nodes are the same shape.

    The head-node IDs are the nodes to train or inference on: the embeddings
    calculated for these nodes are passed to the downstream task.

    Note that subgraphs around each head-node are sampled from all nodes in
    the graph.

    The targets are a list of numeric targets for the head-nodes to be used
    in the downstream task. They should be given in the same order as the
    list of head-node IDs. If they are not specified, None will be returned
    in place of the batch targets.

    Args:
        G: StellarGraph or NetworkX graph.
        ids: The head-node IDs to train/inference on.
        batch_size: Size of batch to return.
        num_samples: List of number of samples per layer (hop) to take.
        targets: List of numeric targets for supervised models these will
            be extracted from the graph if not given.
        name: Name of mapper (optional)
    """

    def __init__(
        self,
        G: StellarGraphBase,
        ids: List[Any],
        batch_size: int,
        num_samples: List[int],
        targets: bool = True,
        name: AnyStr = None,
        # TODO: add a check=True argument, toggling the checks for node ids and features
    ):
        self.graph = G
        self.num_samples = num_samples
        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.batch_size = batch_size
        self.name = name
        self.use_target = targets

        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        # We don't know if we need targets here as we could be used for training or inference
        # TODO: Perhaps we shouldn't do the checks here but somewhere that we know
        # TODO: will be the entry point for training or inference?
        G.check_graph_for_ml(features=True, supervised=targets)

        # Create sampler for GraphSAGE
        self.sampler = SampledBreadthFirstWalk(G)

        # We need the node types
        node_types = G.get_node_types()
        if len(node_types) > 1:
            print(
                "Warning: running homogeneous GraphSAGE on a graph with multiple node types"
            )
        self._node_type = node_types.pop()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, batch_num: int) -> [List[np.ndarray], List[np.ndarray]]:
        "Generate one batch of data"
        start_idx = self.batch_size * batch_num
        end_idx = start_idx + self.batch_size

        if start_idx >= self.data_size:
            raise IndexError("Mapper: batch_num larger than length of data")
        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # Get head nodes
        head_nodes = self.ids[start_idx:end_idx]

        # Get targets for nodes
        if self.use_target:
            batch_targets = self.graph.get_target_for_nodes(head_nodes, self._node_type)
        else:
            batch_targets = None

        # Get sampled nodes
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

        # Get features and labels
        batch_feats = [
            self.graph.get_feature_for_nodes(layer_nodes, self._node_type)
            for layer_nodes in nodes_per_hop
        ]

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [
            np.reshape(a, (len(head_nodes), -1 if np.size(a) > 0 else 0, a.shape[1]))
            for a in batch_feats
        ]

        return batch_feats, batch_targets


class HinSAGENodeMapper(Sequence):
    """Keras-compatible data mapper for Heterogeneous GraphSAGE (HinSAGE)

    At minimum, supply the graph, the head-node IDs, the batch size,
    and the number of node samples for each layer.

    Currently, all graph nodes must have a "feature" attribute that is
    a numpy array to be used as the input features to the graph layers.
    If the feature size is not specified all nodes are looked at to
    and the size of the feature attributes for each node type is calculated.
    Tt is assumed that the features for all nodes of the same type are the
    same size, but different node types can have different sizes.

    The head-node IDs are the nodes to train or inference on: the embeddings
    calculated for these nodes are passed to the downstream task.
    The head-nodes should all be of the same type, and passed as the
    parameter `node_type`.

    If the `node_type` parameter is not given, the type of the head-nodes
    will be calculated at each batch - this could be time-consuming.

    Note that subgraphs around each head-node are sampled from all nodes in
    the graph.

    The targets are a list of numeric targets for the head-nodes to be used
    in the downstream task. They should be given in the same order as the
    list of head-node IDs. If they are not specified, None will be returned
    in place of the batch targets.

    Args:
        G: StellarGraph or NetworkX graph.
        ids: The head-node IDs to train/inference on.
        batch_size: Size of batch to return.
        num_samples: List of number of samples per layer (hop) to take.
        node_type: The node type of the head-nodes. Currently only a single
                   type is admitted.
        targets: List of numeric targets for supervised models(optional).
        feature_size_by_type: Node feature size for each node type (optional)
        name: Name of mapper (optional)
    """

    def __init__(
        self,
        G: StellarGraphBase,
        ids: List[Any],
        batch_size: int,
        num_samples: List[int],
        node_type: Optional[AnyStr] = None,
        targets: bool = True,
        name: Optional[AnyStr] = None,
    ):
        self.graph = G
        self.num_samples = num_samples
        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.batch_size = batch_size
        self.name = name
        self.targets = targets

        # We require a StellarGraph for this
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        # We don't know if we need targets here as we could be used for training or inference
        # TODO: Perhaps we shouldn't do the checks here but somewhere that we know
        # TODO: will be the entry point for training or inference?
        G.check_graph_for_ml(features=True, supervised=targets)

        # Create sampler for GraphSAGE
        self.sampler = SampledHeterogeneousBreadthFirstWalk(G)

        # Generate schema
        self.schema = G.create_graph_schema(create_type_maps=True)

        # Infer head_node_type
        # TODO: Generalize to multiple head node target types?
        if node_type is None:
            head_node_types = set([self.schema.get_node_type(n) for n in ids])
            if len(head_node_types) > 1:
                raise ValueError(
                    "Only a single head node type is currently supported for HinSAGE models"
                )
            head_node_type = head_node_types.pop()
        elif (
            isinstance(node_type, collections.Hashable)
            and node_type in self.schema.node_types
        ):
            head_node_type = node_type
        else:
            raise ValueError(
                "Target type '{}' not found in graph node types".format(node_type)
            )

        # Save head node type and generate sampling schema
        self._head_node_type = head_node_type
        self._sampling_schema = self.schema.get_sampling_layout(
            [head_node_type], num_samples
        )

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, batch_num: int) -> [List[np.ndarray], List[np.ndarray]]:
        "Generate one batch of data"
        start_idx = self.batch_size * batch_num
        end_idx = start_idx + self.batch_size

        if start_idx >= self.data_size:
            raise IndexError("Mapper: batch_num larger than length of data")
        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # Get batch head nodes
        head_nodes = self.ids[start_idx:end_idx]

        # Get batch targets - if given
        if self.targets:
            batch_targets = self.graph.get_target_for_nodes(
                head_nodes, self._head_node_type
            )
        else:
            batch_targets = None

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
            for nt, indices in self._sampling_schema[0]
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

        return batch_feats, batch_targets

    def get_head_node_type(self):
        """
        Get the head node for this mapper.
        Returns:
            Node type
        """
        return self._head_node_type