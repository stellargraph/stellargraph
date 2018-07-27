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
from typing import AnyStr, Any, List, Optional, Callable, Dict
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
        targets: List of numeric targets for supervised models(optional).
        feature_size: Node feature size (optional)
        name: Name of mapper (optional)
    """

    def __init__(
        self,
        G: StellarGraphBase,
        ids: List[Any],
        batch_size: int,
        num_samples: List[int],
        targets: List[Any] = None,
        feature_size: Optional[int] = None,
        name: AnyStr = None,
    ):
        self.G = G
        self.num_samples = num_samples
        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.batch_size = batch_size
        self.name = name
        self.target_data = targets

        # Create sampler for GraphSAGE
        self.sampler = SampledBreadthFirstWalk(G)

        # Ensure number of targets matches number of ids
        if targets is not None and len(ids) != len(targets):
            raise ValueError("Length of ids must match length of targets")

        # Check that all nodes have features of the same size
        # Note: if there are no features in the nodes this will be 1!
        feature_sizes = {
            np.size(vdata["feature"]) if "feature" in vdata else None
            for v, vdata in G.nodes(data=True)
        }

        # Check all nodes have features
        if None in feature_sizes:
            raise RuntimeError(
                "Not all nodes have a 'feature' attribute: "
                "this is required for the GraphSAGE mapper."
            )

        # Sanity checks on feature sizes
        if feature_size:
            if feature_size not in feature_sizes:
                raise RuntimeWarning(
                    "Specified feature size doesn't match graph features"
                )
            self.feature_size = feature_size

        elif len(feature_sizes) == 1:
            self.feature_size = int(feature_sizes.pop())

        else:
            raise RuntimeError(
                "Feature sizes in nodes inconsistent: "
                "found feature sizes: {}".format(feature_sizes)
            )

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(self.data_size / self.batch_size))

    def _get_features(
        self, node_samples: List[List[AnyStr]], head_size: int
    ) -> List[np.ndarray]:
        """
        Collect features from sampled nodes.
        Args:
            node_samples: A list of lists of node IDs
            head_size: The number of head nodes (typically the batch size).

        Returns:
            A list of numpy arrays that store the features for each head
            node.
        """
        # Create features and node indices if required
        # Note the if there are no samples for a level, a zero array is returned.
        batch_feats = [
            [self.G.node[v].get("feature") for v in layer_nodes]
            if len(layer_nodes) > 0
            else np.zeros((head_size, self.feature_size))
            for layer_nodes in node_samples
        ]

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [
            np.reshape(a, (head_size, -1, self.feature_size)) for a in batch_feats
        ]
        return batch_feats

    def __getitem__(self, batch_num: int) -> [List[np.ndarray], List[np.ndarray]]:
        "Generate one batch of data"
        start_idx = self.batch_size * batch_num
        end_idx = start_idx + self.batch_size

        if start_idx >= self.data_size:
            raise IndexError("Mapper: batch_num larger than length of data")
        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # Get head nodes
        head_nodes = self.ids[start_idx:end_idx]

        if self.target_data is not None:
            batch_targets = self.target_data[start_idx:end_idx]
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
        batch_feats = self._get_features(nodes_per_hop, len(head_nodes))

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
        node_type: AnyStr = None,
        targets: List[Any] = None,
        feature_size_by_type: Optional[Dict[AnyStr, int]] = None,
        name: AnyStr = None,
    ):
        self.G = G
        self.num_samples = num_samples
        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.batch_size = batch_size
        self.name = name
        self.target_data = targets

        # We require a StellarGraph for this
        if not isinstance(G, StellarGraphBase):
            raise TypeError(
                "Graph must be a StellarGraph or StellarDiGraph to use heterogeneous sampling."
            )

        # Create sampler for GraphSAGE
        self.sampler = SampledHeterogeneousBreadthFirstWalk(G)

        # Generate schema
        self.schema = G.create_graph_schema(create_type_maps=True)

        # Pre-generate sampling schema if target type specified
        # TODO: Should we just make this mandatory?
        # TODO: Generalize to multiple head node target types?
        if node_type is None:
            self._sampling_schema = None
        elif (
            isinstance(node_type, collections.Hashable)
            and node_type in self.schema.node_types
        ):
            self._sampling_schema = self.schema.get_sampling_layout(
                [node_type], num_samples
            )
        else:
            raise ValueError(
                "Target type '{}' not found in graph node types".format(node_type)
            )

        # Ensure number of targets matches number of ids
        if targets is not None and len(ids) != len(targets):
            raise ValueError("Length of ids must match length of targets")

        # If feature size is specified, skip checks
        if feature_size_by_type is None:
            self.feature_size_by_type = {}
            for nt in self.schema.node_types:
                feature_sizes = {
                    np.size(vdata["feature"]) if "feature" in vdata else None
                    for v, vdata in G.nodes(data=True)
                    if self.schema.get_node_type(v) == nt
                }

                if None in feature_sizes:
                    raise RuntimeError(
                        "Not all nodes have a 'feature' attribute: "
                        "this is required for the HinSAGE mapper."
                    )

                if len(feature_sizes) > 1:
                    print("Found feature sizes: {}".format(feature_sizes))
                    raise ValueError(
                        "Feature sizes in nodes of type {} is inconsistent".format(nt)
                    )

                self.feature_size_by_type[nt] = feature_sizes.pop()

        else:
            self.feature_size_by_type = feature_size_by_type

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(self.data_size / self.batch_size))

    def _get_features(
        self, node_samples: List[List[AnyStr]], head_size: int
    ) -> List[np.ndarray]:
        """
        Collect features from sampled nodes.
        Args:
            node_samples: A list of lists of node IDs
            head_size: The number of head nodes (typically the batch size).

        Returns:
            A list of numpy arrays that store the features for each head
            node.
        """
        # Create features and node indices if required
        # Note the if there are no samples for a node a zero array is returned.
        # TODO: Generalize this to an arbitrary vector?
        # Resize features to (batch_size, n_neighbours, feature_size)
        # for each node type (we could have different feature size for each node type)
        batch_feats = [
            np.reshape(
                [
                    np.zeros(self.feature_size_by_type[nt])
                    if v is None
                    else self.G.node[v].get("feature")
                    for v in layer_nodes
                ],
                (head_size, -1, self.feature_size_by_type[nt]),
            )
            for nt, layer_nodes in node_samples
        ]

        return batch_feats

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
        if self.target_data is not None:
            batch_targets = self.target_data[start_idx:end_idx]
        else:
            batch_targets = None

        # Get sampling schema for head nodes
        sampling_schema = self._sampling_schema
        if sampling_schema is None:
            head_node_types = set([self.schema.get_node_type(n) for n in head_nodes])
            if len(head_node_types) > 1:
                raise ValueError(
                    "Only a single head node type is currently supported for HinSAGE models"
                )
            sampling_schema = self.schema.get_sampling_layout(
                head_node_types, self.num_samples
            )

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

        # Get features and labels
        batch_feats = self._get_features(nodes_by_type, len(head_nodes))

        return batch_feats, batch_targets
