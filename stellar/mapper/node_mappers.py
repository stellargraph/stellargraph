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
import networkx as nx
import numpy as np
import itertools as it
from typing import AnyStr, Any, List, Optional, Callable
from keras.utils import Sequence

from stellar.data.explorer import SampledBreadthFirstWalk
from stellar.data.stellargraph import StellarGraphBase


class GraphSAGENodeMapper(Sequence):
    """Keras-compatible data mapper for Homogeneous GraphSAGE

    Args:
        G: NetworkX graph. The nodes must have a "feature" attribute that
            is used as input to the graph layers.
        ids: The node IDs to batch. These are the head nodes which are
             used as the nodes to train or inference and the embeddings
             calculated for these nodes are passed to the downstream task.
             Subgraphs are sampled from all nodes.
        sampler: A sampler instance on graph G.
        batch_size: Size of batch to return.
        num_samples: List of number of samples per layer (hop) to take.
        target_id: Name of target value in node attribute dictionary.
        feature_size: Node feature size (optional)
        name: Name of mapper
    """

    def __init__(
        self,
        G: StellarGraphBase,
        ids: List[Any],
        sampler: Callable[[List[Any]], List[List[Any]]],
        batch_size: int,
        num_samples: List[int],
        target_id: AnyStr = None,
        feature_size: Optional[int] = None,
        name: AnyStr = None,
    ):
        self.G = G
        self.sampler = sampler
        self.num_samples = num_samples
        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.batch_size = batch_size
        self.name = name
        self.label_id = target_id

        # Check correct graph sampler is used
        if not isinstance(sampler, SampledBreadthFirstWalk):
            raise TypeError(
                "Sampler must be an instance of from SampledBreadthFirstWalk"
            )

        # Ensure features are available:
        nodes_have_features = all(
            ["feature" in vdata for v, vdata in G.nodes(data=True)]
        )
        if not nodes_have_features:
            print("Warning: Not all nodes have a 'feature' attribute.")
            print("Warning: This is required for the GraphSAGE mapper.")

        # Check that all nodes have features of the same size
        # Note: if there are no features in the nodes this will be 1!
        feature_sizes = {
            np.size(vdata.get("feature")) for v, vdata in G.nodes(data=True)
        }

        if feature_size:
            self.feature_size = feature_size
        else:
            self.feature_size = int(max(feature_sizes))
            if len(feature_sizes) > 1:
                print("Warning: feature sizes in nodes inconsistent (using max)")
                print("Found feature sizes: {}".format(feature_sizes))

        if self.feature_size not in feature_sizes:
            print("Found feature sizes: {}".format(feature_sizes))
            raise RuntimeWarning("Specified feature size doesn't match graph features")

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

    def _get_labels(self, head_nodes: List[AnyStr]) -> np.ndarray:
        """
        Collects the labels of the head nodes. They are assumed to be stored as
        node attributes with the key self.label_id
        Args:
            head_nodes: Nodes to get labels for.

        Returns:
            An array of labels.
        """
        # Get labels for each node in node_samples
        batch_labels = [self.G.node[v].get(self.label_id) for v in head_nodes]
        return np.array(batch_labels)

    def __getitem__(self, batch_num: int) -> [List[np.ndarray], List[np.ndarray]]:
        "Generate one batch of data"
        start_idx = self.batch_size * batch_num
        end_idx = start_idx + self.batch_size

        if start_idx >= self.data_size:
            raise IndexError("Mapper: batch_num larger than length of data")
        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # Get head nodes
        head_nodes = self.ids[start_idx:end_idx]

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
        batch_labels = self._get_labels(head_nodes) if self.label_id else None

        return batch_feats, batch_labels


class HinSAGENodeMapper(Sequence):
    """Keras-compatible data mapper for Heterogeneous GraphSAGE

    Args:
        G: NetworkX graph. The nodes must have a "feature" attribute that
            is used as input to the graph layers.
        ids: The node IDs to batch. These are the head nodes which are
             used as the nodes to train or inference and the embeddings
             calculated for these nodes are passed to the downstream task.
             Subgraphs are sampled from all nodes.
        sampler: A sampler instance on graph G.
        batch_size: Size of batch to return.
        num_samples: List of number of samples per layer (hop) to take.
        target_id: Name of target value in node attribute dictionary.
        feature_size: Node feature size (optional)
        name: Name of mapper
    """

    def __init__(
        self,
        G: StellarGraphBase,
        ids: List[Any],
        sampler: Callable[[List[Any]], List[List[Any]]],
        batch_size: int,
        num_samples: List[int],
        target_id: AnyStr = None,
        feature_size: Optional[int] = None,
        name: AnyStr = None,
    ):
        self.G = G
        self.sampler = sampler
        self.num_samples = num_samples
        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.batch_size = batch_size
        self.name = name
        self.label_id = target_id

        # Check correct graph sampler is used
        if not isinstance(sampler, SampledBreadthFirstWalk):
            raise TypeError(
                "Sampler must be an instance of from SampledBreadthFirstWalk"
            )

        # Ensure features are available:
        nodes_have_features = all(
            ["feature" in vdata for v, vdata in G.nodes(data=True)]
        )
        if not nodes_have_features:
            print("Warning: Not all nodes have a 'feature' attribute.")
            print("Warning: This is required for the GraphSAGE mapper.")

        # Check that all nodes have features of the same size
        # Note: if there are no features in the nodes this will be 1!
        feature_sizes = {
            np.size(vdata.get("feature")) for v, vdata in G.nodes(data=True)
        }

        if feature_size:
            self.feature_size = feature_size
        else:
            self.feature_size = int(max(feature_sizes))
            if len(feature_sizes) > 1:
                print("Warning: feature sizes in nodes inconsistent (using max)")
                print("Found feature sizes: {}".format(feature_sizes))

        if self.feature_size not in feature_sizes:
            print("Found feature sizes: {}".format(feature_sizes))
            raise RuntimeWarning("Specified feature size doesn't match graph features")

        # Get graph schema
        gs = G.create_graph_schema()
        n_hops = len(num_samples)
        head_node_types = {gs.node_type(n) for n in ids}

        # Currently raise an error if head nodes don't have the same type
        if len(head_node_types) != 1:
            raise ValueError("Expected all head nodes to have the same type")

        # Get sampling tree
        self.sampling_tree = gs.create_type_tree(list(head_node_types), n_hops)

        # Get sampling schema
        self.schema = gs.schema

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

    def _get_labels(self, head_nodes: List[AnyStr]) -> np.ndarray:
        """
        Collects the labels of the head nodes. They are assumed to be stored as
        node attributes with the key self.label_id
        Args:
            head_nodes: Nodes to get labels for.

        Returns:
            An array of labels.
        """
        # Get labels for each node in node_samples
        batch_labels = [self.G.node[v].get(self.label_id) for v in head_nodes]
        return np.array(batch_labels)

    def __getitem__(self, batch_num: int) -> [List[np.ndarray], List[np.ndarray]]:
        "Generate one batch of data"
        start_idx = self.batch_size * batch_num
        end_idx = start_idx + self.batch_size

        if start_idx >= self.data_size:
            raise IndexError("Mapper: batch_num larger than length of data")
        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # Get head nodes
        head_nodes = self.ids[start_idx:end_idx]

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
        batch_labels = self._get_labels(head_nodes) if self.label_id else None

        return batch_feats, batch_labels
