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
import functools as ft

from typing import AnyStr, Any, List, Tuple, Dict, Optional, Callable

from keras import backend as K
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical


class GraphSAGENodeMapper(Sequence):
    """Keras-compatible data mapper for Homogeneous GraphSAGE

    Args:
        G: NetworkX graph. The nodes must have a "feature" attribute that
            is used as input to the graph layers.
        ids: The node IDs to batch. These are the head nodes which are used
            in the training, subgraphs will still be sampled from all nodes.
        sampler: A sampler instance on graph G.
        batch_size: Size of batch to return.
        target_id: Name of target value in node attribute dictionary.
        feature_size: Node feature size (optional)
        name: Name of mapper
    """

    def __init__(
        self,
        G: nx.Graph,
        ids: List[Any],
        sampler: Callable[[List[Any]], List[List[Any]]],
        batch_size: int,
        target_id: AnyStr = None,
        feature_size: Optional[int] = None,
        name: AnyStr = None,
    ):
        self.G = G
        self.sampler = sampler
        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.batch_size = batch_size
        self.name = name
        self.label_id = target_id

        # Check that all nodes have features of the same size
        feature_sizes = {np.size(G.nodes[v].get("feature")) for v in G.nodes}

        if feature_size:
            self.feature_size = feature_size
        else:
            self.feature_size = int(max(feature_sizes))
            if len(feature_sizes) > 1:
                print("Warning: feature sizes in nodes inconsistent (using max)")
                print("Found feature sizes: {}".format(feature_sizes))

        if self.feature_size not in feature_sizes:
            raise RuntimeWarning("Specified feature size doesn't match graph features")

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(self.data_size / self.batch_size))

    def _get_features(
        self, node_samples: List[List[AnyStr]], head_size: int
    ) -> List[np.ndarray]:
        # Create features and node indices if required
        batch_feats = [
            [self.G.nodes[v].get("feature") for v in layer_nodes]
            if len(layer_nodes) > 0
            else np.zeros((head_size, self.feature_size))
            for layer_nodes in node_samples
        ]

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [
            np.reshape(a, (head_size, -1, self.feature_size)) for a in batch_feats
        ]
        return batch_feats

    def _get_labels(self, head_nodes: List[AnyStr]) -> List[Any]:
        # Get labels for each node in node_samples
        batch_labels = [self.G.nodes[v].get(self.label_id) for v in head_nodes]
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
        node_samples = self.sampler(head_nodes)

        # Get features and labels
        batch_feats = self._get_features(node_samples, len(head_nodes))
        batch_labels = self._get_labels(head_nodes) if self.label_id else None
        return batch_feats, batch_labels
