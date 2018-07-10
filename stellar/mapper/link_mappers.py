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
Mappers to provide input data for link prediction/link attribute inference problems using GraphSAGE and HinSAGE.

"""

import networkx as nx
import numpy as np
import itertools as it
from typing import AnyStr, Any, List, Tuple, Callable, Optional
from keras.utils import Sequence

from stellar.data.explorer import SampledBreadthFirstWalk


class GraphSAGELinkMapper(Sequence):
    """Keras-compatible link data mapper for link prediction using Homogeneous GraphSAGE

    Args:
        G: NetworkX graph. The graph nodes must have a "feature" attribute that
            is used as input to the GraphSAGE model.
        ids: Link IDs to batch, each link id being a tuple of (src, dst) node ids.
            (The graph nodes must have a "feature" attribute that is used as input to the GraphSAGE model.)
            These are the links that are to be used to train or inference, and the embeddings
            calculated for these links via a binary operator applied to their source and destination nodes,
            are passed to the downstream task of link prediction or link attribute inference.
            The source and target nodes of the links are used as head nodes for which subgraphs are sampled.
            The subgraphs are sampled from all nodes.
        link_labels: Labels of the above links, e.g., 0 or 1 for the link prediction problem.
        batch_size: Size of batch of links to return.
        num_samples: List of number of neighbour node samples per GraphSAGE layer (hop) to take.
        feature_size: Node feature size (optional)
        name: Name of mapper
    """

    def __init__(
        self,
        G: nx.Graph,
        ids: List[
            Tuple[Any, Any]
        ],  # allow for node ids to be anything, e.g., str or int
        link_labels: List[Any] or np.ndarray,
        batch_size: int,
        num_samples: List[int],
        feature_size: Optional[int] = None,
        name: AnyStr = None,
    ):
        self.G = G
        self.sampler = SampledBreadthFirstWalk(G)
        self.num_samples = num_samples
        self.ids = list(ids)
        self.labels = link_labels
        self.data_size = len(self.ids)
        self.batch_size = batch_size
        self.name = name

        # Check correct graph sampler is used
        if not isinstance(self.sampler, SampledBreadthFirstWalk):
            raise TypeError(
                "Sampler must be an instance of SampledBreadthFirstWalk class"
            )

        # Ensure features are available:
        nodes_have_features = all(
            ["feature" in vdata for v, vdata in self.G.nodes(data=True)]
        )
        if not nodes_have_features:
            print("Warning: Not all nodes have a 'feature' attribute.")
            print("Warning: This is required for the GraphSAGE mapper.")

        # Check that all nodes have features of the same size
        # Note: if there are no features in the nodes this will be 1!
        feature_sizes = {
            np.size(vdata.get("feature")) for v, vdata in self.G.nodes(data=True)
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

    def __getitem__(self, batch_num: int):
        "Generate one batch of data"
        start_idx = self.batch_size * batch_num
        end_idx = start_idx + self.batch_size

        if start_idx >= self.data_size:
            raise IndexError(
                "{}: batch_num larger than length of data".format(type(self).__name__)
            )

        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # Get edges and labels
        edges = self.ids[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        batch_feats = [[] for ii in range(2)]

        # Extract head nodes from edges; recall that each edge is a tuple of 2 nodes
        head_size = len(edges)
        head_nodes = [[e[ii] for e in edges] for ii in range(2)]

        # Get sampled nodes
        for ii in range(2):
            node_samples = self.sampler.run(
                nodes=head_nodes[ii], n=1, n_size=self.num_samples
            )

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

            # Get features
            batch_feats[ii] = self._get_features(nodes_per_hop, len(head_nodes[ii]))

        # re-pack into a list where source, target feats alternate:
        batch_feats = [
            feats for ab in zip(batch_feats[0], batch_feats[1]) for feats in ab
        ]

        return batch_feats, batch_labels
