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
from stellar.data.stellargraph import StellarGraphBase
import numpy as np
import itertools as it
from typing import AnyStr, Any, List, Tuple, Optional, Dict
from keras.utils import Sequence
import operator
from functools import reduce

from stellar.data.explorer import (
    SampledBreadthFirstWalk,
    SampledHeterogeneousBreadthFirstWalk,
)


class GraphSAGELinkMapper(Sequence):
    """Keras-compatible link data mapper for link prediction using Homogeneous GraphSAGE

    Args:
        g: StellarGraph or NetworkX graph. The graph nodes must have a "feature" attribute that
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
        g: StellarGraphBase or nx.Graph,
        ids: List[
            Tuple[Any, Any]
        ],  # allow for node ids to be anything, e.g., str or int
        link_labels: List[Any] or np.ndarray,
        batch_size: int,
        num_samples: List[int],
        feature_size: Optional[int] = None,
        name: AnyStr = None,
    ):
        self.g = g
        self.sampler = SampledBreadthFirstWalk(g)
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
            ["feature" in vdata for v, vdata in self.g.nodes(data=True)]
        )
        if not nodes_have_features:
            print("Warning: Not all nodes have a 'feature' attribute.")
            print("Warning: This is required for the GraphSAGE mapper.")

        # Check that all nodes have features of the same size
        # Note: if there are no features in the nodes this will be 1!
        feature_sizes = {
            np.size(vdata.get("feature")) for v, vdata in self.g.nodes(data=True)
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
            [self.g.node[v].get("feature") for v in layer_nodes]
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
        """
        Generate one batch of data for links as (node_src, node_dst) pairs

        Args:
            batch_num: number of a batch

        Returns:
            batch_feats: node features for 2 sampled subgraphs with head nodes being node_src, node_dst extracted from links in the batch
            batch_labels: link labels

        """
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

        # Extract head nodes from edges; recall that each edge is a tuple of 2 nodes, so we are extracting 2 head nodes per edge
        head_nodes = [[e[ii] for e in edges] for ii in range(2)]

        # Get sampled nodes for the subgraphs for the head nodes
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

            # Get features for the sampled nodes
            batch_feats[ii] = self._get_features(nodes_per_hop, len(head_nodes[ii]))

        # re-pack features into a list where source, target feats alternate, to feed into GraphSAGE model with
        # (node_src, node_dst) input sockets:
        batch_feats = [
            feats for ab in zip(batch_feats[0], batch_feats[1]) for feats in ab
        ]

        return batch_feats, batch_labels


class HinSAGELinkMapper(Sequence):
    """Keras-compatible link data mapper for link prediction using Heterogeneous GraphSAGE (HinSAGE)

    Notes:
         We don't need to pass link_type (target link type) to the link mapper, considering that:
            1. The mapper actually only cares about (src,dst) node types, and these can be inferred from the passed
                link ids (although this might be expensive, as it requires parsing the links ids passed - yet only once)
            2. It's possible to do link prediction on a graph where that link type is completely removed from the graph
                (e.g., "same_as" links in ER)

    Args:
        g: StellarGraph or NetworkX graph. The graph nodes must have a "feature" attribute that
            is used as input to the HinSAGE model.
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
        feature_size_by_type: Node feature size for each node type in provided links (optional)
        name: Name of mapper
    """

    def _infer_head_node_types(self):
        """Get head node types from all src, dst nodes extracted from all links in self.ids"""
        head_node_types = []
        for src, dst in self.ids:  # loop over all edges in self.ids
            head_node_types.append(
                tuple(self.schema.get_node_type(v) for v in (src, dst))
            )
        head_node_types = list(set(head_node_types))

        assert (
            len(head_node_types) == 1
        ), "All (src,dst) node types for inferred links must be of the same type!"

        # assert head_node_types[0] != ('',''), "Head node types should not be empty"

        return head_node_types[0]

    def __init__(
        self,
        g: StellarGraphBase,
        ids: List[
            Tuple[Any, Any]
        ],  # allow for node ids to be anything, e.g., str or int
        link_labels: List[Any] or np.ndarray,
        batch_size: int,
        num_samples: List[int],
        feature_size_by_type: Optional[Dict[AnyStr, int]] = None,
        name: AnyStr = None,
    ):
        self.g = g
        self.num_samples = num_samples
        self.ids = list(ids)
        self.labels = link_labels
        self.data_size = len(self.ids)
        self.batch_size = batch_size
        self.name = name

        # We require a StellarGraph for this
        if not isinstance(g, StellarGraphBase):
            raise TypeError(
                "Graph must be a StellarGraph or StellarDiGraph to use heterogeneous sampling."
            )

        # Generate graph schema
        self.schema = self.g.create_graph_schema(create_type_maps=True)

        # Get head node types from all src, dst nodes extracted from all links,
        # and make sure there's only one pair of node types:
        self.head_node_types = self._infer_head_node_types()

        # Create sampler for HinSAGE
        self.sampler = SampledHeterogeneousBreadthFirstWalk(g)

        self.sampling_schema = self.schema.get_sampling_layout(
            self.head_node_types, num_samples
        )

        self.type_adjacency_list = self.schema.get_type_adjacency_list(
            self.head_node_types, len(self.num_samples)
        )

        # Ensure number of labels matches number of ids
        if link_labels is not None and len(ids) != len(link_labels):
            raise ValueError("Length of link ids must match length of link labels")

        # If feature size is specified, skip checks
        if feature_size_by_type is None:
            self.feature_size_by_type = {}
            for nt in self.schema.node_types:
                feature_sizes = {
                    # np.size(vdata["feature"]) if "feature" in vdata else None    # YT: this was a bug! We need np.shape(vdata["feature"])[1] instead
                    np.shape(vdata["feature"])[1] if "feature" in vdata else None
                    for v, vdata in self.g.nodes(data=True)
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
        # for each node type (note that we can have different feature size for each node type)
        batch_feats = [
            np.reshape(
                [
                    np.zeros(self.feature_size_by_type[nt])
                    if v is None
                    else self.g.node[v].get("feature")
                    for v in layer_nodes
                ],
                (head_size, -1, self.feature_size_by_type[nt]),
            )
            for nt, layer_nodes in node_samples
        ]

        return batch_feats

    def __getitem__(self, batch_num: int):
        """
        Generate one batch of data for links as (node_src, node_dst) pairs

        Args:
            batch_num: number of a batch

        Returns:
            batch_feats: node features for 2 sampled subgraphs with head nodes being node_src, node_dst extracted from links in the batch
            batch_labels: link labels

        """
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

        # Extract head nodes from edges; recall that each edge is a tuple of 2 nodes, so we are extracting 2 head nodes per edge
        head_nodes = [[e[ii] for e in edges] for ii in range(2)]

        # Get sampled nodes for the subgraphs starting from the (src, dst) head nodes
        # nodes_samples is list of two lists: [[samples for src], [samples for dst]]
        node_samples = []
        for ii in range(2):
            node_samples.append(
                self.sampler.run(nodes=head_nodes[ii], n=1, n_size=self.num_samples)
            )

        # Reshape node samples to the required format for the HinSAGE model
        # This requires grouping the sampled nodes by edge type and in order
        nodes_by_type = []
        for ii in range(2):
            nodes_by_type.append(
                [
                    (
                        nt,
                        reduce(
                            operator.concat,
                            (
                                samples[ks]
                                for samples in node_samples[ii]
                                for ks in indices
                            ),
                            [],
                        ),
                    )
                    for nt, indices in self.sampling_schema[ii]
                ]
            )

        # Interlace the two lists, nodes_by_type[0] (for src head nodes) and nodes_by_type[1] (for dst head nodes)
        nodes_by_type = [
            tuple((ab[0][0], reduce(operator.concat, (ab[0][1], ab[1][1]))))
            for ab in zip(nodes_by_type[0], nodes_by_type[1])
        ]

        batch_feats = self._get_features(nodes_by_type, len(edges))

        return batch_feats, batch_labels
