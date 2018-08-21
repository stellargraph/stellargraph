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

from stellar.data.stellargraph import StellarGraphBase
import numpy as np
import itertools as it
from typing import AnyStr, Any, List, Tuple, Optional
from keras.utils import Sequence
import operator
from functools import reduce
import time

from stellar.data.explorer import (
    SampledBreadthFirstWalk,
    SampledHeterogeneousBreadthFirstWalk,
)


class GraphSAGELinkMapper(Sequence):
    """Keras-compatible link data mapper for link prediction using Homogeneous GraphSAGE

    Args:
        g: StellarGraph graph.
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
        g: StellarGraphBase,
        ids: List[
            Tuple[Any, Any]
        ],  # allow for node ids to be anything, e.g., str or int
        link_labels: List[Any] or np.ndarray,
        batch_size: int,
        num_samples: List[int],
        name: AnyStr = None,
    ):
        if not isinstance(g, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        # We don't know if we need targets here as we could be used for training or inference
        # TODO: Perhaps we shouldn't do the checks here but somewhere that we know will be the entry point for training or inference?
        # YT: this check is needed to ensure that _get_features(), which calls self.graph.get_feature_for_nodes(), will work
        g.check_graph_for_ml(features=True, supervised=False)

        self.graph = g
        self.sampler = SampledBreadthFirstWalk(g)
        self.num_samples = num_samples
        self.ids = list(ids)
        self.labels = link_labels
        self.data_size = len(self.ids)
        self.batch_size = batch_size
        self.name = name

        # We need the node type, to be able to get features from self.g
        node_types = g.get_node_types()
        if len(node_types) > 1:
            print(
                "Warning: running homogeneous GraphSAGE on a graph with multiple node types"
            )
        self.node_type = node_types.pop()

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
            self.graph.get_feature_for_nodes(layer_nodes, self.node_type)
            for layer_nodes in node_samples
        ]

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [np.reshape(a, (head_size, -1, a.shape[1])) for a in batch_feats]
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
        g: StellarGraph graph.
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

        return head_node_types.pop()

    def __init__(
        self,
        g: StellarGraphBase,
        ids: List[
            Tuple[Any, Any]
        ],  # allow for node ids to be anything, e.g., str or int
        link_labels: List[Any] or np.ndarray,
        batch_size: int,
        num_samples: List[int],
        name: AnyStr = None,
    ):
        self.graph = g
        self.num_samples = num_samples
        self.ids = list(ids)
        self.labels = np.array(link_labels)
        self.data_size = len(self.ids)
        self.batch_size = batch_size
        self.name = name
        self.timeit = False  # used for timing purposes only

        # We require a StellarGraph for this
        if not isinstance(g, StellarGraphBase):
            raise TypeError(
                "Graph must be a StellarGraph or StellarDiGraph to use heterogeneous sampling."
            )

        # We don't know if we need targets here as we could be used for training or inference
        # TODO: Perhaps we shouldn't do the checks here but somewhere that we know will be the entry point for training or inference?
        # YT: this check is needed to ensure that _get_features(), which calls self.graph.get_feature_for_nodes(), will work
        g.check_graph_for_ml(features=True, supervised=False)

        # Generate graph schema
        self.schema = self.graph.create_graph_schema(create_type_maps=True)

        # Get head node types from all src, dst nodes extracted from all links,
        # and make sure there's only one pair of node types:
        self.head_node_types = self._infer_head_node_types()

        # Create sampler for HinSAGE
        self.sampler = SampledHeterogeneousBreadthFirstWalk(g, graph_schema=self.schema)

        self.sampling_schema = self.schema.get_sampling_layout(
            self.head_node_types, num_samples
        )

        self.type_adjacency_list = self.schema.get_type_adjacency_list(
            self.head_node_types, len(self.num_samples)
        )

        # Ensure number of labels matches number of ids
        if link_labels is not None and len(ids) != len(link_labels):
            raise ValueError("Length of link ids must match length of link labels")

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
        # Note the if there are no samples for a node a zero array is returned.
        # Resize features to (batch_size, n_neighbours, feature_size)
        # for each node type (note that we can have different feature size for each node type)
        batch_feats = [
            self.graph.get_feature_for_nodes(layer_nodes, nt)
            for nt, layer_nodes in node_samples
        ]

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [np.reshape(a, (head_size, -1, a.shape[1])) for a in batch_feats]

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

        # Get edges and labels
        edges = self.ids[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]

        # print("Fetching {} batch {} (data size: {})".format(self.name, batch_num, len(edges)))

        # Extract head nodes from edges; recall that each edge is a tuple of 2 nodes, so we are extracting 2 head nodes per edge
        head_nodes = [[e[ii] for e in edges] for ii in range(2)]

        # Get sampled nodes for the subgraphs starting from the (src, dst) head nodes
        # nodes_samples is list of two lists: [[samples for src], [samples for dst]]
        node_samples = []
        if self.timeit:
            t_start = time.time()
        for ii in range(2):
            node_samples.append(
                self.sampler.run(nodes=head_nodes[ii], n=1, n_size=self.num_samples)
            )
        if self.timeit:
            print(
                "batch {} of size {}: node sampling time per node: {} s, total time {} s".format(
                    batch_num,
                    self.batch_size,
                    (time.time() - t_start) / self.batch_size,
                    time.time() - t_start,
                )
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
