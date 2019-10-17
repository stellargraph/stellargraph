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
Generators that create batches of data from a machine-learnign ready graph
for link prediction/link attribute inference problems using GraphSAGE, HinSAGE and Attri2Vec.

"""
__all__ = [
    "LinkSequence",
    "OnDemandLinkSequence",
    "GraphSAGELinkGenerator",
    "HinSAGELinkGenerator",
    "Attri2VecLinkGenerator",
]

import random
import operator
import numpy as np
import itertools as it
import operator
import collections
from functools import reduce
import threading

from tensorflow import keras
from tensorflow.keras.utils import Sequence
from stellargraph.core.graph import StellarGraphBase
from stellargraph.data.explorer import (
    SampledBreadthFirstWalk,
    SampledHeterogeneousBreadthFirstWalk,
    UniformRandomWalk,
)
from ..core.utils import is_real_iterable


from stellargraph.data.unsupervised_sampler import UnsupervisedSampler
from stellargraph.core.utils import is_real_iterable


class LinkSequence(Sequence):
    """
    Keras-compatible data generator to use with Keras methods :meth:`keras.Model.fit_generator`,
    :meth:`keras.Model.evaluate_generator`, and :meth:`keras.Model.predict_generator`
    This class generates data samples for link inference models
    and should be created using the :meth:`flow` method of
    :class:`GraphSAGELinkGenerator` or :class:`HinSAGELinkGenerator` or :class:`Attri2VecLinkGenerator`.
    Args:
        generator: An instance of :class:`GraphSAGELinkGenerator` or :class:`HinSAGELinkGenerator` or 
        :class:`Attri2VecLinkGenerator`.
        ids (list or iterable): Link IDs to batch, each link id being a tuple of (src, dst) node ids.
            (The graph nodes must have a "feature" attribute that is used as input to the GraphSAGE/Attri2Vec model.)
            These are the links that are to be used to train or inference, and the embeddings
            calculated for these links via a binary operator applied to their source and destination nodes,
            are passed to the downstream task of link prediction or link attribute inference.
            The source and target nodes of the links are used as head nodes for which subgraphs are sampled.
            The subgraphs are sampled from all nodes.
        targets (list or iterable): Labels corresponding to the above links, e.g., 0 or 1 for the link prediction problem.
        shuffle (bool): If True (default) the ids will be randomly shuffled every epoch.
    """

    def __init__(self, generator, ids, targets=None, shuffle=True):
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
            self.targets = np.asanyarray(targets)
        else:
            self.targets = None

        # Ensure number of labels matches number of ids
        if targets is not None and len(ids) != len(targets):
            raise ValueError("Length of link ids must match length of link targets")

        if (
            isinstance(generator, GraphSAGELinkGenerator)
            or isinstance(generator, HinSAGELinkGenerator)
            or isinstance(generator, Attri2VecLinkGenerator)
        ):
            self.generator = generator
        else:
            raise TypeError(
                "({}) GraphSAGELinkGenerator, HinSAGELinkGenerator or Attri2VecLinkGenerator is required.".format(
                    type(self).__name__
                )
            )

        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.shuffle = shuffle

        # Shuffle the IDs to begin
        self.on_epoch_end()

        if isinstance(self.generator, GraphSAGELinkGenerator) or isinstance(
            self.generator, HinSAGELinkGenerator
        ):
            # Get head node types from all src, dst nodes extracted from all links,
            # and make sure there's only one pair of node types:
            self.head_node_types = self._infer_head_node_types(generator.schema)

            self._sampling_schema = generator.schema.sampling_layout(
                self.head_node_types, generator.num_samples
            )

            self.type_adjacency_list = generator.schema.type_adjacency_list(
                self.head_node_types, len(generator.num_samples)
            )

    def _infer_head_node_types(self, schema):
        """Get head node types from all src, dst nodes extracted from all links in self.ids"""
        head_node_types = []
        for src, dst in self.ids:  # loop over all edges in self.ids
            head_node_types.append(tuple(schema.get_node_type(v) for v in (src, dst)))
        head_node_types = list(set(head_node_types))

        if len(head_node_types) != 1:
            raise RuntimeError(
                "All (src,dst) node types for inferred links must be of the same type!"
            )

        return head_node_types.pop()

    def __len__(self):
        """Denotes the number of batches per epoch"""
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
        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # The ID indices for this batch
        batch_indices = self.indices[start_idx:end_idx]

        # Get head (root) nodes for links
        head_ids = [self.ids[ii] for ii in batch_indices]

        # Get targets for nodes
        batch_targets = None if self.targets is None else self.targets[batch_indices]

        if isinstance(self.generator, GraphSAGELinkGenerator) or isinstance(
            self.generator, HinSAGELinkGenerator
        ):
            # Get sampled nodes for GraphSAGELinkGenerator and HinSAGELinkGenerator
            batch_feats = self.generator.sample_features(
                head_ids, self._sampling_schema
            )
        else:
            # Get sampled nodes for Attri2VecLinkGenerator
            batch_feats = self.generator.sample_features(head_ids)

        return batch_feats, batch_targets

    def on_epoch_end(self):
        """
        Shuffle all link IDs at the end of each epoch
        """
        self.indices = list(range(self.data_size))
        if self.shuffle:
            random.shuffle(self.indices)


class OnDemandLinkSequence(Sequence):
    """
    Keras-compatible data generator to use with Keras methods :meth:`keras.Model.fit_generator`,
    :meth:`keras.Model.evaluate_generator`, and :meth:`keras.Model.predict_generator`

    This class generates data samples for link inference models
    and should be created using the :meth:`flow` method of
    :class:`GraphSAGELinkGenerator` or :class:`Attri2VecLinkGenerator`.

    Args:
        generator: An instance of :class:`GraphSAGELinkGenerator` or 'Attri2VecLinkGenerator'.
        sampler:  An instance of :class:`UnsupervisedSampler` that encapsulates the neighbourhood sampling of a graph.
        The generator method of this class returns `batch_size` of positive and negative samples on demand.
    """

    def __init__(self, generator, walker):

        self.lock = threading.Lock()

        if isinstance(generator, GraphSAGELinkGenerator) or isinstance(
            generator, Attri2VecLinkGenerator
        ):
            self.generator = generator
        else:
            raise TypeError(
                "({}) GraphSAGELinkGenerator or Attri2VecLinkGenerator is required.".format(
                    type(self).__name__
                )
            )

        if isinstance(self.generator, GraphSAGELinkGenerator):

            self.head_node_types = self.generator.schema.node_types * 2

            # YT: we need to have self._sampling_schema for GraphSAGE.build() method to work
            self._sampling_schema = generator.schema.sampling_layout(
                self.head_node_types, generator.num_samples
            )

        if isinstance(walker, UnsupervisedSampler):

            self.walker = walker

            self.data_size = (
                2
                * len(self.walker.nodes)
                * self.walker.length
                * self.walker.number_of_walks
            )  # an estimate of the  upper bound on how many samples are generated in each epoch

            if isinstance(self.generator, GraphSAGELinkGenerator):
                print(
                    "Running GraphSAGELinkGenerator with an estimated {} batches generated on the fly per epoch.".format(
                        round(self.data_size / self.generator.batch_size)
                    )
                )
            else:
                print(
                    "Running Attri2VecLinkGenerator with an estimated {} batches generated on the fly per epoch.".format(
                        round(self.data_size / self.generator.batch_size)
                    )
                )

            self._gen = self.walker.generator(
                self.generator.batch_size
            )  # the generator method from the sampler with the batch-size from the link generator method
        else:
            raise TypeError(
                "({}) UnsupervisedSampler is required.".format(type(self).__name__)
            )

    def __getitem__(self, batch_num):
        """
        Generate one batch of data.

        Args:
            batch_num<int>: number of a batch

        Returns:
            batch_feats<list>: Node features for nodes and neighbours sampled from a
                batch of the supplied IDs
            batch_targets<list>: Targets/labels for the batch.

        """

        if batch_num >= self.__len__():
            raise IndexError(
                "Mapper: batch_num larger than number of esstaimted  batches for this epoch."
            )
        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # Get head nodes and labels
        self.lock.acquire()
        head_ids, batch_targets = next(self._gen)
        self.lock.release()

        if isinstance(self.generator, GraphSAGELinkGenerator):

            if self.head_node_types is None:

                # Get head node types from all src, dst nodes extracted from all links,
                # and make sure there's only one pair of node types:
                self.head_node_types = self._infer_head_node_types(
                    self.generator.schema, head_ids
                )

                self._sampling_schema = self.generator.schema.sampling_layout(
                    self.head_node_types, self.generator.num_samples
                )

                self.type_adjacency_list = self.generator.schema.type_adjacency_list(
                    self.head_node_types, len(self.generator.num_samples)
                )

            # Get sampled nodes for GraphSAGELinkGenerator
            batch_feats = self.generator.sample_features(
                head_ids, self._sampling_schema
            )
        else:
            # Get sampled nodes for Attri2VecLinkGenerator
            batch_feats = self.generator.sample_features(head_ids)

        return batch_feats, batch_targets

    def _infer_head_node_types(self, schema, head_ids):
        """Get head node types from all src, dst nodes extracted from all links in self.ids"""
        head_node_types = []
        for src, dst in head_ids:  # loop over all edges in self.ids
            head_node_types.append(tuple(schema.get_node_type(v) for v in (src, dst)))
        head_node_types = list(set(head_node_types))

        if len(head_node_types) != 1:
            raise RuntimeError(
                "All (src,dst) node types for inferred links must be of the same type!"
            )

        return head_node_types.pop()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.data_size / self.generator.batch_size))


class GraphSAGELinkGenerator:
    """
    A data generator for link prediction with Homogeneous GraphSAGE models

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`.flow` method supplying the nodes and (optionally) targets,
    or an UnsupervisedSampler instance that generates node samples on demand,
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = GraphSageLinkGenerator(G, 50, [10,10])
        train_data_gen = G_generator.flow(edge_ids)

    Args:
        G (StellarGraph): A machine-learning ready graph.
        batch_size (int): Size of batch of links to return.
        num_samples (list): List of number of neighbour node samples per GraphSAGE layer (hop) to take.
        seed (int or str), optional: Random seed for the sampling methods.
        name, optional: Name of generator
    """

    def __init__(self, G, batch_size, num_samples, seed=None, name=None):
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        G.check_graph_for_ml(features=True)

        self.graph = G
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.name = name

        # We need a schema for compatibility with HinSAGE
        self.schema = G.create_graph_schema(create_type_maps=True)

        # The sampler used to generate random samples of neighbours
        self.sampler = SampledBreadthFirstWalk(G, graph_schema=self.schema, seed=seed)

    def sample_features(self, head_links, sampling_schema):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_links: An iterable of edges to perform sampling for.
            sampling_schema: The sampling schema for the model

        Returns:
            A list of the same length as ``num_samples`` of collected features from
            the sampled nodes of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        node_type = sampling_schema[0][0][0]
        head_size = len(head_links)

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

        # Get sampled nodes for the subgraphs for the edges where each edge is a tuple
        # of 2 nodes, so we are extracting 2 head nodes per edge
        batch_feats = []
        for hns in zip(*head_links):
            node_samples = self.sampler.run(nodes=hns, n=1, n_size=self.num_samples)

            nodes_per_hop = get_levels(0, 1, self.num_samples, node_samples)

            # Get features for the sampled nodes
            batch_feats.append(
                [
                    self.graph.get_feature_for_nodes(layer_nodes, node_type)
                    for layer_nodes in nodes_per_hop
                ]
            )

        # Resize features to (batch_size, n_neighbours, feature_size)
        # and re-pack features into a list where source, target feats alternate
        # This matches the GraphSAGE link model with (node_src, node_dst) input sockets:
        batch_feats = [
            np.reshape(feats, (head_size, -1, feats.shape[1]))
            for ab in zip(*batch_feats)
            for feats in ab
        ]
        return batch_feats

    def flow(self, link_ids, targets=None, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied edge IDs and numeric targets.

        The edge IDs are the edges to train or inference on. They are
        expected to by tuples of (source_id, destination_id).

        The targets are an array of numeric targets corresponding to the
        supplied link_ids to be used by the downstream task. They should
        be given in the same order as the list of link IDs.
        If they are not specified (for example, for use in prediction),
        the targets will not be available to the downstream task.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            link_ids (list or UnsupervisedSampler): an iterable of (src_id, dst_id) tuples
                specifying the edges or an UnsupervisedSampler object that has a generator
                method to generate samples on the fly.
            targets (optional, array): a 2D array of numeric targets with shape
                `(len(link_ids), target_size)`
            shuffle (optional, bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A LinkSequence or OnDemandLinkSequence object to use with the GraphSAGE model
            methods :meth:`fit_generator`, :meth:`evaluate_generator`, and :meth:`predict_generator`
        """
        # Pass sampler to on-demand link sequence generation
        if isinstance(link_ids, UnsupervisedSampler):
            return OnDemandLinkSequence(self, link_ids)

        # Otherwise pass iterable (check?) to standard LinkSequence
        elif isinstance(link_ids, collections.Iterable):
            return LinkSequence(self, link_ids, targets, shuffle)

        else:
            raise TypeError(
                "Argument to .flow not recognised. "
                "Please pass a list of samples or a UnsupervisedSampler object."
            )


class HinSAGELinkGenerator:
    """
    A data generator for link prediction with Heterogeneous HinSAGE models

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Note that you don't need to pass link_type (target link type) to the link mapper, considering that:

    * The mapper actually only cares about (src,dst) node types, and these can be inferred from the passed
      link ids (although this might be expensive, as it requires parsing the links ids passed - yet only once)

    * It's possible to do link prediction on a graph where that link type is completely removed from the graph
      (e.g., "same_as" links in ER)


    Example::

        G_generator = HinSAGELinkGenerator(G, 50, [10,10])
        data_gen = G_generator.flow(edge_ids)

    Args:
        g (StellarGraph): A machine-learning ready graph.
        batch_size (int): Size of batch of links to return.
        num_samples (list): List of number of neighbour node samples per GraphSAGE layer (hop) to take.
        seed (int or str), optional: Random seed for the sampling methods.
        name., optional: Name of generator
    """

    def __init__(self, G, batch_size, num_samples, seed=None, name=None):
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        G.check_graph_for_ml(features=True)

        self.graph = G
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.name = name

        # We need a schema for compatibility with HinSAGE
        self.schema = G.create_graph_schema(create_type_maps=True)

        # The sampler used to generate random samples of neighbours
        self.sampler = SampledHeterogeneousBreadthFirstWalk(
            G, graph_schema=self.schema, seed=seed
        )

    def _get_features(self, node_samples, head_size):
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

    def sample_features(self, head_links, sampling_schema):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_links: An iterable of edges to perform sampling for.
            sampling_schema: The sampling schema for the model

        Returns:
            A list of the same length as `num_samples` of collected features from
            the sampled nodes of shape:
                `(len(head_nodes), num_sampled_at_layer, feature_size)`
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        nodes_by_type = []
        for ii in range(2):
            # Extract head nodes from edges: each edge is a tuple of 2 nodes, so we are extracting 2 head nodes per edge
            head_nodes = [e[ii] for e in head_links]

            # Get sampled nodes for the subgraphs starting from the (src, dst) head nodes
            # nodes_samples is list of two lists: [[samples for src], [samples for dst]]
            node_samples = self.sampler.run(
                nodes=head_nodes, n=1, n_size=self.num_samples
            )

            # Reshape node samples to the required format for the HinSAGE model
            # This requires grouping the sampled nodes by edge type and in order
            nodes_by_type.append(
                [
                    (
                        nt,
                        reduce(
                            operator.concat,
                            (samples[ks] for samples in node_samples for ks in indices),
                            [],
                        ),
                    )
                    for nt, indices in sampling_schema[ii]
                ]
            )

        # Interlace the two lists, nodes_by_type[0] (for src head nodes) and nodes_by_type[1] (for dst head nodes)
        nodes_by_type = [
            tuple((ab[0][0], reduce(operator.concat, (ab[0][1], ab[1][1]))))
            for ab in zip(nodes_by_type[0], nodes_by_type[1])
        ]

        batch_feats = self._get_features(nodes_by_type, len(head_links))

        return batch_feats

    def flow(self, link_ids, targets=None, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied edge IDs and numeric targets.

        The edge IDs are the edges to train or inference on. They are
        expected to by tuples of (source_id, destination_id).

        The targets are an array of numeric targets corresponding to the
        supplied link_ids to be used by the downstream task. They should
        be given in the same order as the list of link IDs.
        If they are not specified (for example, for use in prediction),
        the targets will not be available to the downstream task.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            link_ids: an iterable of (src_id, dst_id) tuples specifying the
                edges.
            targets: a 2D array of numeric targets with shape
                ``(len(link_ids), target_size)``
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A LinkSequence object to use with the GraphSAGE model
            methods :meth:`fit_generator`, :meth:`evaluate_generator`, and :meth:`predict_generator`
        """
        if not isinstance(link_ids, collections.Iterable):
            raise TypeError(
                "Argument to .flow not recognised. "
                "Please pass a list of samples or a UnsupervisedSampler object."
            )

        return LinkSequence(self, link_ids, targets, shuffle)


class Attri2VecLinkGenerator:
    """
    A data generator for context node prediction with the attri2vec model.

    At minimum, supply the StellarGraph and the batch size.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`.flow` method supplying the nodes and targets,
    or an UnsupervisedSampler instance that generates node samples on demand,
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = Attri2VecLinkGenerator(G, 50)
        train_data_gen = G_generator.flow(edge_ids, edge_labels)

    Args:
        G (StellarGraph): A machine-learning ready graph.
        batch_size (int): Size of batch of links to return.
        name, optional: Name of generator.
    """

    def __init__(self, G, batch_size, name=None):
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        # Check if the graph has features
        G.check_graph_for_ml(features=True)

        self.graph = G
        self.batch_size = batch_size
        self.name = name

    def sample_features(self, head_links):
        """
        Sample content features of the target nodes and the ids of the context nodes
        and return these as a list of feature arrays for the attri2vec algorithm.

        Args:
            head_links: An iterable of edges to perform sampling for.

        Returns:
            A list of feature arrays, with each element being the feature of a
            target node and the id of the corresponding context node.
        """

        target_ids = [head_link[0] for head_link in head_links]
        context_ids = [head_link[1] for head_link in head_links]
        target_feats = self.graph.get_feature_for_nodes(target_ids)
        context_feats = self.graph.get_index_for_nodes(context_ids)
        batch_feats = [target_feats, np.array(context_feats)]

        return batch_feats

    def flow(self, link_ids, targets=None):
        """
        Creates a generator/sequence object for training
        with the supplied edge IDs and numeric targets.

        The edge IDs are the edges to train. They are
        expected to by tuples of (source_id, destination_id).

        The targets are an array of numeric targets corresponding to the
        supplied link_ids to be used by the attri2vec context node prediction task. 
        They should be given in the same order as the list of link IDs.

        Args:
            link_ids (list or UnsupervisedSampler): an iterable of (src_id, dst_id) tuples
                specifying the edges or an UnsupervisedSampler object that has a generator
                method to generate samples on the fly.
            targets (optional, array): a 2D array of numeric targets with shape
                `(len(link_ids), target_size)`.

        Returns:
            An OnDemandLinkSequence object to use with the attri2vec model.
        """
        # Pass sampler to on-demand link sequence generation
        if isinstance(link_ids, UnsupervisedSampler):
            return OnDemandLinkSequence(self, link_ids)

        # Otherwise pass iterable to standard LinkSequence
        elif isinstance(link_ids, collections.Iterable):
            return LinkSequence(self, link_ids, targets, shuffle=True)

        else:
            raise TypeError(
                "Argument to .flow not recognised. "
                "Please pass a list of samples or a UnsupervisedSampler object."
            )
