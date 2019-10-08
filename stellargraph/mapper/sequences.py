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
Sequences to provide input to Keras

"""
__all__ = [
    "NodeSequence",
    "LinkSequence",
    "OnDemandLinkSequence",
    "FullBatchNodeSequence",
    "SparseFullBatchNodeSequence",
]

import warnings
import operator
import random
import numpy as np
import itertools as it
import networkx as nx
import scipy.sparse as sps
import threading
from tensorflow.keras import backend as K
from functools import reduce
from tensorflow.keras.utils import Sequence
from ..data.unsupervised_sampler import UnsupervisedSampler
from ..core.utils import is_real_iterable


class NodeSequence(Sequence):
    """Keras-compatible data generator to use with the Keras
    methods :meth:`keras.Model.fit_generator`, :meth:`keras.Model.evaluate_generator`,
    and :meth:`keras.Model.predict_generator`.

    This class generated data samples for node inference models
    and should be created using the `.flow(...)` method of
    :class:`GraphSAGENodeGenerator` or :class:`HinSAGENodeGenerator`.

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

        # Check all IDs are actually in the graph
        if any(n not in generator.graph for n in ids):
            raise KeyError(
                "Head nodes supplied to generator contain IDs not found in graph"
            )

        # Infer head_node_type
        if (
            generator.schema.node_type_map is None
            or generator.schema.edge_type_map is None
        ):
            raise RuntimeError("Schema must have node and edge type maps.")
        else:
            head_node_types = {generator.schema.get_node_type(n) for n in ids}
        if len(head_node_types) > 1:
            raise ValueError(
                "Only a single head node type is currently supported for HinSAGE models"
            )
        head_node_type = head_node_types.pop()

        # Store the generator to draw samples from graph
        self.generator = generator
        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.shuffle = shuffle

        # Shuffle IDs to start
        self.on_epoch_end()

        # Save head node type and generate sampling schema
        self.head_node_types = [head_node_type]
        ### Experimental; for directed sampling
        num_samples = getattr(generator, "num_samples", [])
        self._sampling_schema = generator.schema.sampling_layout(
            self.head_node_types, num_samples
        )

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
        # print("Fetching batch {} [{}]".format(batch_num, start_idx))

        # The ID indices for this batch
        batch_indices = self.indices[start_idx:end_idx]

        # Get head (root) nodes
        head_ids = [self.ids[ii] for ii in batch_indices]

        # Get corresponding targets
        batch_targets = None if self.targets is None else self.targets[batch_indices]

        # Get sampled nodes
        batch_feats = self.generator.sample_features(head_ids, self._sampling_schema)

        return batch_feats, batch_targets

    def on_epoch_end(self):
        """
        Shuffle all head (root) nodes at the end of each epoch
        """
        self.indices = list(range(self.data_size))
        if self.shuffle:
            random.shuffle(self.indices)


class LinkSequence(Sequence):
    """
    Keras-compatible data generator to use with Keras methods :meth:`keras.Model.fit_generator`,
    :meth:`keras.Model.evaluate_generator`, and :meth:`keras.Model.predict_generator`
    This class generates data samples for link inference models
    and should be created using the :meth:`flow` method of
    :class:`GraphSAGELinkGenerator` or :class:`HinSAGELinkGenerator` .
    Args:
        generator: An instance of :class:`GraphSAGELinkGenerator` or :class:`HinSAGELinkGenerator`.
        ids (list or iterable): Link IDs to batch, each link id being a tuple of (src, dst) node ids.
            (The graph nodes must have a "feature" attribute that is used as input to the GraphSAGE model.)
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

        self.generator = generator
        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.shuffle = shuffle

        # Shuffle the IDs to begin
        self.on_epoch_end()

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
        try:
            head_node_types = []
            for src, dst in self.ids:  # loop over all edges in self.ids
                head_node_types.append(
                    tuple(schema.get_node_type(v) for v in (src, dst))
                )
            head_node_types = list(set(head_node_types))
        except KeyError:
            raise KeyError("All supplied nodes must be in the graph schema")

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

        # Get sampled nodes
        batch_feats = self.generator.sample_features(head_ids, self._sampling_schema)

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
    :class:`GraphSAGELinkGenerator` ` .

    Args:
        generator: An instance of :class:`GraphSAGELinkGenerator`.
        sampler:  An instance of :class:`UnsupervisedSampler` that encapsulates the neighbourhood sampling of a graph.
        The generator method of this class returns `batch_size` of positive and negative samples on demand.
    """

    def __init__(self, generator, walker):

        self.lock = threading.Lock()

        self.generator = generator  # graphlinkgenerator instance

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

            print(
                "Running GraphSAGELinkGenerator with an estimated {} batches generated on the fly per epoch.".format(
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

        # Get sampled nodes
        batch_feats = self.generator.sample_features(head_ids, self._sampling_schema)

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


class FullBatchNodeSequence(Sequence):
    """
    Keras-compatible data generator for for node inference models
    that require full-batch training (e.g., GCN, GAT).
    Use this class with the Keras methods :meth:`keras.Model.fit_generator`,
        :meth:`keras.Model.evaluate_generator`, and
        :meth:`keras.Model.predict_generator`,

    This class should be created using the `.flow(...)` method of
    :class:`FullBatchNodeGenerator`.

    Args:
        features (np.ndarray): An array of node features of size (N x F),
            where N is the number of nodes in the graph, F is the node feature size
        A (np.ndarray or sparse matrix): An adjacency matrix of the graph of size (N x N).
        targets (np.ndarray, optional): An optional array of node targets of size (N x C),
            where C is the target size (e.g., number of classes for one-hot class targets)
        indices (np.ndarray, optional): Array of indices to the feature and adjacency matrix
            of the targets. Required if targets is not None.
    """

    use_sparse = False

    def __init__(self, features, A, targets=None, indices=None):

        if (targets is not None) and (len(indices) != len(targets)):
            raise ValueError(
                "When passed together targets and indices should be the same length."
            )

        # Store features and targets as np.ndarray
        self.features = np.asanyarray(features)
        self.target_indices = np.asanyarray(indices)

        # Convert sparse matrix to dense:
        if sps.issparse(A) and hasattr(A, "toarray"):
            self.A_dense = A.toarray()
        elif isinstance(A, (np.ndarray, np.matrix)):
            self.A_dense = np.asanyarray(A)
        else:
            raise TypeError(
                "Expected input matrix to be either a Scipy sparse matrix or a Numpy array."
            )

        # Reshape all inputs to have batch dimension of 1
        self.features = np.reshape(self.features, (1,) + self.features.shape)
        self.A_dense = self.A_dense.reshape((1,) + self.A_dense.shape)
        self.target_indices = np.reshape(
            self.target_indices, (1,) + self.target_indices.shape
        )

        self.inputs = [self.features, self.target_indices, self.A_dense]

        if targets is not None:
            self.targets = np.asanyarray(targets)
            self.targets = np.reshape(self.targets, (1,) + self.targets.shape)
        else:
            self.targets = None

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.inputs, self.targets


class SparseFullBatchNodeSequence(Sequence):
    """
    Keras-compatible data generator for for node inference models
    that require full-batch training (e.g., GCN, GAT).
    Use this class with the Keras methods :meth:`keras.Model.fit_generator`,
        :meth:`keras.Model.evaluate_generator`, and
        :meth:`keras.Model.predict_generator`,

    This class uses sparse matrix representations to send data to the models,
    and only works with the Keras tensorflow backend. For any other backends,
    use the :class:`FullBatchNodeSequence` class.

    This class should be created using the `.flow(...)` method of
    :class:`FullBatchNodeGenerator`.

    Args:
        features (np.ndarray): An array of node features of size (N x F),
            where N is the number of nodes in the graph, F is the node feature size
        A (sparse matrix): An adjacency matrix of the graph of size (N x N).
        targets (np.ndarray, optional): An optional array of node targets of size (N x C),
            where C is the target size (e.g., number of classes for one-hot class targets)
        indices (np.ndarray, optional): Array of indices to the feature and adjacency matrix
            of the targets. Required if targets is not None.
    """

    use_sparse = True

    def __init__(self, features, A, targets=None, indices=None):

        if (targets is not None) and (len(indices) != len(targets)):
            raise ValueError(
                "When passed together targets and indices should be the same length."
            )

        # Store features and targets as np.ndarray
        self.features = np.asanyarray(features)
        self.target_indices = np.asanyarray(indices)

        # Ensure matrix is in COO format to extract indices
        if sps.isspmatrix(A):
            A = A.tocoo()

        else:
            raise ValueError("Adjacency matrix not in expected sparse format")

        # Convert matrices to list of indices & values
        self.A_indices = np.expand_dims(np.hstack((A.row[:, None], A.col[:, None])), 0)
        self.A_values = np.expand_dims(A.data, 0)

        # Reshape all inputs to have batch dimension of 1
        self.target_indices = np.reshape(
            self.target_indices, (1,) + self.target_indices.shape
        )
        self.features = np.reshape(self.features, (1,) + self.features.shape)
        self.inputs = [
            self.features,
            self.target_indices,
            self.A_indices,
            self.A_values,
        ]

        if targets is not None:
            self.targets = np.asanyarray(targets)
            self.targets = np.reshape(self.targets, (1,) + self.targets.shape)
        else:
            self.targets = None

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.inputs, self.targets
