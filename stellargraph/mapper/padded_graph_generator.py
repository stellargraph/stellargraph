# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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
from ..core.graph import StellarGraph
from ..core.utils import is_real_iterable, normalize_adj
from ..random import random_state
import numpy as np
from tensorflow.keras.utils import Sequence
from .base import Generator


class PaddedGraphGenerator(Generator):
    """
    A data generator for use with graph classification algorithms.

    The supplied graphs should be :class:`StellarGraph` objects with node features.
    Use the :meth:`flow` method supplying the graph indexes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    This generator supplies the features arrays and the adjacency matrices to a mini-batch Keras
    graph classification model. Differences in the number of nodes are resolved by padding each
    batch of features and adjacency matrices, and supplying a boolean mask indicating which are
    valid and which are padding.

    Args:
        graphs (list): a collection of StellarGraph objects
        name (str): an optional name of the generator
    """

    def __init__(self, graphs, name=None):

        self.node_features_size = None
        self._check_graphs(graphs)

        self.graphs = graphs
        self.name = name

    def _check_graphs(self, graphs):
        for graph in graphs:
            if not isinstance(graph, StellarGraph):
                raise TypeError(
                    f"graphs: expected every element to be a StellarGraph object, found {type(graph).__name__}."
                )

            if graph.number_of_nodes() == 0:
                # an empty graph has no information at all and breaks things like mean pooling, so
                # let's disallow them
                raise ValueError(
                    "graphs: expected every graph to be non-empty, found graph with no nodes"
                )

            # Check that there is only a single node type for GAT or GCN
            node_type = graph.unique_node_type(
                "graphs: expected only graphs with a single node type, found a graph with node types: %(found)s"
            )

            graph.check_graph_for_ml()

            # we require that all graphs have node features of the same dimensionality
            f_dim = graph.node_feature_sizes()[node_type]
            if self.node_features_size is None:
                self.node_features_size = f_dim
            elif self.node_features_size != f_dim:
                raise ValueError(
                    "graphs: expected node features for all graph to have same dimensions,"
                    f"found {self.node_features_size} vs {f_dim}"
                )

    def num_batch_dims(self):
        return 1

    def flow(
        self,
        graphs,
        targets=None,
        symmetric_normalization=True,
        weighted=False,
        batch_size=1,
        name=None,
        shuffle=False,
        seed=None,
    ):
        """
        Creates a generator/sequence object for training, evaluation, or prediction
        with the supplied graph indexes and targets.

        Args:
            graphs (iterable): an iterable of graph indexes in self.graphs or an iterable of :class:`StellarGraph` objects
                for the graphs of interest (e.g., training, validation, or test set nodes).
            targets (2d array, optional): a 2D array of numeric graph targets with shape ``(len(graphs),
                len(targets))``.
            symmetric_normalization (bool, optional): The type of normalization to be applied on the graph adjacency
                matrices. If True, the adjacency matrix is left and right multiplied by the inverse square root of the
                degree matrix; otherwise, the adjacency matrix is only left multiplied by the inverse of the degree
                matrix.
            weighted (bool, optional): if True, use the edge weights from ``G``; if False, treat the
                graph as unweighted.
            batch_size (int, optional): The batch size.
            name (str, optional): An optional name for the returned generator object.
            shuffle (bool, optional): If True the node IDs will be shuffled at the end of each epoch.
            seed (int, optional): Random seed to use in the sequence object.

        Returns:
            A :class:`PaddedGraphSequence` object to use with Keras methods :meth:`fit`, :meth:`evaluate`, and :meth:`predict`

        """
        if targets is not None:
            # Check targets is an iterable
            if not is_real_iterable(targets):
                raise TypeError(
                    f"targets: expected an iterable or None object, found {type(targets).__name__}"
                )

            # Check targets correct shape
            if len(targets) != len(graphs):
                raise ValueError(
                    f"expected targets to be the same length as node_ids, found {len(targets)} vs {len(graphs)}"
                )

        if not isinstance(batch_size, int):
            raise TypeError(
                f"expected batch_size to be integer type, found {type(batch_size).__name__}"
            )

        if batch_size <= 0:
            raise ValueError(
                f"expected batch_size to be strictly positive integer, found {batch_size}"
            )

        graphs_array = np.asarray(graphs)

        if len(graphs_array.shape) == 1:
            graphs_array = graphs_array[:, None]
        elif len(graphs_array.shape) != 2:
            raise ValueError(
                f"graphs: expected a shape of length 1 or 2, found shape {graphs_array.shape}"
            )

        flat_graphs = graphs_array.ravel()

        if isinstance(flat_graphs[0], StellarGraph):
            self._check_graphs(flat_graphs)
            graphs = flat_graphs
            selected_ilocs = np.arange(len(graphs)).reshape(graphs_array.shape)
        else:
            selected_ilocs = graphs_array
            graphs = self.graphs

        return PaddedGraphSequence(
            graphs=graphs,
            selected_ilocs=selected_ilocs,
            targets=targets,
            symmetric_normalization=symmetric_normalization,
            weighted=weighted,
            batch_size=batch_size,
            name=name,
            shuffle=shuffle,
            seed=seed,
        )


class PaddedGraphSequence(Sequence):
    """
    A Keras-compatible data generator for training and evaluating graph classification models.
    Use this class with the Keras methods :meth:`keras.Model.fit`,
        :meth:`keras.Model.evaluate`, and
        :meth:`keras.Model.predict`,

    This class should be created using the `.flow(...)` method of
    :class:`PaddedGraphGenerator`.

    Args:
        graphs (list)): The graphs as StellarGraph objects.
        selected_ilocs (array): an array of indices into ``graphs``, of shape N × K for some N and K.
        targets (np.ndarray, optional): An optional array of graph targets of size (N x C),
            where N is the number of selected graph ilocs and C is the target size (e.g., number of classes.)
        normalize (bool, optional): Specifies whether the adjacency matrix for each graph should
            be normalized or not. The default is True.
        symmetric_normalization (bool, optional): Use symmetric normalization if True, that is left and right multiply
            the adjacency matrix by the inverse square root of the degree matrix; otherwise left multiply the adjacency
            matrix by the inverse of the degree matrix. This parameter is ignored if normalize=False.
        batch_size (int, optional): The batch size. It defaults to 1.
        name (str, optional): An optional name for this generator object.
        shuffle (bool, optional): If True the node IDs will be shuffled at the end of each epoch.
        seed (int, optional): Random seed.
    """

    def __init__(
        self,
        graphs,
        selected_ilocs,
        targets=None,
        normalize=True,
        symmetric_normalization=True,
        weighted=False,
        batch_size=1,
        name=None,
        shuffle=False,
        seed=None,
    ):

        self.name = name
        self.graphs = np.asanyarray(graphs)

        if not isinstance(selected_ilocs, np.ndarray):
            raise TypeError(
                "selected_ilocs: expected a NumPy array, found {type(selected_ilocs).__name__}"
            )

        if not len(selected_ilocs.shape) == 2:
            raise ValueError(
                "selected_ilocs: expected a NumPy array of rank 2, found shape {selected_ilocs.shape}"
            )

        # each row of the input corresponds to a single dataset example, but we want to handle
        # columns as bulk operations, and iterating over the major axis is easier
        self.selected_ilocs = selected_ilocs.transpose()

        self.normalize_adj = normalize
        self.targets = targets
        self.batch_size = batch_size

        if targets is not None:
            if len(selected_ilocs) != len(targets):
                raise ValueError(
                    "expected the number of target values and the number of graph ilocs to be the same length,"
                    f"found {len(selected_ilocs)} graph ilocs and {len(targets)} targets."
                )

            self.targets = np.asanyarray(targets)

        adjacencies = [graph.to_adjacency_matrix(weighted=weighted) for graph in graphs]

        if self.normalize_adj:
            self.normalized_adjs = [
                normalize_adj(
                    adj, symmetric=symmetric_normalization, add_self_loops=True,
                )
                for adj in adjacencies
            ]
        else:
            self.normalize_adjs = adjacencies

        self.normalized_adjs = np.asanyarray(self.normalized_adjs)
        _, self._np_rs = random_state(seed)
        self.shuffle = shuffle

        self.on_epoch_end()

    def _epoch_size(self):
        return self.selected_ilocs.shape[1]

    def __len__(self):
        return int(np.ceil(self._epoch_size() / self.batch_size))

    def _pad_graphs(self, graphs, adj_graphs, max_nodes):
        # pad adjacency and feature matrices to equal the size of those from the largest graph
        features = [
            np.pad(
                graph.node_features(),
                pad_width=((0, max_nodes - graph.number_of_nodes()), (0, 0)),
            )
            for graph in graphs
        ]
        features = np.stack(features)

        for adj in adj_graphs:
            adj.resize((max_nodes, max_nodes))
        adj_graphs = np.stack([adj.toarray() for adj in adj_graphs])

        masks = np.full((len(graphs), max_nodes), fill_value=False, dtype=np.bool)
        for index, graph in enumerate(graphs):
            masks[index, : graph.number_of_nodes()] = True

        # features is array of dimensionality
        #      batch size x N x F
        # masks is array of dimensionality
        #      batch size x N
        # adj_graphs is array of dimensionality
        #      batch size x N x N
        # graph_targets is array of dimensionality
        #      batch size x C
        # where N is the maximum number of nodes for largest graph in the batch, F is
        # the node feature dimensionality, and C is the number of target classes
        return [features, masks, adj_graphs]

    def __getitem__(self, index):

        batch_start, batch_end = index * self.batch_size, (index + 1) * self.batch_size

        batch_ilocs = self.selected_ilocs[:, batch_start:batch_end]
        graphs = self.graphs[batch_ilocs]
        adj_graphs = self.normalized_adjs[batch_ilocs]

        # The number of nodes for the largest graph in the batch. We are going to pad with 0 rows and columns
        # the adjacency and node feature matrices (only the rows in this case) to equal in size the adjacency and
        # feature matrices of the largest graph.
        max_nodes = max(graph.number_of_nodes() for graph in graphs.ravel())

        graph_targets = None
        if self.targets is not None:
            graph_targets = self.targets[batch_start:batch_end]

        padded = [
            self._pad_graphs(g, adj, max_nodes) for g, adj in zip(graphs, adj_graphs)
        ]

        return [output for arrays in padded for output in arrays], graph_targets

    def on_epoch_end(self):
        """
         Shuffle all graphs at the end of each epoch
        """
        if self.shuffle:
            indexes = self._np_rs.permutation(self._epoch_size())
            self.selected_ilocs = self.selected_ilocs[:, indexes]
            if self.targets is not None:
                self.targets = self.targets[indexes]
