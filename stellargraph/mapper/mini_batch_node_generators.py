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
Mappers to provide input data for the graph models in layers.

"""
__all__ = [
    "ClusterNodeGenerator",
]

import random
import copy
import numpy as np
import networkx as nx
from tensorflow.keras.utils import Sequence

from scipy import sparse
from ..core.graph import StellarGraphBase
from ..core.utils import is_real_iterable


class ClusterNodeGenerator:
    """
    A data generator for use with ClusterGCN models on homogeneous graphs.

    The supplied graph G should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features to be available for all
    nodes in the graph.
    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    This generator will supply the features array and the adjacency matrix to a
    mini-batch Keras graph ML model.

    For these algorithms the adjacency matrix requires pre-processing and the
    'method' option should be specified with the correct pre-processing for
    each algorithm. The options are as follows:

    Note: method can only be 'none' and the pre-processing will happen when the data
    is returned. Since there is only one option for method, the user will not be able
    to specify the method in the constructor.

    [1] `W. Chiang, X. Liu, S. Si, Y. Li, S. Bengio, C. Hsieh, 2019 <https://arxiv.org/abs/1905.07953>`_.

    For more information, please see the ClusterGCN demo:
        `<https://github.com/stellargraph/stellargraph/blob/master/demos/>`_

    Args:
        G (StellarGraphBase): a machine-learning StellarGraph-type graph
        k (int): The number of clusters if parameter `clusters` is None. Otherwise it is ignored.
        lam (float): The mixture coefficient for adjacency matrix normalisation.
        clusters (list): a list of lists of node IDs such that each list corresponds to a cluster of nodes
        in G. The clusters should be non-overlapping. If None, the G is clustered into k clusters.
        name (str): an optional name of the generator
    """

    def __init__(self, G, k=1, q=1, lam=0.1, clusters=None, name=None):

        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        self.graph = G
        self.name = name
        self.k = k
        self.q = q  # The number of clusters to sample per mini-batch
        self.lam = lam
        self.clusters = clusters

        if clusters:
            self.k = len(clusters)

        # Some error checking on the given parameter values
        if lam < 0 or lam > 1 or not isinstance(lam, float):
            raise ValueError(
                "{}: lam must be a float in the range [0, 1].".format(
                    type(self).__name__
                )
            )

        if q <= 0 or not isinstance(q, int):
            raise ValueError(
                "{}: q must be a positive integer.".format(type(self).__name__)
            )

        if k <= 0 or not isinstance(k, int):
            raise ValueError(
                "{}: k must be a positive integer.".format(type(self).__name__)
            )

        if k % q != 0:
            raise ValueError(
                "{}: k must be exactly divisible by q.".format(type(self).__name__)
            )

        # Check if the graph has features
        G.check_graph_for_ml()

        self.node_list = list(G.nodes())

        # We need a schema to check compatibility with ClusterGCN
        self.schema = G.create_graph_schema(create_type_maps=True)

        # Check that there is only a single node type
        if len(self.schema.node_types) > 1:
            raise TypeError(
                "{}: node generator requires graph with single node type; "
                "a graph with multiple node types is passed. Stopping.".format(
                    type(self).__name__
                )
            )

        if not clusters:
            # We are not given graph clusters.
            # We are going to split the graph into self.k random clusters
            all_nodes = list(G.nodes())
            random.shuffle(all_nodes)
            cluster_size = len(all_nodes) // self.k
            self.clusters = [
                all_nodes[i : i + cluster_size]
                for i in range(0, len(all_nodes), cluster_size)
            ]
            if len(self.clusters) > self.k:
                # for the case that the number of nodes is not exactly divisible by k, we combine
                # the last cluster with the second last one
                self.clusters[-2].extend(self.clusters[-1])
                del self.clusters[-1]

        print(f"Number of clusters {len(self.clusters)}")
        for i, c in enumerate(self.clusters):
            print(f"{i} cluster has size {len(c)}")

        # Get the features for the nodes
        self.features = G.get_feature_for_nodes(self.node_list)

    def flow(self, node_ids, targets=None, name=None):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_ids (iterable): and iterable of node ids for the nodes of interest
                (e.g., training, validation, or test set nodes)
            targets (2d array, optional): a 2D array of numeric node targets with shape `(len(node_ids),
                target_size)`
            name (str, optional): An optional name for the returned generator object.

        Returns:
            A NodeSequence object to use with ClusterGCN
            in Keras methods :meth:`fit_generator`, :meth:`evaluate_generator`,
            and :meth:`predict_generator`

        """
        if targets is not None:
            # Check targets is an iterable
            if not is_real_iterable(targets):
                raise TypeError("{}: Targets must be an iterable or None".format(type(self).__name__))

            # Check targets correct shape
            if len(targets) != len(node_ids):
                raise TypeError("{}: Targets must be the same length as node_ids".format(type(self).__name__))

        # The list of indices of the target nodes in self.node_list
        # node_indices = np.array([self.node_list.index(n) for n in node_ids])

        return ClusterNodeSequence(
            self.graph,
            self.clusters,
            targets=targets,
            node_ids=node_ids,
            q=self.q,
            name=name,
        )


class ClusterNodeSequence(Sequence):
    """
    Keras-compatible data generator for for node inference using Cluster GCN model.
    Use this class with the Keras methods :meth:`keras.Model.fit_generator`,
        :meth:`keras.Model.evaluate_generator`, and
        :meth:`keras.Model.predict_generator`,

    This class should be created using the `.flow(...)` method of
    :class:`ClusterNodeGenerator`.

    Args:
        graph (StellarGraph): The graph
        clusters (list): A list of lists such that each sub-list indicates the nodes in a cluster.
            The length of this list, len(clusters) indicates the number of batches in one epoch.
        targets (np.ndarray, optional): An optional array of node targets of size (N x C),
            where C is the target size (e.g., number of classes for one-hot class targets)
        node_ids (iterable, optional): The node IDs for the target nodes. Required if targets is not None.
        normalize_adj (bool, optional): Specifies whether the adjacency matrix for each mini-batch should
            be normalized or not. The default is True.
        q (int, optional): The number of subgraphs to combine for each batch. The default value is
            1 such that the generator treats each subgraph as a batch.
        lam (float, optional): The mixture coefficient for adjacency matrix normalisation (the
        'diagonal enhancement' method). Valid
            values are in the interval [0, 1] and the default value is 0.1.
        name (str, optional): An optional name for this generator object.
    """
    def __init__(
        self,
        graph,
        clusters,
        targets=None,
        node_ids=None,
        normalize_adj=True,
        q=1,
        lam=0.1,
        name=None,
    ):

        if (targets is not None) and (len(node_ids) != len(targets)):
            raise ValueError(
                "When passed together targets and indices should be the same length."
            )

        self.name = name
        self.clusters = list()  # clusters
        self.clusters_original = copy.deepcopy(clusters)
        self.graph = graph
        self.target_ids = list(node_ids)
        self.node_list = list(graph.nodes())
        self.normalize_adj = normalize_adj
        self.q = q
        self.lam = lam
        self.node_order = list()  # node_ids  # initially it should be in this order
        self._node_order_in_progress = list()
        self.__node_buffer = dict()

        if targets is not None:
            self.targets = np.asanyarray(targets)
            self.target_node_lookup = dict(
                zip(self.target_ids, range(len(self.target_ids)))
            )
        else:
            self.targets = None

        self.on_epoch_end()

    def __len__(self):
        num_batches = len(self.clusters_original) // self.q
        return num_batches

    def __getitem__(self, index):
        # The next batch should be the adjacency matrix for the cluster and the corresponding feature vectors
        # and targets if available.
        # print(f"In __getitem_ index: {index}")
        # print(f"self.clusters: {self.clusters}")
        cluster = self.clusters[index]
        g_cluster = self.graph.subgraph(
            cluster
        )  # Get the subgraph; returns SubGraph view

        adj_cluster = nx.adjacency_matrix(
            g_cluster
        )  # order is given by order of IDs in cluster

        # The operations to normalize the adjacency matrix are too slow.
        # Either optimize this or implement as a layer(?)
        if self.normalize_adj:
            # add self loops
            adj_cluster.setdiag(1)  # add self loops
            degree_matrix_diag = 1.0 / (adj_cluster.sum(axis=1) + 1)
            degree_matrix_diag = np.squeeze(np.asarray(degree_matrix_diag))
            degree_matrix = sparse.lil_matrix(adj_cluster.shape)
            degree_matrix.setdiag(degree_matrix_diag)
            adj_cluster = degree_matrix.tocsr() @ adj_cluster
            adj_cluster.setdiag((1.0 + self.lam) * adj_cluster.diagonal())

        adj_cluster = adj_cluster.toarray()

        g_node_list = list(g_cluster.nodes())

        # Determine the target nodes that exist in this cluster
        target_nodes_in_cluster = np.asanyarray(
            list(set(g_node_list).intersection(self.target_ids))
        )

        self.__node_buffer[index] = target_nodes_in_cluster

        # Dictionary to store node indices for quicker node index lookups
        node_lookup = dict(zip(g_node_list, range(len(g_node_list))))

        # The list of indices of the target nodes in self.node_list
        target_node_indices = np.array(
            [node_lookup[n] for n in target_nodes_in_cluster]
        )

        if index == (len(self.clusters_original) // self.q) - 1:
            # last batch
            self.__node_buffer_dict_to_list()

        cluster_targets = None
        #
        if self.targets is not None:
            # Dictionary to store node indices for quicker node index lookups
            # The list of indices of the target nodes in self.node_list
            cluster_target_indices = np.array(
                [self.target_node_lookup[n] for n in target_nodes_in_cluster]
            )
            cluster_targets = self.targets[cluster_target_indices]
            cluster_targets = cluster_targets.reshape((1,) + cluster_targets.shape)

        features = self.graph.get_feature_for_nodes(g_node_list)

        features = np.reshape(features, (1,) + features.shape)
        adj_cluster = adj_cluster.reshape((1,) + adj_cluster.shape)
        target_node_indices = target_node_indices[np.newaxis, np.newaxis, :]

        return [features, target_node_indices, adj_cluster], cluster_targets

    def __node_buffer_dict_to_list(self):
        self.node_order = []
        for k, v in self.__node_buffer.items():
            self.node_order.extend(v)

    def on_epoch_end(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        if self.q > 1:
            # combine clusters
            cluster_indices = list(range(len(self.clusters_original)))
            random.shuffle(cluster_indices)
            self.clusters = []

            for i in range(0, len(cluster_indices) - 1, self.q):
                cc = cluster_indices[i : i + self.q]
                tmp = []
                for l in cc:
                    tmp.extend(list(self.clusters_original[l]))
                self.clusters.append(tmp)
        else:
            self.clusters = copy.deepcopy(self.clusters_original)

        self.__node_buffer = dict()

        random.shuffle(self.clusters)
