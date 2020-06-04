# -*- coding: utf-8 -*-
#
# Copyright 2017-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["EdgeSplitter"]

import datetime
import warnings
import networkx as nx
import pandas as pd
import numpy as np
from math import isclose

from ..core import StellarGraph
from ..globalvar import FEATURE_ATTR_NAME


class EdgeSplitter(object):
    """
    Class for generating training and test data for link prediction in graphs.

    The class requires as input a graph (in networkx format) and a percentage as a function of the total number of edges
    in the given graph of the number of positive and negative edges to sample. For heterogeneous graphs, the caller
    can also specify the type of edge and an edge property to split on. In the latter case, only a date property
    can be used and it must be in the format dd/mm/yyyy. A date to be used as a threshold value such that only
    edges that have date after the threshold must be given. This effects only the sampling of positive edges.

    Negative edges are sampled at random by (for 'global' method) selecting two nodes in the graph and
    then checking if these edges are connected or not. If not, the pair of nodes is considered a negative sample.
    Otherwise, it is discarded and the process repeats. Alternatively, negative edges are sampled (for 'local' method)
    using DFS search at a distance from the source node (selected at random from all nodes in the graph)
    sampled according to a given set of probabilities.

    Positive edges can be sampled so that when they are subsequently removed from the graph, the reduced graph is either
    guaranteed, or not guaranteed, to remain connected. In the former case, graph connectivity is maintained by first
    calculating the minimum spanning tree. The edges that belong to the minimum spanning tree are protected from
    removal, and therefore cannot be sampled for the training set. The edges that do not belong to the minimum spanning
    tree are then sampled uniformly at random, until the required number of positive edges have been sampled for the
    training set. In the latter case, when connectedness of the reduced graph is not guaranteed, positive edges are
    sampled uniformly at random from all the edges in the graph, regardless of whether they belong to the spanning tree
    (which is not calculated in this case).

    Args:
        g (StellarGraph or networkx object): The graph to sample edges from.
        g_master (StellarGraph or networkx object): The graph representing the original dataset and a superset of the
            graph g. If it is not None, then when positive and negative edges are sampled, care is taken to make sure
            that a true positive edge is not sampled as a negative edge.

    """

    def __init__(self, g, g_master=None):
        # rather than rewrite this to use StellarGraph natively, this has the desired API (StellarGraphs in and
        # StellarGraphs out) by converting to/from NetworkX at the boundaries
        self._input_was_stellargraph = isinstance(g, StellarGraph)
        if self._input_was_stellargraph:
            g = g.to_networkx()

        if isinstance(g_master, StellarGraph):
            g_master = g_master.to_networkx()

        # the original graph copied over
        self.g = g.copy()
        self.g_master = g_master
        # placeholder: it will hold the subgraph of self.g after edges are removed as positive training samples
        self.g_train = None

        self.positive_edges_ids = None
        self.positive_edges_labels = None
        self.negative_edges_ids = None
        self.negative_edges_labels = None

        self.negative_edge_node_distances = None
        self.minedges = None  # the minimum spanning tree as a list of edges.
        self.minedges_set = None  # lookup dictionary for edges in minimum spanning tree
        self._random = None

    def _train_test_split_homogeneous(
        self, p, method, probs=None, keep_connected=False
    ):
        """
        Method for edge splitting applied to homogeneous graphs.

        Args:
            p (float): Percent of edges to be returned. It is calculated as a function of the total number of edges
             in the original graph. If the graph is heterogeneous, the percentage is calculated
             as a function of the total number of edges that satisfy the edge_label, edge_attribute_label and
             edge_attribute_threshold values given.
            method (string): Should be 'global' or 'local'. Specifies the method for selecting negative examples.
            probs (list of float, optional): If method is 'local' then this vector of floats specifies the probabilities for
             sampling at each depth from the source node. The first value should be 0.0 and all values should sum to 1.0.
            keep_connected (bool): If True then when positive edges are removed care is taken that the reduced graph
             remains connected. If False, positive edges are removed without guaranteeing the connectivity of the reduced graph.

        Returns:
            2 numpy arrays, the first Nx2 holding the node ids for the edges and the second Nx1 holding the edge
        labels, 0 for negative and 1 for positive example.

        """
        # minedges are those edges that if removed we might end up with a disconnected graph after the positive edges
        # have been sampled.
        if keep_connected:
            self.minedges = self._get_minimum_spanning_edges()
        else:
            self.minedges = []
            self.minedges_set = set()

        # Sample the positive examples
        positive_edges = self._reduce_graph(minedges=self.minedges_set, p=p)
        df = pd.DataFrame(positive_edges)
        self.positive_edges_ids = np.array(df.iloc[:, 0:2])
        self.positive_edges_labels = np.array(df.iloc[:, 2])

        if method == "global":
            negative_edges = self._sample_negative_examples_global(
                p=p, limit_samples=len(positive_edges)
            )
        else:  # method == 'local'
            if probs is None:  # use default values if not given, by warn user
                probs = [0.0, 0.25, 0.50, 0.25]
                warnings.warn(
                    "Using default sampling probabilities (distance from source node): {}".format(
                        probs
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
            negative_edges = self._sample_negative_examples_local_dfs(
                p=p, probs=probs, limit_samples=len(positive_edges)
            )

        df = pd.DataFrame(negative_edges)
        self.negative_edges_ids = np.array(df.iloc[:, 0:2])
        self.negative_edges_labels = np.array(df.iloc[:, 2])

        if len(self.positive_edges_ids) == 0:
            raise Exception("Could not sample any positive edges")
        if len(self.negative_edges_ids) == 0:
            raise Exception("Could not sample any negative edges")

        edge_data_ids = np.vstack((self.positive_edges_ids, self.negative_edges_ids))
        edge_data_labels = np.hstack(
            (self.positive_edges_labels, self.negative_edges_labels)
        )
        print(
            "** Sampled {} positive and {} negative edges. **".format(
                len(self.positive_edges_ids), len(self.negative_edges_ids)
            )
        )

        return edge_data_ids, edge_data_labels

    def _train_test_split_heterogeneous(
        self,
        p,
        method,
        edge_label,
        probs=None,
        keep_connected=False,
        edge_attribute_label=None,
        edge_attribute_threshold=None,
    ):
        """
        Splitting edge data based on edge type or edge type and edge property. The edge property must be a date in the
        format dd/mm/yyyy. If splitting by date, then a threshold value must also be given such that only edges with
        date larger than the threshold can be in the set of positive examples. The edge property does not effect the
        sampling of negative examples.

        Args:
            p (float): Percent of edges to be returned. It is calculated as a function of the total number of edges
             in the original graph. If the graph is heterogeneous, the percentage is calculated
             as a function of the total number of edges that satisfy the edge_label, edge_attribute_label and
             edge_attribute_threshold values given.
            method (str): Should be 'global' or 'local'. Specifies the method for selecting negative examples.
            edge_label (str): The edge type to split on
            probs (list of float, optional): If method=='local' then this vector of floats specifies the probabilities for
             sampling at each depth from the source node. The first value should be 0.0 and all values should sum to 1.0.
            keep_connected (bool): If True then when positive edges are removed care is taken that the reduced graph
             remains connected. If False, positive edges are removed without guaranteeing the connectivity of the reduced graph.
            edge_attribute_label (str): The label for the edge attribute to split on
            edge_attribute_threshold (str, optional): The threshold value applied to the edge attribute when sampling positive
             examples

        Returns:
            2 numpy arrays, the first Nx2 holding the node ids for the edges and the second Nx1 holding the edge
        labels, 0 for negative and 1 for positive example.
        """
        # minedges are those edges that if removed we might end up with a disconnected graph after the positive edges
        # have been sampled.
        if keep_connected:
            self.minedges = self._get_minimum_spanning_edges()
        else:
            self.minedges = []
            self.minedges_set = set()

        # Note: The caller guarantees the edge_label is not None so we don't have to check here again.
        if edge_attribute_threshold is None:
            positive_edges = self._reduce_graph_by_edge_type(
                minedges=self.minedges_set, p=p, edge_label=edge_label
            )
        else:
            positive_edges = self._reduce_graph_by_edge_type_and_attribute(
                minedges=self.minedges_set,
                p=p,
                edge_label=edge_label,
                edge_attribute_label=edge_attribute_label,
                edge_attribute_threshold=edge_attribute_threshold,
            )

        if len(positive_edges) == 0:
            raise Exception(
                "ERROR: Unable to sample any positive edges of type '{}'".format(
                    edge_label
                )
            )

        df = pd.DataFrame(positive_edges)
        self.positive_edges_ids = np.array(df.iloc[:, 0:2])
        self.positive_edges_labels = np.array(df.iloc[:, 2])

        if method == "global":
            negative_edges = self._sample_negative_examples_by_edge_type_global(
                p=p,
                edges=positive_edges,
                edge_label=edge_label,
                limit_samples=len(positive_edges),
            )
        else:  # method == 'local'
            if probs is None:
                probs = [0.0, 0.25, 0.50, 0.25]
                warnings.warn(
                    "Using default sampling probabilities (distance from source node): {}".format(
                        probs
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
            negative_edges = self._sample_negative_examples_by_edge_type_local_dfs(
                p=p,
                probs=probs,
                edges_positive=positive_edges,
                edge_label=edge_label,
                limit_samples=len(positive_edges),
            )

        df = pd.DataFrame(negative_edges)
        self.negative_edges_ids = np.array(df.iloc[:, 0:2])
        self.negative_edges_labels = np.array(df.iloc[:, 2])

        if len(self.positive_edges_ids) == 0:
            raise Exception("Could not sample any positive edges")
        if len(self.negative_edges_ids) == 0:
            raise Exception("Could not sample any negative edges")

        edge_data_ids = np.vstack((self.positive_edges_ids, self.negative_edges_ids))
        edge_data_labels = np.hstack(
            (self.positive_edges_labels, self.negative_edges_labels)
        )
        print(
            "** Sampled {} positive and {} negative edges. **".format(
                len(self.positive_edges_ids), len(self.negative_edges_ids)
            )
        )

        return edge_data_ids, edge_data_labels

    def train_test_split(
        self,
        p=0.5,
        method="global",
        probs=None,
        keep_connected=False,
        edge_label=None,
        edge_attribute_label=None,
        edge_attribute_threshold=None,
        attribute_is_datetime=None,
        seed=None,
    ):
        """
        Generates positive and negative edges and a graph that has the same nodes as the original but the positive
        edges removed. It can be used to generate data from homogeneous and heterogeneous graphs.

        For heterogeneous graphs, positive and negative examples can be generated based on specified edge type or
        edge type and edge property given a threshold value for the latter.

        Args:
            p (float): Percent of edges to be returned. It is calculated as a function of the total number of edges
             in the original graph. If the graph is heterogeneous, the percentage is calculated
             as a function of the total number of edges that satisfy the edge_label, edge_attribute_label and
             edge_attribute_threshold values given.
            method (str): How negative edges are sampled. If 'global', then nodes are selected at random.
             If 'local' then the first nodes is sampled from all nodes in the graph, but the second node is
             chosen to be from the former's local neighbourhood.
            probs (list): list The probabilities for sampling a node that is k-hops from the source node,
             e.g., [0.25, 0.75] means that there is a 0.25 probability that the target node will be 1-hope away from the
             source node and 0.75 that it will be 2 hops away from the source node. This only affects sampling of
             negative edges if method is set to 'local'.
            keep_connected (bool): If True then when positive edges are removed care is taken that the reduced graph
             remains connected. If False, positive edges are removed without guaranteeing the connectivity of the reduced graph.
            edge_label (str, optional) If splitting based on edge type, then this parameter specifies the key for the type
             of edges to split on.
            edge_attribute_label (str, optional): The label for the edge attribute to split on.
            edge_attribute_threshold (str, optional): The threshold value applied to the edge attribute when sampling positive
             examples.
            attribute_is_datetime (bool, optional): Specifies if edge attribute is datetime or not.
            seed (int, optional): seed for random number generator, positive int or 0

        Returns:
            The reduced graph (positive edges removed) and the edge data as 2 numpy arrays, the first array of
            dimensionality Nx2 (where N is the number of edges) holding the node ids for the edges and the second of
            dimensionality Nx1 holding the edge labels, 0 for negative and 1 for positive examples. The graph
            matches the input graph passed to the :class:`EdgeSplitter` constructor: the returned graph is a
            :class:`StellarGraph` instance if the input graph was one, and, similarly, a NetworkX graph if the input
            graph was one.
        """
        if p <= 0 or p >= 1:
            raise ValueError("The value of p must be in the interval (0,1)")

        if method != "global" and method != "local":
            raise ValueError(
                "Invalid method {}; valid options are 'local' or 'global'".format(
                    method
                )
            )

        if not isinstance(keep_connected, (bool,)):
            raise ValueError(
                "({}) The flag keep_connected be boolean type.".format(
                    type(self).__name__
                )
            )

        if seed is not None:
            if seed < 0:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be positive integer or None.".format(
                        type(self).__name__
                    )
                )
            if type(seed) != int:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be integer type or None.".format(
                        type(self).__name__
                    )
                )
        if self._random is None:  # only do this one
            self._random = np.random.RandomState(seed=seed)

        if edge_label is not None:  # working with a heterogeneous graph
            if (
                edge_attribute_label
                and edge_attribute_threshold
                and not attribute_is_datetime
            ):
                raise ValueError("You can only split by datetime edge attribute")
            else:  # all three are True
                edge_data_ids, edge_data_labels = self._train_test_split_heterogeneous(
                    p=p,
                    method=method,
                    edge_label=edge_label,
                    edge_attribute_label=edge_attribute_label,
                    edge_attribute_threshold=edge_attribute_threshold,
                    keep_connected=keep_connected,
                )
        else:  # working with a homogeneous graph
            edge_data_ids, edge_data_labels = self._train_test_split_homogeneous(
                p=p, method=method, probs=probs, keep_connected=keep_connected
            )

        if self._input_was_stellargraph:
            # if the graphs came in as a StellarGraph, return one too
            result_graph = StellarGraph.from_networkx(
                self.g_train, node_features=FEATURE_ATTR_NAME
            )
        else:
            result_graph = self.g_train

        return result_graph, edge_data_ids, edge_data_labels

    def _get_edges(
        self, edge_label, edge_attribute_label=None, edge_attribute_threshold=None
    ):
        """
        Method that filters the edges in the self.g (heterogeneous) graph based on either the edge type
        specified by edge_label, or based on edges of edge_label type that have property edge_attribute_label and
        the value of the latter property is larger than the edge_attribute_threshold.

        Args:
            edge_label (str): The type of edges to filter
            edge_attribute_label (str, optional): The edge attribute to use for filtering graph edges
            edge_attribute_threshold (str, optional): The threshold applied to the edge attribute for filtering edges.

        Returns:
            (list) List of edges that satisfy the filtering criteria.

        """
        # the graph in networkx format is stored in self.g_train
        if self.g.is_multigraph():
            all_edges = list(self.g.edges(keys=True))
        else:
            all_edges = list(self.g.edges())

        if edge_attribute_label is None or edge_attribute_threshold is None:
            # filter by edge_label
            edges_with_label = [
                e for e in all_edges if self.g.get_edge_data(*e)["label"] == edge_label
            ]

        else:
            # filter by edge label, edge attribute and threshold value
            edge_attribute_threshold_dt = datetime.datetime.strptime(
                edge_attribute_threshold, "%d/%m/%Y"
            )
            edges_with_label = [
                e
                for e in all_edges
                if (
                    self.g.get_edge_data(*e)["label"] == edge_label
                    and datetime.datetime.strptime(
                        self.g.get_edge_data(*e)[edge_attribute_label], "%d/%m/%Y"
                    )
                    > edge_attribute_threshold_dt
                )
            ]

        return edges_with_label

    def _get_edge_source_and_target_node_types(self, edges):
        """
        Method that given a list of edges, for each edge it determines the type of the source and target
        nodes and then returns them as a list of tuples.

        This routine is necessary because networkx does not provide a direct method for determining the type of nodes
        given an edge.

        Args:
            edges (list): List of edges as returned by networkx graph method edges().

        Returns: (list) Returns a list of 2-tuples such that each value in the tuple holds the type (as str) of the
        source and target nodes for each element in edges.

        """
        # uses self.g_train but any graph object would do since nodes are shared
        all_nodes = self.g_train.nodes(data=True)
        # dictionary that maps node id to node attributes
        all_nodes_as_dict = {n[0]: n[1] for n in all_nodes}
        edge_node_types = set()
        for edge in edges:
            edge_node_types.add(
                (
                    all_nodes_as_dict[edge[0]]["label"],
                    all_nodes_as_dict[edge[1]]["label"],
                )
            )

        return edge_node_types

    def _reduce_graph_by_edge_type_and_attribute(
        self,
        minedges,
        p=0.5,
        edge_label=None,
        edge_attribute_label=None,
        edge_attribute_threshold=None,
    ):
        """
        Reduces the graph self.g_train by a factor p by removing existing edges not on minedges list such that
        the reduced tree remains connected. Edges are removed based on the edge type and the values of a given edge
        attribute and a threshold applied to the latter.

        Args:
            minedges (list): Spanning tree edges that cannot be removed.
            p (float): Factor by which to reduce the size of the graph.
            edge_label (str): The edge type to consider.
            edge_attribute_label (str): The edge attribute to consider.
            edge_attribute_threshold (str): The threshold value; only edges with attribute value larger than the
             threshold can be removed.

        Returns:
            Returns the list of edges removed from the graph (also modifies the graph self.g_train
            by removing the said edges)

        """
        # We check that the parameters are given values but we don't check if the graph has edges with label
        # edge_label and edge attributes with label edge_attribute_label. For now, we assume that the given values
        # are valid; if not, then some cryptic exception is bound to be raised later on in the code.
        if edge_label is None:
            raise ValueError("edge_label must be specified.")
        if edge_attribute_label is None:
            raise ValueError("edge_attribute_label must be specified.")
        if edge_attribute_threshold is None:
            raise ValueError("attribute_threshold must be specified.")

        # copy the original graph and start over in case this is not the first time
        # reduce_graph has been called.
        self.g_train = self.g.copy()

        # Filter the graph's edges based on the edge type, edge attribute, and attribute threshold value given.
        all_edges = self._get_edges(
            edge_label=edge_label,
            edge_attribute_label=edge_attribute_label,
            edge_attribute_threshold=edge_attribute_threshold,
        )
        # Also, calculate the number of these edges in the graph.
        num_edges_total = len(all_edges)
        # print("Graph has {} edges of type {}".format(num_edges_total, edge_label))
        # Multiply this number by p to determine the number of positive edge examples to sample
        num_edges_to_remove = int(num_edges_total * p)

        # shuffle the edges
        self._random.shuffle(all_edges)
        #
        # iterate over the list of edges and for each edge if the edge is not in minedges, remove it from the graph
        # until num_edges_to_remove edges have been removed and the graph reduced to p of its original size
        count = 0
        removed_edges = []
        for edge in all_edges:
            # Support minedges having keys (NetworkX 2.x) or not (NetworkX 1.x)
            if edge not in minedges and (edge[0], edge[1]) not in minedges:
                removed_edges.append(
                    (
                        edge[0],
                        edge[1],
                        1,
                    )  # should this be edge + (1,) to support multigraphs?
                )  # the last entry is the label
                self.g_train.remove_edge(*edge)

                count += 1
            if count == num_edges_to_remove:
                return removed_edges

        if len(removed_edges) < num_edges_to_remove:
            raise ValueError(
                "Unable to sample {} positive edges (could only sample {} positive edges). Consider using smaller value for p or set keep_connected=False".format(
                    num_edges_to_remove, len(removed_edges)
                )
            )

    def _reduce_graph_by_edge_type(self, minedges, p=0.5, edge_label=None):
        """
        Reduces the graph self.g_train by a factor p by removing existing edges not on minedges list such that
        the reduced tree remains connected. Edges are removed based on the edge type.

        Args:
            minedges (list): Minimum spanning tree edges that cannot be removed.
            p (float): Factor by which to reduce the size of the graph.
            edge_label (str): The edge type to consider.

        Returns:
            (list) Returns the list of edges removed from self.g_train (also modifies self.g_train by removing said
            edges)
        """
        if edge_label is None:
            raise ValueError("edge_label must be specified")

        # copy the original graph and start over in case this is not the first time
        # reduce_graph has been called.
        self.g_train = self.g.copy()

        # Filter the graph's edges based on the specified edge_label
        all_edges = self._get_edges(edge_label=edge_label)
        num_edges_total = len(all_edges)
        print("Network has {} edges of type {}".format(num_edges_total, edge_label))
        # Multiply this number by p to determine the number of positive edge examples to sample
        num_edges_to_remove = int(num_edges_total * p)
        # shuffle the edges
        self._random.shuffle(all_edges)

        # iterate over the list of filtered edges and for each edge if the edge is not in minedges, remove it from
        # the graph until num_edges_to_remove edges have been removed and the graph is reduced to p of its original
        #  size
        count = 0
        removed_edges = []
        for edge in all_edges:
            # Support minedges having keys (NetworkX 2.x) or not (NetworkX 1.x)
            if edge not in minedges and (edge[0], edge[1]) not in minedges:
                removed_edges.append(
                    (edge[0], edge[1], 1)
                )  # the last entry is the label
                self.g_train.remove_edge(*edge)
                count += 1
            if count == num_edges_to_remove:
                return removed_edges

        if len(removed_edges) < num_edges_to_remove:
            raise ValueError(
                "Unable to sample {} positive edges (could only sample {} positive edges). Consider using smaller value for p or set keep_connected=False".format(
                    num_edges_to_remove, len(removed_edges)
                )
            )

    def _reduce_graph(self, minedges, p=0.5):
        """
        Reduces the graph self.g_train by a factor p by removing existing edges not on minedges list such that
        the reduced tree remains connected. Edge type is ignored and all edges are treated equally.

        Args:
            minedges (list): Minimum spanning tree edges that cannot be removed.
            p (float): Factor by which to reduce the size of the graph.

        Returns:
            (list) Returns the list of edges removed from self.g_train (also modifies self.g_train by removing the
            said edges)
        """
        # copy the original graph and start over in case this is not the first time
        # reduce_graph has been called.
        self.g_train = self.g.copy()

        # For multigraphs we should probably use keys
        use_keys_in_edges = self.g.is_multigraph()

        # For NX 1.x/2.x compatibilty we need to match length of minedges
        if len(minedges) > 0:
            use_keys_in_edges = len(next(iter(minedges))) == 3

        if use_keys_in_edges:
            all_edges = list(self.g_train.edges(keys=True))
        else:
            all_edges = list(self.g_train.edges())

        num_edges_to_remove = int(self.g_train.number_of_edges() * p)

        if num_edges_to_remove > (self.g_train.number_of_edges() - len(self.minedges)):
            raise ValueError(
                "Not enough positive edges to sample after reserving {} number of edges for maintaining graph connectivity. Consider setting keep_connected=False.".format(
                    len(self.minedges)
                )
            )

        # shuffle the edges
        self._random.shuffle(all_edges)
        # iterate over the list of edges and for each edge if the edge is not in minedges, remove it from the graph
        # until num_edges_to_remove edges have been removed and the graph reduced to p of its original size
        count = 0
        removed_edges = []
        for edge in all_edges:
            if edge not in minedges:
                removed_edges.append(
                    (edge[0], edge[1], 1)
                )  # the last entry is the label
                self.g_train.remove_edge(*edge)

                count += 1
            if count == num_edges_to_remove:
                return removed_edges

    def _sample_negative_examples_by_edge_type_local_dfs(
        self,
        p=0.5,
        probs=None,
        edges_positive=None,
        edge_label=None,
        limit_samples=None,
    ):
        """
        This method produces a list of edges that don't exist in graph self.g (negative examples). The number of
        negative edges produced is equal to the number of edges in the graph times p (that should be in the range (0,1]
        or limited to maximum limit_samples if the latter is not None. The negative samples are between node types
        as inferred from the edge type of the positive examples previously removed from the graph and given in
        edges_positive.

        This method uses depth-first search to efficiently (memory-wise) sample negative edges based on the local
        neighbourhood of randomly (uniformly) sampled source nodes at distances defined by the probabilities in probs.
        The source graph is not modified.

        Args:
            p (float): Factor that multiplies the number of edges in the graph and determines the number of negative
            edges to be sampled.
            probs (list): Probability distribution for the distance between source and target nodes.
            edges_positive (list): The positive edge examples that have previously been removed from the graph
            edge_label (str): The edge type to sample negative examples of
            limit_samples (int, optional): It limits the maximum number of samples to the given number, if not None

        Returns:
            (list) A list of 2-tuples that are pairs of node IDs that don't have an edge between them in the graph.
        """
        if probs is None:
            probs = [0.0, 0.25, 0.50, 0.25]
            warnings.warn(
                "Using default sampling probabilities up to 3 hops from source node with values {}".format(
                    probs
                )
            )

        if not isclose(sum(probs), 1.0):
            raise ValueError("Sampling probabilities do not sum to 1")

        self.negative_edge_node_distances = []
        n = len(probs)

        # determine the number of edges in the graph that have edge_label type
        # Multiply this number by p to determine the number of positive edge examples to sample
        all_edges = self._get_edges(edge_label=edge_label)
        num_edges_total = len(all_edges)
        print("Network has {} edges of type {}".format(num_edges_total, edge_label))
        #
        num_edges_to_sample = int(num_edges_total * p)

        if limit_samples is not None:
            if num_edges_to_sample > limit_samples:
                num_edges_to_sample = limit_samples

        edge_source_target_node_types = self._get_edge_source_and_target_node_types(
            edges=edges_positive
        )

        if self.g_master is None:
            edges = self.g.edges()
        else:
            edges = self.g_master.edges()

        # to speed up lookup of edges in edges list, create a set the values stored are the concatenation of
        # the source and target node ids.
        edges_set = set(edges)
        edges_set.update({(e[1], e[0]) for e in edges})
        sampled_edges_set = set()

        start_nodes = list(self.g.nodes(data=True))
        nodes_dict = {node[0]: node[1]["label"] for node in start_nodes}

        count = 0
        sampled_edges = []

        num_iter = int(np.ceil(num_edges_to_sample / (1.0 * len(start_nodes)))) + 1

        for _ in np.arange(0, num_iter):
            self._random.shuffle(start_nodes)
            # sample the distance to the target node using probs
            target_node_distances = (
                self._random.choice(n, len(start_nodes), p=probs) + 1
            )
            for u, d in zip(start_nodes, target_node_distances):
                # perform DFS search up to d distance from the start node u.
                visited = {
                    node[0]: False for node in start_nodes
                }  # for marking already visited nodes
                nodes_stack = list()
                # start at node u
                nodes_stack.append((u[0], 0))  # tuple is (node, depth)
                while len(nodes_stack) > 0:
                    next_node = nodes_stack.pop()
                    v = next_node[0]  # retrieve node id
                    dv = next_node[1]  # retrieve node distance from u
                    if not visited[v]:
                        visited[v] = True
                        # Check if this nodes is at depth d; if it is, then this could be selected as the
                        # target node for a negative edge sample. Otherwise add its neighbours to the stack, only
                        # if the depth is less than the search depth d.
                        if dv == d:
                            u_v_edge_type = (nodes_dict[u[0]], nodes_dict[v])
                            # if no edge between u and next_node[0] then this is the sample, so record and stop
                            # searching
                            # Note: The if statement below is very expensive to evaluate because it need to checks
                            # the membership of an element in a number of lists that can grow very large for large
                            # graphs and number examples to sample. Later, we should have a closer look at how we can
                            # speed this up.
                            if (
                                (u_v_edge_type in edge_source_target_node_types)
                                and (u[0] != v)
                                and ((u[0], v) not in edges_set)
                                and ((u[0], v) not in sampled_edges_set)
                            ):

                                sampled_edges.append(
                                    (u[0], v, 0)
                                )  # the last entry is the class label
                                sampled_edges_set.add((u[0], v))
                                sampled_edges_set.add((v, u[0]))
                                count += 1
                                self.negative_edge_node_distances.append(d)
                            break
                        elif dv < d:
                            neighbours = list(nx.neighbors(self.g, v))
                            self._random.shuffle(neighbours)
                            neighbours = [(k, dv + 1) for k in neighbours]
                            nodes_stack.extend(neighbours)

                if count == num_edges_to_sample:
                    return sampled_edges

        if len(sampled_edges) != num_edges_to_sample:
            raise ValueError(
                "Unable to sample {} negative edges. Consider using smaller value for p.".format(
                    num_edges_to_sample
                )
            )

    def _sample_negative_examples_local_dfs(
        self, p=0.5, probs=None, limit_samples=None
    ):
        """
        This method produces a list of edges that don't exist in graph self.g (negative examples). The number of
        negative edges produced is equal to the number of edges in the graph times p (that should be in the range (0,1]
        or limited to maximum limit_samples if the latter is not None.

        This method uses depth-first search to efficiently (memory-wise) sample negative edges based on the local
        neighbourhood of randomly (uniformly) sampled source nodes at distances defined by the probabilities in probs.
        The source graph is not modified.

        Args:
            p (float): Factor that multiplies the number of edges in the graph and determines the number of no-edges to
            be sampled.
            probs (list): Probability distribution for the distance between source and target nodes.
            limit_samples (int, optional): It limits the maximum number of samples to the given number, if not None

        Returns:
            (list) A list of 2-tuples that are pairs of node IDs that don't have an edge between them in the graph.
        """
        if probs is None:
            probs = [0.0, 0.25, 0.50, 0.25]
            warnings.warn(
                "Using default sampling probabilities up to 3 hops from source node with values {}".format(
                    probs
                ),
                RuntimeWarning,
            )

        if not isclose(sum(probs), 1.0):
            raise ValueError("Sampling probabilities do not sum to 1")

        self.negative_edge_node_distances = []
        n = len(probs)

        num_edges_to_sample = int(self.g.number_of_edges() * p)

        if limit_samples is not None:
            if num_edges_to_sample > limit_samples:
                num_edges_to_sample = limit_samples

        if self.g_master is None:
            edges = self.g.edges()
        else:
            edges = self.g_master.edges()

        # to speed up lookup of edges in edges list, create a set the values stored are the concatenation of
        # the source and target node ids.
        edges_set = set(edges)
        edges_set.update({(e[1], e[0]) for e in edges})
        sampled_edges_set = set()

        start_nodes = list(self.g.nodes(data=False))

        count = 0
        sampled_edges = []

        num_iter = int(np.ceil(num_edges_to_sample / (1.0 * len(start_nodes))))

        for _ in np.arange(0, num_iter):
            self._random.shuffle(start_nodes)
            # sample the distance to the target node using probs
            target_node_distances = (
                self._random.choice(n, len(start_nodes), p=probs) + 1
            )
            for u, d in zip(start_nodes, target_node_distances):
                # perform DFS search up to d distance from the start node u.
                visited = {node: False for node in start_nodes}
                nodes_stack = list()
                # start at node u
                nodes_stack.append((u, 0))  # tuple is node, depth
                while len(nodes_stack) > 0:
                    next_node = nodes_stack.pop()
                    v = next_node[0]
                    dv = next_node[1]
                    if not visited[v]:
                        visited[v] = True
                        # Check if this nodes is at depth d; if it is, then this could be selected as the
                        # target node for a negative edge sample. Otherwise add its neighbours to the stack, only
                        # if the depth is less than the search depth d.
                        if dv == d:
                            # if no edge between u and next_node[0] then this is the sample, so record and stop
                            # searching
                            if (
                                (u != v)
                                and ((u, v) not in edges_set)
                                and ((u, v) not in sampled_edges_set)
                            ):
                                sampled_edges.append(
                                    (u, v, 0)
                                )  # the last entry is the class label
                                sampled_edges_set.add((u, v))
                                sampled_edges_set.add((v, u))
                                count += 1
                                self.negative_edge_node_distances.append(d)
                                break
                        elif dv < d:
                            neighbours = list(nx.neighbors(self.g, v))
                            self._random.shuffle(neighbours)
                            neighbours = [(k, dv + 1) for k in neighbours]
                            nodes_stack.extend(neighbours)
                if count == num_edges_to_sample:
                    return sampled_edges

        if len(sampled_edges) != num_edges_to_sample:
            raise ValueError(
                "Unable to sample {} negative edges. Consider using smaller value for p.".format(
                    num_edges_to_sample
                )
            )

    def _sample_negative_examples_global(self, p=0.5, limit_samples=None):
        """
        This method samples uniformly at random nodes from the graph and, if they don't have an edge in the graph,
        it records the pair as a negative edge.

        Args:
            p: (float) factor that multiplies the number of edges in the graph and determines the number of negative
            edges to be sampled.
            limit_samples: (int, optional) it limits the maximum number of samples to the given number, if not None

        Returns:
            (list) A list of 2-tuples that are pairs of node IDs that don't have an edge between them in the graph.
        """
        self.negative_edge_node_distances = []

        num_edges_to_sample = int(self.g.number_of_edges() * p)

        if limit_samples is not None:
            if num_edges_to_sample > limit_samples:
                num_edges_to_sample = limit_samples

        if self.g_master is None:
            edges = list(self.g.edges())
        else:
            edges = list(self.g_master.edges())

        # to speed up lookup of edges in edges list, create a set the values stored are the concatenation of
        # the source and target node ids.
        edges_set = set(edges)
        edges_set.update({(u[1], u[0]) for u in edges})
        sampled_edges_set = set()

        start_nodes = list(self.g.nodes(data=False))
        end_nodes = list(self.g.nodes(data=False))

        count = 0
        sampled_edges = []

        num_iter = int(np.ceil(num_edges_to_sample / (1.0 * len(start_nodes)))) + 1
        for _ in np.arange(0, num_iter):
            self._random.shuffle(start_nodes)
            self._random.shuffle(end_nodes)
            for u, v in zip(start_nodes, end_nodes):
                if (
                    (u != v)
                    and ((u, v) not in edges_set)
                    and ((u, v) not in sampled_edges_set)
                ):
                    sampled_edges.append((u, v, 0))  # the last entry is the class label
                    sampled_edges_set.update(
                        {(u, v), (v, u)}
                    )  # test for bi-directional edges
                    count += 1
                if count == num_edges_to_sample:
                    return sampled_edges

        if len(sampled_edges) != num_edges_to_sample:
            raise ValueError(
                "Unable to sample {} negative edges. Consider using smaller value for p.".format(
                    num_edges_to_sample
                )
            )

    def _sample_negative_examples_by_edge_type_global(
        self, edges, edge_label, p=0.5, limit_samples=None
    ):
        """
        This method produces a list of edges that don't exist in graph self.g (negative examples). The number of
        negative edges produced is equal to the number of edges with label edge_label in the graph times p (that should
        be in the range (0,1] or limited to maximum limit_samples if the latter is not None. The negative samples are
        between node types as inferred from the edge type of the positive examples previously removed from the graph
        and given in edges_positive.

        The source graph is not modified.

        Args:
            edges (list): The positive edge examples that have previously been removed from the graph
            edge_label (str): The edge type to sample negative examples of
            p (float): Factor that multiplies the number of edges in the graph and determines the number of negative
            edges to be sampled.
            limit_samples (int, optional): It limits the maximum number of samples to the given number, if not None

        Returns:
            (list) A list of 2-tuples that are pairs of node IDs that don't have an edge between them in the graph.
        """
        self.negative_edge_node_distances = []
        # determine the number of edges in the graph that have edge_label type
        # Multiply this number by p to determine the number of positive edge examples to sample
        all_edges = self._get_edges(edge_label=edge_label)
        num_edges_total = len(all_edges)
        print("Network has {} edges of type {}".format(num_edges_total, edge_label))
        #
        num_edges_to_sample = int(num_edges_total * p)

        if limit_samples is not None:
            if num_edges_to_sample > limit_samples:
                num_edges_to_sample = limit_samples

        edge_source_target_node_types = self._get_edge_source_and_target_node_types(
            edges=edges
        )

        # to speed up lookup of edges in edges list, create a set the values stored are the concatenation of
        # the source and target node ids.
        edges_set = set(edges)
        edges_set.update({(u[1], u[0]) for u in edges})
        sampled_edges_set = set()

        start_nodes = list(self.g.nodes(data=True))
        end_nodes = list(self.g.nodes(data=True))

        count = 0
        sampled_edges = []

        num_iter = int(np.ceil(num_edges_to_sample / (1.0 * len(start_nodes)))) + 1

        for _ in np.arange(0, num_iter):
            self._random.shuffle(start_nodes)
            self._random.shuffle(end_nodes)
            for u, v in zip(start_nodes, end_nodes):
                u_v_edge_type = (u[1]["label"], v[1]["label"])
                if (
                    (u_v_edge_type in edge_source_target_node_types)
                    and (u != v)
                    and ((u[0], v[0]) not in edges_set)
                    and ((u[0], v[0]) not in sampled_edges_set)
                ):
                    sampled_edges.append(
                        (u[0], v[0], 0)
                    )  # the last entry is the class label
                    sampled_edges_set.update({(u[0], v[0]), (v[0], u[0])})
                    count += 1

                    if count == num_edges_to_sample:
                        return sampled_edges

        if len(sampled_edges) != num_edges_to_sample:
            raise ValueError(
                "Unable to sample {} negative edges. Consider using smaller value for p.".format(
                    num_edges_to_sample
                )
            )

    def _get_minimum_spanning_edges(self):
        """
        Given an undirected graph, it calculates the minimum set of edges such that graph connectivity is preserved.

        Returns:
            (list) The minimum spanning edges of the undirected graph self.g

        """
        mst = nx.minimum_spanning_edges(self.g, data=False)
        edges = list(mst)

        # to speed up lookup of edges in edges list, create a set the values stored are the concatenation of
        # the source and target node ids.
        self.minedges_set = {(u[0], u[1]) for u in edges}
        self.minedges_set.update({(u[1], u[0]) for u in edges})

        return edges
