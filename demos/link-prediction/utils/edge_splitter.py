# -*- coding: utf-8 -*-
#
# Copyright 2017-2018 Data61, CSIRO
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

import datetime
import networkx as nx
import pandas as pd
import numpy as np
from math import isclose


class EdgeSplitter(object):
    """
    Class for generating training and test data for link prediction in graphs.

    The class requires as input a graph (in netowrkx format) and a percentage as a function of the total number of edges
    in the given graph of the number of positive and negative edges to sample.

    Negative edges are sampled at random by uniformly (for 'global' method) selecting two nodes in the graph and
    then checking if these edges are connected or not. If not, the pair of nodes is considered a negative sample.
    Otherwise, it is discarded and the process repeats. Alternatively, negative edges are sampled (for 'local' method)
    using BFS search at a distance from the source node (selected uniformly at random from all nodes in the graph)
    sampled according to a given set of probabilities.

    Positive edges are sampled such that the original graph remains connected. This is achieved by first calculating
    the minimum spanning tree. The edges in the minimum spanning tree cannot be removed, i.e., selected as positive
    training edges. The remaining edges, not those on the minimum spanning tree are sampled uniformly at random until
    either the maximum number of edges that can be sampled up to the required number are sampled or the required number
    of edges have been sampled as positive examples.
    """

    def __init__(self, g, g_master=None):
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
        self.minedges = None

    def _train_test_split_homogeneous(self, p, method, **kwargs):
        # minedges are those edges that if removed we might end up with a disconnected graph after the positive edges
        # have been sampled.
        self.minedges = self._get_minimum_spanning_edges()

        positive_edges = self._reduce_graph(minedges=self.minedges, p=p)
        df = pd.DataFrame(positive_edges)
        self.positive_edges_ids = np.array(df.iloc[:, 0:2])
        self.positive_edges_labels = np.array(df.iloc[:, 2])

        if method == 'global':
            negative_edges = self._sample_negative_examples_global(p=p,
                                                                   limit_samples=len(positive_edges))
        elif method == 'local':
            probs = kwargs.get('probs', [0.0, 0.25, 0.50, 0.25])
            print("Using sampling probabilities (distance from source node): {}".format(probs))
            negative_edges = self._sample_negative_examples_local_dfs(p=p,
                                                                      probs=probs,
                                                                      limit_samples=len(positive_edges))
        else:
            raise ValueError('Invalid method {}'.format(method))

        df = pd.DataFrame(negative_edges)
        self.negative_edges_ids = np.array(df.iloc[:, 0:2])
        self.negative_edges_labels = np.array(df.iloc[:, 2])

        if len(self.positive_edges_ids) == 0:
            raise Exception("Could not sample any positive edges")
        if len(self.negative_edges_ids) == 0:
            raise Exception("Could not sample any negative edges")

        edge_data_ids = np.vstack((self.positive_edges_ids, self.negative_edges_ids))
        edge_data_labels = np.hstack((self.positive_edges_labels, self.negative_edges_labels))
        print("** Sampled {} positive and {} negative edges. **".format(len(self.positive_edges_ids),
                                                                        len(self.negative_edges_ids)))

        return edge_data_ids, edge_data_labels

    def _train_test_split_heterogeneous(self, p, method, edge_label, **kwargs):
        """
        Splitting edge data based on edge type.
        :param p: <float> Percentage (with respect to the total number of edges) of positive and negative examples to
        sample.
        :param method: <str> Should be 'global' or 'local' specifying how negative edge samples are drawn
        :param edge_label: <str> The edge type to split on
        :return:
        """
        # minedges are those edges that if removed we might end up with a disconnected graph after the positive edges
        # have been sampled.
        edge_attribute_label = kwargs.get('edge_attribute_label', None)
        edge_attribute_threshold = kwargs.get('edge_attribute_threshold', None)
        self.minedges = self._get_minimum_spanning_edges()

        if edge_attribute_threshold is None:
            positive_edges = self._reduce_graph_by_edge_type(minedges=self.minedges, p=p, edge_label=edge_label)
        else:
            positive_edges = self._reduce_graph_by_edge_type_and_attribute(minedges=self.minedges,
                                                                           p=p,
                                                                           edge_label=edge_label,
                                                                           edge_attribute_label=edge_attribute_label,
                                                                           edge_attribute_threshold=edge_attribute_threshold)

        df = pd.DataFrame(positive_edges)
        self.positive_edges_ids = np.array(df.iloc[:, 0:2])
        self.positive_edges_labels = np.array(df.iloc[:, 2])

        if method == 'global':
            negative_edges = self._sample_negative_examples_by_edge_type_global(p=p,
                                                                                edges=positive_edges,
                                                                                edge_label=edge_label,
                                                                                limit_samples=len(positive_edges))
        elif method == 'local':
            # Ah oh, where is kwargs?
            probs = kwargs.get('probs', [0.0, 0.25, 0.50, 0.25])
            print("Using sampling probabilities (distance from source node): {}".format(probs))
            negative_edges = self._sample_negative_examples_by_edge_type_local_dfs(p=p,
                                                                                   probs=probs,
                                                                                   edges_positive=positive_edges,
                                                                                   edge_label=edge_label,
                                                                                   limit_samples=len(positive_edges))
            # negative_edges = self._sample_negative_examples_local_dfs(p=p,
            #                                                           probs=probs,
            #                                                           limit_samples=len(positive_edges))
        else:
            raise ValueError('Invalid method {}'.format(method))

        df = pd.DataFrame(negative_edges)
        self.negative_edges_ids = np.array(df.iloc[:, 0:2])
        self.negative_edges_labels = np.array(df.iloc[:, 2])

        if len(self.positive_edges_ids) == 0:
            raise Exception("Could not sample any positive edges")
        if len(self.negative_edges_ids) == 0:
            raise Exception("Could not sample any negative edges")

        edge_data_ids = np.vstack((self.positive_edges_ids, self.negative_edges_ids))
        edge_data_labels = np.hstack((self.positive_edges_labels, self.negative_edges_labels))
        print("** Sampled {} positive and {} negative edges. **".format(len(self.positive_edges_ids),
                                                                        len(self.negative_edges_ids)))

        return edge_data_ids, edge_data_labels

    def train_test_split(self, p=0.5, method='global', **kwargs):
        """
        Generates positive and negative edges and a graph that has the same nodes as the original but the positive
        edges removed
        :param p: Percent of edges to be generated as a function of the total number of edges in the original graph
        :param method: <str> How negative edges are sampled. If 'global', then nodes are selected uniformaly at random.
        If 'local' then the first nodes is sampled uniformly from all nodes in the graph, but the second node is chosen
        to be from the former's local neighbourhood.
        :param probs: list The probabilities for sampling a node that is k-hops from the source node, e.g., [0.25, 0.75]
        means that there is a 0.25 probability that the target node will be 1-hope away from the source node and 0.75
        that it will be 2 hops away from the source node. This only affects sampling of negative edges if method is set
        to 'local'.
        :param edge_label: <str> If splitting based on edge type, then this parameter specifies the key for the type
        of edges to split on
        :return: The reduced graph (positive edges removed) and the edge data as numpy array with first two columns the
        node id defining the edge and the last column 1 or 0 for positive or negative example respectively.
        """
        if p <= 0 or p >= 1:
            raise ValueError("The value of p must be in the interval (0,1)")

        edge_label = kwargs.get('edge_label', None)
        edge_attribute_label = kwargs.get('edge_attribute_label', None)
        edge_attribute_threshold = kwargs.get('edge_attribute_threshold', None)
        attribute_is_datetime = kwargs.get('attribute_is_datetime', None)

        if edge_label is not None:
            if edge_attribute_label and edge_attribute_threshold and not attribute_is_datetime:
                raise ValueError("You can only split by datetime edge attribute")
            else:  # all three are True
                print("Call a method that does the splitting  on heterogeneous graph")
                edge_data_ids, edge_data_labels = self._train_test_split_heterogeneous(p=p,
                                                                                       method=method,
                                                                                       edge_label=edge_label,
                                                                                       edge_attribute_label=edge_attribute_label,
                                                                                       edge_attribute_threshold=edge_attribute_threshold)
        else:
            # treats all edge types equally
            edge_data_ids, edge_data_labels = self._train_test_split_homogeneous(p=p,
                                                                                 method=method,
                                                                                 kwargs=kwargs)

        return self.g_train, edge_data_ids, edge_data_labels

    def _get_edges(self, edge_label, edge_attribute_label=None, edge_attribute_threshold=None):
        # the graph in networkx format is stored in self.g_train
        all_edges = self.g.edges(data=True)
        if edge_attribute_label is None or edge_attribute_threshold is None:
            # select those edges with edge_label
            edges_with_label = [e for e in all_edges if e[2]['label'] == edge_label]
        elif edge_attribute_threshold is not None and edge_attribute_threshold is not None:
            edge_attribute_threshold_dt = datetime.datetime.strptime(edge_attribute_threshold, '%d/%m/%Y')
            edges_with_label = [e for e in all_edges if (e[2]['label'] == edge_label
                                and datetime.datetime.strptime(e[2][edge_attribute_label], '%d/%m/%Y') > edge_attribute_threshold_dt)]
        else:
            raise ValueError("Invalid parameters")  # not the most informative error!

        return edges_with_label

    def _get_edge_source_and_target_node_types(self, edges):

        all_nodes = self.g_train.nodes(data=True)
        all_nodes_as_dict = { n[0]: n[1] for n in all_nodes}
        edge_node_types = set()
        for edge in edges:
            edge_node_types.add((all_nodes_as_dict[edge[0]]['label'], all_nodes_as_dict[edge[1]]['label']))

        return edge_node_types

    def _reduce_graph_by_edge_type_and_attribute(self,
                                                 minedges,
                                                 p=0.5,
                                                 edge_label=None,
                                                 edge_attribute_label=None,
                                                 edge_attribute_threshold=None):
        """
        Reduces the graph G by a factor p by removing existing edges not on minedges list such that the reduced tree
        remains connected. The edges removed must of the type specified by edge_label.

        :param minedges: spanning tree edges that cannot be removed
        :param p: factor by which to reduce the size of the graph
        :param edge_label: <str> The edge type to consider
        :param attribute_label: <str> The edge attribute to consider
        :param attribute_threshold: <str> The threshold value; only edges with attribute value larger than the
        threshold can be removed
        :return: Returns the list of edges removed from G (also modifies G by removing said edges)
        """
        if edge_label is None:
            raise ValueError("edge_label cannot be None")
        if edge_attribute_threshold is None:
            raise ValueError("attribute_threshold cannot be None")

        # copy the original graph and start over in case this is not the first time
        # reduce_graph has been called.
        self.g_train = self.g.copy()

        # determine the number of edges in the graph that have edge_label type and edge attribute value
        # large than attribute_threshold
        # Multiply this number by p to determine the number of positive edge examples to sample
        all_edges = self._get_edges(edge_label=edge_label,
                                    edge_attribute_label=edge_attribute_label,
                                    edge_attribute_threshold=edge_attribute_threshold)
        num_edges_total = len(all_edges)
        print("Network has {} edges of type {}".format(num_edges_total, edge_label))
        #
        num_edges_to_remove = int(num_edges_total * p)
        # shuffle the edges
        np.random.shuffle(all_edges)
        #
        # iterate over the list of edges and for each edge if the edge is not in minedges, remove it from the graph
        # until num_edges_to_remove edges have been removed and the graph reduced to p of its original size
        count = 0
        removed_edges = []
        for edge in all_edges:
            edge_uv = (edge[0], edge[1])
            edge_vu = (edge[1], edge[0])
            if (edge_uv not in minedges) and (edge_vu not in minedges):  # for sanity check (u,v) and (v,u)
                removed_edges.append((edge[0], edge[1], 1))  # the last entry is the label
                self.g_train.remove_edge(edge[0], edge[1])
                count += 1
                if count % 1000 == 0:
                    print("Removed", count, "edges")
            if count == num_edges_to_remove:
                return removed_edges

        return removed_edges

    def _reduce_graph_by_edge_type(self, minedges, p=0.5, edge_label=None):
        """
        Reduces the graph G by a factor p by removing existing edges not on minedges list such that the reduced tree
        remains connected. The edges removed must of the type specified by edge_label.

        :param minedges: spanning tree edges that cannot be removed
        :param p: factor by which to reduce the size of the graph
        :param edge_label: <str> The edge type to consider
        :return: Returns the list of edges removed from G (also modifies G by removing said edges)
        """
        if edge_label is None:
            raise ValueError("edge_label cannot be None")

        # copy the original graph and start over in case this is not the first time
        # reduce_graph has been called.
        self.g_train = self.g.copy()

        # determine the number of edges in the graph that have edge_label type
        # Multiply this number by p to determine the number of positive edge examples to sample
        all_edges = self._get_edges(edge_label=edge_label)
        num_edges_total = len(all_edges)
        print("Network has {} edges of type {}".format(num_edges_total, edge_label))
        #
        num_edges_to_remove = int(num_edges_total * p)
        # shuffle the edges
        np.random.shuffle(all_edges)
        #
        # iterate over the list of edges and for each edge if the edge is not in minedges, remove it from the graph
        # until num_edges_to_remove edges have been removed and the graph reduced to p of its original size
        count = 0
        removed_edges = []
        for edge in all_edges:
            edge_uv = (edge[0], edge[1])
            edge_vu = (edge[1], edge[0])
            if (edge_uv not in minedges) and (edge_vu not in minedges):  # for sanity check (u,v) and (v,u)
                removed_edges.append((edge[0], edge[1], 1))  # the last entry is the label
                self.g_train.remove_edge(edge[0], edge[1])
                count += 1
                if count % 1000 == 0:
                    print("Removed", count, "edges")
            if count == num_edges_to_remove:
                return removed_edges

        return removed_edges

    def _reduce_graph(self, minedges, p=0.5):
        """
        Reduces the graph G by a factor p by removing existing edges not on minedges list such that the reduced tree
        remains connected.

        :param minedges: spanning tree edges that cannot be removed
        :param p: factor by which to reduce the size of the graph
        :return: Returns the list of edges removed from G (also modifies G by removing said edges)
        """
        # copy the original graph and start over in case this is not the first time
        # reduce_graph has been called.
        self.g_train = self.g.copy()

        num_edges_to_remove = int((self.g_train.number_of_edges() - len(minedges)) * p)
        all_edges = self.g_train.edges()
        # shuffle the edges
        np.random.shuffle(all_edges)
        # iterate over the list of edges and for each edge if the edge is not in minedges, remove it from the graph
        # until num_edges_to_remove edges have been removed and the graph reduced to p of its original size
        count = 0
        removed_edges = []
        for edge_uv in all_edges:
            edge_vu = (edge_uv[1], edge_uv[0])
            if (edge_uv not in minedges) and (edge_vu not in minedges):  # for sanity check (u,v) and (v,u)
                removed_edges.append((edge_uv[0], edge_uv[1], 1))  # the last entry is the label
                self.g_train.remove_edge(edge_uv[0], edge_uv[1])
                count += 1
                if count % 1000 == 0:
                    print("Removed", count, "edges")
            if count == num_edges_to_remove:
                return removed_edges

    def _sample_negative_examples_by_edge_type_local_dfs(self,
                                                         p=0.5,
                                                         probs=None,
                                                         edges_positive=None,
                                                         edge_label=None,
                                                         limit_samples=None):
        """
        It generates a list of edges that don't exist in graph G. The number of edges is equal to the number of edges in
        G times p (that should be in the range (0,1] or limited to limit_samples if the latter is not None. This method
        uses depth-first search to efficiently sample negative edges based on the local neighbourhood of randomly
        (uniformly) sampled source nodes at distances defined by the probabilities in probs. The graph G is not
        modified.
        :param p: factor that multiplies the number of edges in the graph and determines the number of no-edges to
            be sampled.
        :param probs:
        :param edges: <list> The positive edge examples
        :param edge_label: <str> The edge type to sample negative examples of
        :param limit_samples: int or None, it limits the maximum number of samples to the given number
        :return: Up to num_edges_to_sample*p edges that don't exist in the graph
        """
        if probs is None:
            probs = [0.0, 0.25, 0.50, 0.25]
            print("Warning: Using default sampling probabilities up to 3 hops from source node with values {}".format(probs))

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

        # if self.minedges is None:
        #     num_edges_to_sample = int(self.g.number_of_edges() * p)
        # else:
        #     num_edges_to_sample = int((self.g.number_of_edges() - len(self.minedges)) * p)
        if limit_samples is not None:
            if num_edges_to_sample > limit_samples:
                num_edges_to_sample = limit_samples

        edge_source_target_node_types = self._get_edge_source_and_target_node_types(edges=edges_positive)

        if self.g_master is None:
            edges = self.g.edges()
        else:
            edges = self.g_master.edges()

        start_nodes = self.g.nodes(data=True)
        nodes_dict = {node[0]: node[1]['label'] for node in start_nodes}

        count = 0
        sampled_edges = []

        num_iter = int(np.ceil(num_edges_to_sample / (1.0*len(start_nodes))))+1

        for _ in np.arange(0, num_iter):
            np.random.shuffle(start_nodes)
            # sample the distance to the target node using probs
            target_node_distances = np.random.choice(n, len(start_nodes), p=probs)+1
            for u, d in zip(start_nodes, target_node_distances):
                # perform DFS search up to d distance from the start node u.
                visited = {node[0]: False for node in start_nodes}
                nodes_stack = list()
                # start at node u
                nodes_stack.append((u[0], 0))  # tuple is node, depth
                while len(nodes_stack) > 0:
                    next_node = nodes_stack.pop()
                    v = next_node[0]   # retrieve node id
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
                            if (u_v_edge_type in edge_source_target_node_types) and (u[0] != v) and \
                                    ((u[0], v) not in edges) and ((v, u[0]) not in edges) and \
                                    ((u[0], v, 0) not in sampled_edges) and ((v, u[0], 0) not in sampled_edges):
                                sampled_edges.append((u[0], v, 0))  # the last entry is the class label
                                count += 1
                                self.negative_edge_node_distances.append(d)
                                break
                        elif dv < d:
                            neighbours = nx.neighbors(self.g, v)
                            np.random.shuffle(neighbours)
                            neighbours = [(k, dv+1) for k in neighbours]
                            nodes_stack.extend(neighbours)

                if count == num_edges_to_sample:
                    return sampled_edges

        return sampled_edges


    def _sample_negative_examples_local_dfs(self, p=0.5, probs=None, limit_samples=None):
        """
        It generates a list of edges that don't exist in graph G. The number of edges is equal to the number of edges in
        G times p (that should be in the range (0,1] or limited to limit_samples if the latter is not None. This method
        uses depth-first search to efficiently sample negative edges based on the local neighbourhood of randomly
        (uniformly) sampled source nodes at distances defined by the probabilities in probs. The graph G is not
        modified.
        :param G: The graph to sample edges from (NetworkX type graph)
        :param p: factor that multiplies the number of edges in the graph and determines the number of no-edges to
            be sampled.
        :param limit_samples: int or None, it limits the maximum number of samples to the given number
        :return: Up to num_edges_to_sample*p edges that don't exist in the graph
        """
        if probs is None:
            probs = [0.0, 0.25, 0.50, 0.25]
            print("Warning: Using default sampling probabilities up to 3 hops from source node with values {}".format(probs))

        if not isclose(sum(probs), 1.0):
            raise ValueError("Sampling probabilities do not sum to 1")

        self.negative_edge_node_distances = []
        n = len(probs)

        if self.minedges is None:
            num_edges_to_sample = int(self.g.number_of_edges() * p)
        else:
            num_edges_to_sample = int((self.g.number_of_edges() - len(self.minedges)) * p)

        if limit_samples is not None:
            if num_edges_to_sample > limit_samples:
                num_edges_to_sample = limit_samples

        if self.g_master is None:
            edges = self.g.edges()
        else:
            edges = self.g_master.edges()

        start_nodes = self.g.nodes(data=False)

        count = 0
        sampled_edges = []

        num_iter = int(np.ceil(num_edges_to_sample / (1.0*len(start_nodes))))

        for _ in np.arange(0, num_iter):
            np.random.shuffle(start_nodes)
            # sample the distance to the target node using probs
            target_node_distances = np.random.choice(n, len(start_nodes), p=probs)+1
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
                            if (u != v) and ((u, v) not in edges) and ((v, u) not in edges) and \
                                    ((u, v, 0) not in sampled_edges) and ((v, u, 0) not in sampled_edges):
                                sampled_edges.append((u, v, 0))  # the last entry is the class label
                                count += 1
                                self.negative_edge_node_distances.append(d)
                                break
                        elif dv < d:
                            neighbours = nx.neighbors(self.g, v)
                            np.random.shuffle(neighbours)
                            neighbours = [(k, dv+1) for k in neighbours]
                            nodes_stack.extend(neighbours)

                if count == num_edges_to_sample:
                    return sampled_edges

        return sampled_edges

    def _sample_negative_examples_local_bfs(self, p=0.5, probs=None, limit_samples=None):
        """
        It generates a list of edges that don't exist in graph G. The number of edges is equal to the number of edges in
        G times p (that should be in the range (0,1] or limited to limit_samples if the latter is not None. This method
        uses breadth-first search to sample negative edges based on the local neighbourhood of randomly (uniformly)
        sampled source nodes at distances defined by the probabilities in probs. G is not modified.
        :param G: The graph to sample negative edges from (NetworkX type graph)
        :param p: factor that multiplies the number of edges in the graph and determines the number of no-edges to
            be sampled.
        :param limit_samples: int or None, it limits the maximum number of sample to the given number
        :return: Up to num_edges_to_sample*p or limit_samples edges that don't exist in the graph
        """
        if probs is None:
            probs = [0.0, 0.25, 0.50, 0.25]
            print("Warning: Using default sampling probabilities up to 3 hops from source node with values {}".format(probs))

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

        start_nodes = self.g.nodes(data=False)

        count = 0
        sampled_edges = list()

        num_iter = int(np.ceil(num_edges_to_sample / (1.0*len(start_nodes))))

        for _ in np.arange(0, num_iter):
            np.random.shuffle(start_nodes)

            target_node_distances = np.random.choice(n, len(start_nodes), p=probs)+1
            for u, d in zip(start_nodes, target_node_distances):
                # collect all the nodes that are d links away from u
                nodes_at_frontier = list()
                nodes_at_frontier.append(u)
                for _ in np.arange(d):
                    next_level_nodes = list()
                    if len(nodes_at_frontier) == 0:
                        break
                    [next_level_nodes.extend(nx.neighbors(self.g, n)) for n in nodes_at_frontier]
                    nodes_at_frontier = next_level_nodes
                if len(nodes_at_frontier) == 0:
                    break
                # check if u, v where v in nodes_at_frontier have an edge. The first pair that has no edge in the graph
                # becomes a negative sample
                np.random.shuffle(nodes_at_frontier)
                for v in nodes_at_frontier:
                    if (u != v) and ((u, v) not in edges) and ((v, u) not in edges) and \
                                    ((u, v, 0) not in sampled_edges) and ((v, u, 0) not in sampled_edges):
                        sampled_edges.append((u, v, 0))  # the last entry is the class label
                        count += 1
                        self.negative_edge_node_distances.append(d)
                        break

                if count == num_edges_to_sample:
                    return sampled_edges

        return sampled_edges

    def _sample_negative_examples_global(self, p=0.5, limit_samples=None):
        """
        It generates a list of edges that don't exist in graph G. The number of edges is equal to the number of edges in
        G times p (that should be in the range (0,1] or limited to limit_samples if the latter is not None. The graph
        G is not modified. This method samples uniformly at random nodes from the graph and, if they don't have an
        edge in the graph, records the pair as a negative edge.
        :param G: The graph to sample from (NetworkX type graph)
        :param p: factor that multiplies the number of edges in the graph and determines the number of no-edges to
            be sampled.
        :param limit_samples: int or None, it limits the maximum number of sample to the given number
        :return: Up to num_edges_to_sample*p or limit_samples edges that don't exist in the graph
        """
        self.negative_edge_node_distances = []

        if self.minedges is None:
            num_edges_to_sample = int(self.g.number_of_edges() * p)
        else:
            num_edges_to_sample = int((self.g.number_of_edges() - len(self.minedges)) * p)

        if limit_samples is not None:
            if num_edges_to_sample > limit_samples:
                num_edges_to_sample = limit_samples

        if self.g_master is None:
            edges = self.g.edges()
        else:
            edges = self.g_master.edges()

        start_nodes = self.g.nodes(data=False)
        end_nodes = self.g.nodes(data=False)

        count = 0
        sampled_edges = []

        num_iter = int(np.ceil(num_edges_to_sample / (1.0*len(start_nodes)))) + 1

        for _ in np.arange(0, num_iter):
            np.random.shuffle(start_nodes)
            np.random.shuffle(end_nodes)
            for u, v in zip(start_nodes, end_nodes):
                if (u != v) and ((u, v) not in edges) and ((v, u) not in edges) and \
                        ((u, v, 0) not in sampled_edges) and ((v, u, 0) not in sampled_edges):
                    sampled_edges.append((u, v, 0))  # the last entry is the class label
                    count += 1
                    self.negative_edge_node_distances.append(nx.shortest_path_length(self.g,
                                                                                     source=u,
                                                                                     target=v))
                if count == num_edges_to_sample:
                    return sampled_edges

        return sampled_edges

    def _sample_negative_examples_by_edge_type_global(self, edges, edge_label, p=0.5, limit_samples=None):
        """
        It generates a list of edges that don't exist in graph G. The number of edges is equal to the number of edges in
        G times p (that should be in the range (0,1] or limited to limit_samples if the latter is not None. The graph
        G is not modified. This method samples uniformly at random nodes from the graph and, if they don't have an
        edge in the graph, records the pair as a negative edge. However, this method makes certain that the negative
        edges are between node types of interest for edge prediction in heterogeneous graphs.
        :param edges: The positive edge samples; it is used to infer the types of source and target nodes to sample for
        negative examples.
        :param p: factor that multiplies the number of edges in the graph and determines the number of no-edges to
            be sampled.
        :param limit_samples: int or None, it limits the maximum number of sample to the given number
        :return: Up to num_edges_to_sample*p or limit_samples edges that don't exist in the graph
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

        edge_source_target_node_types = self._get_edge_source_and_target_node_types(edges=edges)

        start_nodes = self.g.nodes(data=True)
        end_nodes = self.g.nodes(data=True)

        count = 0
        sampled_edges = []

        num_iter = int(np.ceil(num_edges_to_sample / (1.0*len(start_nodes)))) + 1

        for _ in np.arange(0, num_iter):
            np.random.shuffle(start_nodes)
            np.random.shuffle(end_nodes)
            for u, v in zip(start_nodes, end_nodes):
                u_v_edge_type = (u[1]['label'], v[1]['label'])
                if (u_v_edge_type in edge_source_target_node_types) and (u != v) and ((u[0], v[0]) not in edges) \
                        and ((v[0], u[0]) not in edges) and ((u[0], v[0], 0) not in sampled_edges) \
                        and ((v[0], u[0], 0) not in sampled_edges):
                    sampled_edges.append((u[0], v[0], 0))  # the last entry is the class label
                    count += 1
                    self.negative_edge_node_distances.append(nx.shortest_path_length(self.g,
                                                                                     source=u[0],
                                                                                     target=v[0]))
                    if count % 1000 == 0:
                        print("Sampled", count, "negative edges")

                    if count == num_edges_to_sample:
                        return sampled_edges

        return sampled_edges

    def _get_minimum_spanning_edges(self):
        """
        Given an undirected graph, it calculates the minimum set of edges such that graph connectivity is preserved
        :return: The minimum spanning edges of the undirected graph G
        """
        mst = nx.minimum_spanning_edges(self.g, data=False)
        edges = list(mst)
        return edges
