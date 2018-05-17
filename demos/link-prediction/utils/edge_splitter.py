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

import networkx as nx
import numpy as np


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

        self.positive_edges = None
        self.negative_edges = None

        self.negative_edge_node_distances = None

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
        :return: The reduced graph (positive edges removed) and the edge data as numpy array with first two columns the
        node id defining the edge and the last column 1 or 0 for positive or negative example respectively.
        """

        # minedges are those edges that if removed we might end up with a disconnected graph after the positive edges
        # have been sampled.
        minedges = self._get_minimum_spanning_edges()
        self.positive_edges = self._reduce_graph(minedges=minedges, p=p)

        if method == 'global':
            self.negative_edges = self._sample_negative_examples_global(p=p,
                                                                        limit_samples=len(self.positive_edges))
        elif method == 'local':
            probs = kwargs.get('probs', [0.0, 0.25, 0.50, 0.25])
            print("Using sampling probabilities (distance from source node): {}".format(probs))
            self.negative_edges = self._sample_negative_examples_local_dfs(p=p,
                                                                           probs=probs,
                                                                           limit_samples=len(self.positive_edges))
        else:
            raise ValueError('Invalid method {}'.format(method))

        if len(self.positive_edges) == 0:
            raise Exception("Could not sample any positive edges")
        if len(self.negative_edges) == 0:
            raise Exception("Could not sample any negative edges")

        edge_data = np.vstack((self.positive_edges, self.negative_edges))
        print("** Sampled {} positive and {} negative edges. **".format(len(self.positive_edges),
                                                                        len(self.negative_edges)))

        return self.g_train, edge_data

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
        num_edges_to_sample = int(self.g.number_of_edges() * p)

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

        num_iter = int(np.ceil(num_edges_to_sample / (1.0*len(start_nodes))))

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

    def _get_minimum_spanning_edges(self):
        """
        Given an undirected graph, it calculates the minimum set of edges such that graph connectivity is preserved
        :return: The minimum spanning edges of the undirected graph G
        """
        mst = nx.minimum_spanning_edges(self.g, data=False)
        edges = list(mst)
        return edges
