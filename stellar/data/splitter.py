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


class DataSplitter(object):
    def __init__(self, graph, graph_master=None):
        self.graph = graph
        self.graph_master = graph_master
        print("Called DataSplitter.__init__()")


    def train_test_split(self, p=0.25, **kwargs):
        pass  # must be implemented by subclass


class EdgeSplitter(DataSplitter):

    def train_test_split(self, p=0.25, **kwargs):  # implements base class
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
        pass

    def _reduce_graph(self, minedges, p=0.5):
        """
        Reduces the graph G by a factor p by removing existing edges not on minedges list such that the reduced tree
        remains connected.

        :param minedges: spanning tree edges that cannot be removed
        :param p: factor by which to reduce the size of the graph
        :return: Returns the list of edges removed from G (also modifies G by removing said edges)
        """

    def _sample_negative_examples_local(self, p=0.5, probs=None, limit_samples=None):
        """
        It generates a list of edges that don't exist in graph G. The number of edges is equal to the number of edges in
        G times p (that should be in the range (0,1]. The graph G is not modified.
        :param G: The graph to reduce by factor p (NetworkX type graph)
        :param p: factor that multiplies the number of edges in the graph and determines the number of no-edges to
            be sampled.
        :param limit_samples: int or None, it limits the maximum number of sample to the given number
        :return: Up to num_edges_to_sample*p edges that don't exist in the graph
        """
        pass

    def _sample_negative_examples_global(self, p=0.5, limit_samples=None):
        """
        It generates a list of edges that don't exist in graph G. The number of edges is equal to the number of edges in
        G times p (that should be in the range (0,1]. The graph G is not modified.
        :param G: The graph to reduce by factor p (NetworkX type graph)
        :param p: factor that multiplies the number of edges in the graph and determines the number of no-edges to
            be sampled.
        :param limit_samples: int or None, it limits the maximum number of sample to the given number
        :return: Up to num_edges_to_sample*p edges that don't exist in the graph
        """
        pass

    def _get_minimum_spanning_edges(self):
        """
        Given an undirected graph, it calculates the minimum set of edges such that graph connectivity is preserved
        :return: The minimum spanning edges of the undirected graph G
        """
        pass


class NodeSplitter(DataSplitter):

    def train_test_split(self, p=0.25, **kwargs):  # implements base class
        """
        Splits the node data according to the scheme in Yang et al, ICML 2016, Revisiting semi-supervised learning
        with graph embeddings.
        Note: values for p, p_train, and p_validation should be set so that the total size of the 3 sets does not
        exceed the total number of nodes in the graph.
        :param p: Test set size is number of nodes in graph times p.
        :param p_train: Training set size as the number of nodes in graph times p_train. If None, then training set
        size is 1-p times the number of nodes in the graph.
        :param p_validation: Validation set size as the number of nodes in the graph times p_validation. If None, then
        the validation set is the empty set.
        :return: y_train, y_val, y_test, y_unlabeled
        """
        pass


