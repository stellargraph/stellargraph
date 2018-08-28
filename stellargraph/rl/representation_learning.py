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


class RepresentationLearning(object):
    """
    Abstract base class for representation learning.
    All public methods should be overridden by sub-classes.
    """

    def __init__(self, graph):
        self.graph = graph

    def fit(self, graph, method="random-walk", **kwargs):  # Call fit to perform RL
        """

        :param graph: The graph
        :param method: Can be one of 'random-walk', 'biased-random-walk, 'gcn, 'fast-gcn', 'graph-sage', 'hin-sage'
        :param kwargs: Any other arguments specific to the RL method implemented in the sub-class
        :return:
        """
        pass


class NodeRepresentationLearning(RepresentationLearning):
    """
    Representation learning for nodes.
    """

    def fit(self, method="random-walk", **kwargs):
        """
        Given the selected method, e.g., random-walk, gcn, fast-gcn, graph-sage, it calculates features for each
        node in the graph.
        :param graph: The graph
        :param random_walk_explorer: Object, sub-class of GraphWalk, for random-walk method.
        :param gcn_learner: Object, for gcn-based methods
        :return:
        """
        pass


class EdgeRepresentationLearning(RepresentationLearning):
    """
    Representation learning for edges.
    """

    def fit(self, method="random-walk", **kwargs):
        """
        Given the selected method, e.g., random-walk, gcn, fast-gcn, graph-sage, it calculates features for each
        node in the graph.
        :param graph: The graph
        :param method: The method for calculating node features
        :param binary_operator: string type, one of 'avg', 'l1', 'l2', 'h'. The binary operator to apply to node
        features to calculate edge features (but what about gcn end-to-end approaches for link prediction, e.g., Decagon style?
        :param random_walk_explorer: Object, sub-class of GraphWalk, for random-walk method.
        :param gcn_learner: Object, for gcn-based methods
        :return:
        """
        pass
