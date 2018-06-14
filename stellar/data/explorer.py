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


class GraphWalk(object):
    '''
    Base class for exploring graphs.
    '''

    def __init__(self, graph):
        self.graph = graph

    def run(self, **kwargs):
        '''
        To be overridden by subclasses. It is the main entry point for performing random walks on the given
        graph.
        It should return the sequences of nodes in each random walk.
        :param kwargs:
        :return:
        '''



class UniformRandomWalk(GraphWalk):
    '''
    Performs uniform random walks on the given graph
    '''

    def run(self, **kwargs):
        '''

        :param n: Total number of random walks
        :param l: Length of each random walk
        :param seed: Random number generator seed
        :parame e_types: List of edge types that the random walker is allowed to traverse. Set to None for homogeneous
        graphs with a single edge type.
        :return:
        '''
        pass


class BiasedRandomWalk(GraphWalk):
    '''
    Performs biased second order random walks (like Node2Vec random walks)
    '''
    def run(self, **kwargs):
        '''

        :param p: p parameter in Node2Vec
        :param q: q parameter in Node2Vec
        :param n: Number of random walks
        :param l: Length of random walks
        :param e_types: List of edge types that the random walk is allowed to traverse. Set to None for homogeneous
        graphs with a single edge type.
        :return:
        '''
        pass


class MetaPathWalk(GraphWalk):
    '''
    For heterogeneous graphs, it performs walks based on given metapaths.
    '''

    def run(self, **kwargs):
        '''

        :param n: Number of walks for each given metapath
        :param l: Length of random walks
        :param mp: List of metapaths to drive the random walks
        :return:
        '''
        pass


class DepthFirstWalk(GraphWalk):
    '''
    Depth First Walk that generates all paths from a starting node to a given depth.
    It can be used to extract, in a memory efficient way, a sub-graph starting from a node and up to a given depth.
    '''

    def run(self, **kwargs):
        '''

        :param n: Number of walks. If it is equal to the number of nodes in the graph, then it generates all the
        sub-graphs with every node as the root and up to the given depth, d.
        :param d: Depth of walk as in distance (number of edges) from starting node.
        :return: (The return value might differ when compared to the other walk types with exception to
        BreadthFirstWalk defined next)
        '''
        pass


class BreadthFirstWalk(GraphWalk):
    '''
    Breadth First Walk that generates all paths from a starting node to a given depth.
    It can be used to extract a sub-graph starting from a node and up to a given depth.
    '''

    def run(self, **kwargs):
        '''

        :param n: Number of walks. If it is equal to the number of nodes in the graph, then it generates all the
        sub-graphs with every node as the root and up to the given depth, d.
        :param d: Depth of walk as in distance (number of edges) from starting node.
        :return: (The return value might differ when compared to the other walk types with exception to
        DepthFirstWalk defined above)
        '''
        pass
