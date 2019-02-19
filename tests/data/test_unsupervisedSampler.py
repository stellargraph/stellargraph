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

import pytest

import numpy as np
import networkx as nx
import keras

from stellargraph.data.unsupervisedSampler import UnsupervisedSampler
from stellargraph.core.graph import StellarGraph
from stellargraph.data.explorer import UniformRandomWalk
from stellargraph.mapper.link_mappers import *
from stellargraph.layer import GraphSAGE, link_classification


def create_test_graph():
    """
    Creates a simple graph for testing the unsupervised sampler

    :return: A simple graph with 10 nodes and 20 edges  in networkx format.
    """
    g = nx.Graph()
    edges = [
        (1, 3),
        (1, 4),
        (1, 9),
        (2, 5),
        (2, 7),
        (3, 6),
        (3, 8),
        (4, 2),
        (4, 8),
        (4, 10),
        (5, 7),
        (5, 9),
        (5, 10),
        (6, 1),
        (6, 7),
        (6, 9),
        (7, 3),
        (8, 2),
        (8, 10),
        (9, 10),
    ]

    g.add_edges_from(edges)

    return g


def example_Graph_1(feature_size=None):
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = np.ones(feature_size)

    G = StellarGraph(G, node_features="feature")
    return G


def example_Graph_2(feature_size=None):
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (4, 2)]
    G.add_edges_from(elist)

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = int(v) * np.ones(feature_size)

    G = StellarGraph(G, node_features="feature")
    return G


class TestUnsupervisedSampler(object):
    def test_parameter_checking(self):

        g = create_test_graph()
        rw = UniformRandomWalk(StellarGraph(g))

        # if no graph is provided
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=None, walker=rw)

        # graph has to be a Stellargraph object
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=g, walker=rw)

        g = StellarGraph(g)

        # only Uniform random walk is supported at the moment
        with pytest.raises(TypeError):
            UnsupervisedSampler(G=g, walker="any random walker")

        # if no walker is provided, default to Uniform Random Walk
        sampler = UnsupervisedSampler(G=g)
        assert isinstance(sampler.walker, UniformRandomWalk)

        """def test_generator_parameter(self):
            
        g = create_test_graph()
        g = StellarGraph(g)
        rw = UniformRandomWalk(g)
        sampler = UnsupervisedSampler(G=g, walker=rw)

        with pytest.raises(ValueError):
            sampler.generator(batch_size = None)
        
        with pytest.raises(ValueError):
            sampler.generator(batch_size = "x")
        
        with pytest.raises(ValueError):
            sampler.generator(batch_size = 0)
        """

    def test_generator(self):

        n_feat = 4
        batch_size = 2
        num_samples = [2, 2]
        layer_sizes = [2, 2]

        G = example_Graph_2(n_feat)
        rw = UniformRandomWalk(G)

        sampler = UnsupervisedSampler(G=G, walker=rw)
        
