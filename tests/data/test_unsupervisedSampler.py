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
import networkx as nx
from stellargraph.data.unsupervisedSampler import UnsupervisedSampler
from stellargraph.core.graph import StellarGraph
from stellargraph.data.explorer import BiasedRandomWalk, UniformRandomWalk


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


class TestUnsupervisedSampler(object):
    def test_parameter_checking(self):

        g = create_test_graph()
        rw = UniformRandomWalk(StellarGraph(g))

        # graph has to be a Stellargraph object
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=g, walker=rw, batch_size=10)

        g = StellarGraph(g)

        # only Uniform random walk is supported at the moment
        with pytest.raises(TypeError):
            UnsupervisedSampler(G=(g), walker=BiasedRandomWalk(g), batch_size=10)

        # batch_size must be provided to calculate the number of samples to generate per Epoch
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=StellarGraph(g), walker=rw, batch_size=None)

        # batch_size must be an integer
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=StellarGraph(g), walker=rw, batch_size="x")

        # batch_size must be greater than 0
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=StellarGraph(g), walker=rw, batch_size=-1)

    def test_generator(self):

        g = create_test_graph()
        g = StellarGraph(g)
        rw = UniformRandomWalk(g)

        sampler = UnsupervisedSampler(G=g, walker=rw, batch_size=10)
