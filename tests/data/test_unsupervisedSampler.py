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
    def test_UnsupervisedSampler_parameter(self):

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

        # walk must have length strictly greater than 1
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=g, walker=rw, length=1)

        # at least 1 walk from each root node
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=g, walker=rw, number_of_walks=0)

        # nodes nodes parameter should be an iterableof node IDs
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=g, walker=rw, nodes=1)

        # if no root nodes are provided for sampling defaulting to using all nodes as root nodes
        sampler = UnsupervisedSampler(G=g, walker=rw, nodes=None)
        assert sampler.nodes == g.nodes()

        # if the seed value is provided, it is set properly
        seed = 123
        sampler = UnsupervisedSampler(G=g, walker=rw, seed=seed)
        assert sampler.seed == seed

    def test_generator_parameter(self):

        g = create_test_graph()
        g = StellarGraph(g)
        rw = UniformRandomWalk(g)
        sampler = UnsupervisedSampler(G=g, walker=rw)

        # generator should be provided with a valid batch size. i.e. an integer >=1

        sample_gen = sampler.generator(batch_size=None)
        with pytest.raises(ValueError):
            next(sample_gen)

        sample_gen = sampler.generator(batch_size="x")
        with pytest.raises(TypeError):
            next(sample_gen)

        sample_gen = sampler.generator(batch_size=0)
        with pytest.raises(ValueError):
            next(sample_gen)

    def test_generator_samples(self):

        n_feat = 4
        batch_size = 4

        G = example_Graph_2(n_feat)
        rw = UniformRandomWalk(G)

        sampler = UnsupervisedSampler(G=G, walker=rw)

        sample_gen = sampler.generator(batch_size)

        samples = next(sample_gen)

        # return two lists: [(target,context)] pairs and [1/0] binary labels
        assert len(samples) == 2

        # each (target, context) pair has a matching label
        assert len(samples[0]) == len(samples[1])

        # batch-size number of samples are returned if batch_size is even
        assert len(samples[0]) == batch_size

        # batch-size +1 number of samples are returned if batch_size is odd
        batch_size = 5
        sample_gen = sampler.generator(batch_size)
        samples = next(sample_gen)
        assert len(samples[0]) == batch_size + 1

    def test_generator_multiple_batches(self):

        n_feat = 4
        batch_size = 4
        number_of_batches = 3

        G = example_Graph_2(n_feat)
        rw = UniformRandomWalk(G)

        sampler = UnsupervisedSampler(G=G, walker=rw)

        sample_gen = sampler.generator(batch_size)

        batches = []
        for batch in range(number_of_batches):
            batches.append(next(sample_gen))

        assert len(batches) == number_of_batches
