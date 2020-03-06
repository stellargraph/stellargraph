# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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

import random

from stellargraph.mapper import *
from stellargraph.core.graph import *

import numpy as np
import networkx as nx
import pytest

from ..test_utils.graphs import example_graph_random


def example_Graph_2(feature_size=None, n_nodes=100, n_edges=200):
    return example_graph_random(feature_size, n_edges, n_nodes, n_isolates=0)


@pytest.mark.benchmark(group="generator")
def test_benchmark_setup_generator_small(benchmark):
    n_feat = 1024
    n_edges = 100
    batch_size = 10
    num_samples = [20, 10]

    G = example_Graph_2(n_feat)
    nodes = list(G.nodes())
    edges_to_sample = np.reshape(random.choices(nodes, k=2 * n_edges), (n_edges, 2))

    def setup_generator():
        generator = GraphSAGELinkGenerator(
            G, batch_size=batch_size, num_samples=num_samples
        )

    benchmark(setup_generator)


@pytest.mark.benchmark(group="generator")
def test_benchmark_setup_generator_large(benchmark):
    n_feat = 1024
    n_edges = 100
    batch_size = 10
    num_samples = [20, 10]

    G = example_Graph_2(n_feat, 5000, 20000)
    nodes = list(G.nodes())
    edges_to_sample = np.reshape(random.choices(nodes, k=2 * n_edges), (n_edges, 2))

    def setup_generator():
        generator = GraphSAGELinkGenerator(
            G, batch_size=batch_size, num_samples=num_samples
        )

    benchmark(setup_generator)


@pytest.mark.benchmark(group="generator")
def test_benchmark_link_generator_small(benchmark):
    n_feat = 1024
    n_edges = 100
    batch_size = 10
    num_samples = [20, 10]

    G = example_Graph_2(n_feat)
    nodes = list(G.nodes())
    edges_to_sample = np.reshape(random.choices(nodes, k=2 * n_edges), (n_edges, 2))

    generator = GraphSAGELinkGenerator(
        G, batch_size=batch_size, num_samples=num_samples
    )

    def read_generator():
        gen = generator.flow(edges_to_sample)

        for ii in range(len(gen)):
            xf, xl = gen[ii]
        return xf, xl

    benchmark(read_generator)


@pytest.mark.benchmark(group="generator")
def test_benchmark_link_generator_large(benchmark):
    n_feat = 1024
    n_edges = 100
    batch_size = 10
    num_samples = [20, 10]

    G = example_Graph_2(n_feat, 5000, 20000)
    nodes = list(G.nodes())
    edges_to_sample = np.reshape(random.choices(nodes, k=2 * n_edges), (n_edges, 2))

    generator = GraphSAGELinkGenerator(
        G, batch_size=batch_size, num_samples=num_samples
    )

    def read_generator():
        gen = generator.flow(edges_to_sample)

        for ii in range(len(gen)):
            xf, xl = gen[ii]
        return xf, xl

    benchmark(read_generator)


@pytest.mark.benchmark(group="generator")
def test_benchmark_node_generator_small(benchmark):
    n_feat = 1024
    n_nodes = 200
    batch_size = 10
    num_samples = [20, 10]

    G = example_Graph_2(n_feat)
    nodes = list(G.nodes())
    nodes_to_sample = random.choices(nodes, k=n_nodes)

    generator = GraphSAGENodeGenerator(
        G, batch_size=batch_size, num_samples=num_samples
    )

    def read_generator():
        gen = generator.flow(nodes_to_sample)

        for ii in range(len(gen)):
            xf, xl = gen[ii]
        return xf, xl

    benchmark(read_generator)


@pytest.mark.benchmark(group="generator")
def test_benchmark_node_generator_large(benchmark):
    n_feat = 1024
    n_nodes = 200
    batch_size = 10
    num_samples = [20, 10]

    G = example_Graph_2(n_feat, 5000, 20000)
    nodes = list(G.nodes())
    nodes_to_sample = random.choices(nodes, k=n_nodes)

    generator = GraphSAGENodeGenerator(
        G, batch_size=batch_size, num_samples=num_samples
    )

    def read_generator():
        gen = generator.flow(nodes_to_sample)

        for ii in range(len(gen)):
            xf, xl = gen[ii]
        return xf, xl

    benchmark(read_generator)
