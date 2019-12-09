# -*- coding: utf-8 -*-
#
# Copyright 2019 Data61, CSIRO
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
import pandas as pd
import networkx as nx
import random
from stellargraph.core.graph import *
from stellargraph.core.graph_homogeneous import *
from ..test_utils.alloc import snapshot, allocation_benchmark


def example_benchmark_homogeneous_graph(n_nodes=100, n_edges=200, feature_size=None):

    edges = [
        (random.randint(0, n_nodes - 1), random.randint(0, n_nodes - 1))
        for _ in range(n_edges)
    ]
    edge_data = pd.DataFrame(edges, columns=["source", "target"])

    # Add example features
    if feature_size is None:
        node_features = None
        node_data = pd.DataFrame(np.arange(n_nodes), columns=["id"]).set_index("id")
    else:
        node_features = ["f{}".format(i) for i in range(feature_size)]
        node_data = pd.DataFrame(
            np.ones((n_nodes, feature_size)), columns=node_features
        )

    return edge_data, node_data, node_features


def to_networkx_stellargraph(edge_data, node_data, node_features):
    edge_data["label"] = "edge"
    Gnx = nx.from_pandas_edgelist(edge_data, edge_attr="label")
    Gnx.add_nodes_from(node_data.index)
    nx.set_node_attributes(Gnx, "node", "label")
    if node_features is not None:
        node_features = node_data[node_features]
    return StellarGraph(Gnx, node_features=node_features)


def to_homogeneous_stellargraph(edge_data, node_data, node_features):
    return HomogeneousStellarGraph(edge_data, node_data, node_features=node_features)


@pytest.mark.benchmark(group="NetworkX StellarGraph creation")
def test_benchmark_nextworkx_creation(benchmark):
    SAMPLE_SIZE = 50
    N_NODES = 500
    N_EDGES = 1000
    edge_data, node_data, node_features = example_benchmark_homogeneous_graph(
        N_NODES, N_EDGES, feature_size=10
    )

    def f():
        to_networkx_stellargraph(edge_data, node_data, node_features)

    benchmark(f)


@pytest.mark.benchmark(group="Homogeneous StellarGraph creation")
def test_benchmark_homogeneous_creation(benchmark):
    SAMPLE_SIZE = 50
    N_NODES = 500
    N_EDGES = 1000
    edge_data, node_data, node_features = example_benchmark_homogeneous_graph(
        N_NODES, N_EDGES, feature_size=10
    )

    def f():
        to_homogeneous_stellargraph(edge_data, node_data, node_features)

    benchmark(f)


@pytest.mark.benchmark(group="NetworkX StellarGraph neighbours")
def test_benchmark_networkx_get_neighbours(benchmark):
    sg = to_networkx_stellargraph(*example_benchmark_homogeneous_graph())
    num_nodes = sg.number_of_nodes()

    # Get the neighbours of every node in the graph
    def f():
        for i in range(num_nodes):
            sg.neighbors(i)

    benchmark(f)


@pytest.mark.benchmark(group="Homogeneous StellarGraph neighbours")
def test_benchmark_homogneous_get_neighbours(benchmark):
    sg = to_homogeneous_stellargraph(*example_benchmark_homogeneous_graph())
    num_nodes = sg.number_of_nodes()

    # Get the neighbours of every node in the graph
    def f():
        for i in range(num_nodes):
            sg.neighbour_nodes(i)

    benchmark(f)


@pytest.mark.benchmark(group="NetworkX StellarGraph node features")
def test_benchmark_nextworkx_get_features(benchmark):
    SAMPLE_SIZE = 50
    N_NODES = 500
    N_EDGES = 1000
    sg = to_networkx_stellargraph(
        *example_benchmark_homogeneous_graph(N_NODES, N_EDGES, feature_size=10)
    )

    def f():
        # Look up features for a random subset of the nodes
        all_ids = list(sg.nodes())
        selected_ids = random.choices(all_ids, k=SAMPLE_SIZE)
        sg.get_feature_for_nodes(selected_ids)

    benchmark(f)


@pytest.mark.benchmark(group="Homogeneous StellarGraph node features")
def test_benchmark_homogeneous_get_features(benchmark):
    SAMPLE_SIZE = 50
    N_NODES = 500
    N_EDGES = 1000
    sg = to_homogeneous_stellargraph(
        *example_benchmark_homogeneous_graph(N_NODES, N_EDGES, feature_size=10)
    )

    def f():
        # Look up features for a random subset of the nodes
        all_ids = list(sg.nodes())
        selected_ids = random.choices(all_ids, k=SAMPLE_SIZE)
        sg.node_features(selected_ids)

    benchmark(f)


@pytest.mark.benchmark(group="NetworkX StellarGraph memory", timer=snapshot)
@pytest.mark.parametrize("num_nodes,num_edges", [(100, 200), (1000, 5000)])
@pytest.mark.parametrize("feature_size", [None, 100])
def test_allocation_benchmark_networkx_creation(
    allocation_benchmark, feature_size, num_nodes, num_edges
):
    edge_data, node_data, node_features = example_benchmark_homogeneous_graph(
        num_nodes, num_edges, feature_size
    )

    def f():
        return to_networkx_stellargraph(edge_data, node_data, node_features)

    allocation_benchmark(f)


@pytest.mark.benchmark(group="Homogeneous StellarGraph memory", timer=snapshot)
@pytest.mark.parametrize("num_nodes,num_edges", [(0, 0), (100, 200), (1000, 5000)])
@pytest.mark.parametrize("feature_size", [None, 100])
def test_allocation_benchmark_homogeneous_creation(
    allocation_benchmark, feature_size, num_nodes, num_edges
):
    edge_data, node_data, node_features = example_benchmark_homogeneous_graph(
        num_nodes, num_edges, feature_size
    )

    def f():
        return to_homogeneous_stellargraph(edge_data, node_data, node_features)

    allocation_benchmark(f)
