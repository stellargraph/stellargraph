# -*- coding: utf-8 -*-
#
# Copyright 2017-2019 Data61, CSIRO
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

import random
import pytest
import networkx as nx
import pandas as pd
from stellargraph.mapper.node_mappers import DirectedGraphSAGENodeGenerator
from stellargraph.core.graph import StellarGraph, StellarDiGraph


def create_simple_graph():
    """
    Creates a simple directed graph for testing. The node ids are integers.

    Returns:
        A small, directed graph with 3 nodes and 2 edges in StellarDiGraph format.
    """

    g = nx.DiGraph()
    edges = [(1, 2), (2, 3)]
    g.add_edges_from(edges)
    nodes = list(g.nodes())
    features = [(node, -1.0 * node) for node in nodes]
    df = pd.DataFrame(features, columns=["id", "f0"]).set_index("id")
    return StellarDiGraph(g, node_features=df)


class TestDirectedNodeGenerator(object):
    """
    Test various aspects of the directed GrapohSAGE node generator, with the focus
    on the sampled neighbourhoods and the extracted features.
    """

    def sample_one_hop(self, num_in_samples, num_out_samples):
        g = create_simple_graph()
        nodes = list(g.nodes())

        in_samples = [num_in_samples]
        out_samples = [num_out_samples]
        gen = DirectedGraphSAGENodeGenerator(g, len(g), in_samples, out_samples)
        flow = gen.flow(node_ids=nodes, shuffle=False)

        # Obtain tree of sampled features
        features = flow.generator.sample_features(nodes, flow._sampling_schema)
        num_hops = len(in_samples)
        tree_len = 2 ** (num_hops + 1) - 1
        assert len(features) == tree_len

        # Check node features
        node_features = features[0]
        assert len(node_features) == len(nodes)
        assert node_features.shape == (len(nodes), 1, 1)
        for idx, node in enumerate(nodes):
            assert node_features[idx, 0, 0] == -1.0 * node

        # Check in-node features
        in_features = features[1]
        assert in_features.shape == (len(nodes), in_samples[0], 1)
        for n_idx in range(in_samples[0]):
            for idx, node in enumerate(nodes):
                if node == 1:
                    # None -> 1
                    assert in_features[idx, n_idx, 0] == 0.0
                elif node == 2:
                    # 1 -> 2
                    assert in_features[idx, n_idx, 0] == -1.0
                elif node == 3:
                    # 2 -> 3
                    assert in_features[idx, n_idx, 0] == -2.0
                else:
                    assert False

        # Check out-node features
        out_features = features[2]
        assert out_features.shape == (len(nodes), out_samples[0], 1)
        for n_idx in range(out_samples[0]):
            for idx, node in enumerate(nodes):
                if node == 1:
                    # 1 -> 2
                    assert out_features[idx, n_idx, 0] == -2.0
                elif node == 2:
                    # 2 -> 3
                    assert out_features[idx, n_idx, 0] == -3.0
                elif node == 3:
                    # 3 -> None
                    assert out_features[idx, n_idx, 0] == 0.0
                else:
                    assert False

    def test_one_hop(self):
        # Test 1 in-node and 1 out-node sampling
        self.sample_one_hop(1, 1)
        # Test 0 in-nodes and 1 out-node sampling
        self.sample_one_hop(0, 1)
        # Test 1 in-node and 0 out-nodes sampling
        self.sample_one_hop(1, 0)
        # Test 0 in-nodes and 0 out-nodes sampling
        self.sample_one_hop(0, 0)
        # Test 2 in-nodes and 3 out-nodes sampling
        self.sample_one_hop(2, 3)

    def test_two_hop(self):
        g = create_simple_graph()
        nodes = list(g.nodes())

        gen = DirectedGraphSAGENodeGenerator(
            g, batch_size=len(g), in_samples=[1, 1], out_samples=[1, 1]
        )
        flow = gen.flow(node_ids=nodes, shuffle=False)

        features = flow.generator.sample_features(nodes, flow._sampling_schema)
        num_hops = 2
        tree_len = 2 ** (num_hops + 1) - 1
        assert len(features) == tree_len

        # Check node features
        node_features = features[0]
        assert len(node_features) == len(nodes)
        assert node_features.shape == (len(nodes), 1, 1)
        for idx, node in enumerate(nodes):
            assert node_features[idx, 0, 0] == -1.0 * node

        # Check in-node features
        in_features = features[1]
        assert in_features.shape == (len(nodes), 1, 1)
        for idx, node in enumerate(nodes):
            if node == 1:
                # *None -> 1
                assert in_features[idx, 0, 0] == 0.0
            elif node == 2:
                # *1 -> 2
                assert in_features[idx, 0, 0] == -1.0
            elif node == 3:
                # *2 -> 3
                assert in_features[idx, 0, 0] == -2.0
            else:
                assert False

        # Check out-node features
        out_features = features[2]
        assert out_features.shape == (len(nodes), 1, 1)
        for idx, node in enumerate(nodes):
            if node == 1:
                # 1 -> *2
                assert out_features[idx, 0, 0] == -2.0
            elif node == 2:
                # 2 -> *3
                assert out_features[idx, 0, 0] == -3.0
            elif node == 3:
                # 3 -> *None
                assert out_features[idx, 0, 0] == 0.0
            else:
                assert False

        # Check in-in-node features
        in_features = features[3]
        assert in_features.shape == (len(nodes), 1, 1)
        for idx, node in enumerate(nodes):
            if node == 1:
                # *None -> None -> 1
                assert in_features[idx, 0, 0] == 0.0
            elif node == 2:
                # *None -> 1 -> 2
                assert in_features[idx, 0, 0] == 0.0
            elif node == 3:
                # *1 -> 2 -> 3
                assert in_features[idx, 0, 0] == -1.0
            else:
                assert False

        # Check in-out-node features
        in_features = features[4]
        assert in_features.shape == (len(nodes), 1, 1)
        for idx, node in enumerate(nodes):
            if node == 1:
                # *None <- None -> 1
                assert in_features[idx, 0, 0] == 0.0
            elif node == 2:
                # *2 <- 1 -> 2
                assert in_features[idx, 0, 0] == -2.0
            elif node == 3:
                # *3 <- 2 -> 3
                assert in_features[idx, 0, 0] == -3.0
            else:
                assert False

        # Check out-in-node features
        out_features = features[5]
        assert out_features.shape == (len(nodes), 1, 1)
        for idx, node in enumerate(nodes):
            if node == 1:
                # 1 -> 2 <- *1
                assert out_features[idx, 0, 0] == -1.0
            elif node == 2:
                # 2 -> 3 <- *2
                assert out_features[idx, 0, 0] == -2.0
            elif node == 3:
                # 3 -> None <- *None
                assert out_features[idx, 0, 0] == 0.0
            else:
                assert False

        # Check out-out-node features
        out_features = features[6]
        assert out_features.shape == (len(nodes), 1, 1)
        for idx, node in enumerate(nodes):
            if node == 1:
                # 1 -> 2 -> *3
                assert out_features[idx, 0, 0] == -3.0
            elif node == 2:
                # 2 -> 3 -> *None
                assert out_features[idx, 0, 0] == 0.0
            elif node == 3:
                # 3 -> None -> *None
                assert out_features[idx, 0, 0] == 0.0
            else:
                assert False
