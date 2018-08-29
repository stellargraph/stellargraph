# -*- coding: utf-8 -*-
#
# Copyright 2018 Data61, CSIRO
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


"""
Mapper tests:

GraphSAGENodeMapper(
        G: nx.Graph,
        ids: List[Any],
        sampler: Callable[[List[Any]], List[List[Any]]],
        batch_size: int,
        num_samples: List[int],
        target_id: AnyStr = None,
        feature_size: Optional[int] = None,
        name: AnyStr = None,
    )
"""
from stellargraph.data.stellargraph import *
from stellargraph.data.utils import *

import networkx as nx
import random
import numpy as np
import itertools as it
import pytest


def example_nx_graph_1(feature_size=None):
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add some node attributes
    if feature_size is not None:
        for v in G.nodes():
            for ii in range(feature_size):
                G.node[v]["a%d" % ii] = np.random.randn()
    return G


def example_stellar_graph_1(feature_size=None):
    G = StellarGraph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add some node attributes
    if feature_size is not None:
        for v in G.nodes():
            for ii in range(feature_size):
                G.node[v]["a%d" % ii] = np.random.randn()
    return G


def example_stellar_graph_2():
    G = StellarGraph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add some node attributes
    G.node[1]["a1"] = 1
    G.node[3]["a1"] = 1
    G.node[1]["a2"] = 1
    G.node[4]["a2"] = 1
    G.node[3]["a3"] = 1
    return G


def example_stellar_graph_2():
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add some node attributes
    G.node[1]["a1"] = 1
    G.node[1]["a2"] = 1
    G.node[3]["a1"] = 1
    G.node[3]["a3"] = 1
    G.node[4]["a2"] = 1

    # Add target attributes
    G.node[1]["tar"] = 1
    G.node[3]["tar"] = 3
    G.node[4]["tar"] = 4
    return G


def test_feature_converter_no_attrs():
    G = example_stellar_graph_1()

    nfc = NodeFeatureConverter(G, G)

    for n, ndata in G.nodes(data=True):
        assert "feature" in ndata
        assert ndata["feature"].shape == (0,)

    assert nfc.feature_size == 0


def test_feature_converter():
    n_feat = 4
    G = example_stellar_graph_1(n_feat)
    nfc = NodeFeatureConverter(G, G)

    for n, ndata in G.nodes(data=True):
        assert "feature" in ndata
        assert ndata["feature"].shape == (n_feat,)

    assert nfc.feature_list == ["a0", "a1", "a2", "a3"]
    assert nfc.feature_size == n_feat


def test_feature_converter_incomplete_attrs():
    G = example_stellar_graph_2()
    nfc = NodeFeatureConverter(G, G, ignored_attributes=["tar"])

    for n, ndata in G.nodes(data=True):
        assert "feature" in ndata
        assert ndata["feature"].shape == (3,)

    assert all(G.node[1]["feature"] == [1, 1, 0])
    assert all(G.node[2]["feature"] == [0, 0, 0])
    assert all(G.node[3]["feature"] == [1, 0, 1])
    assert all(G.node[4]["feature"] == [0, 1, 0])

    assert nfc.feature_size == 3


def test_target_converter_partial_direct():
    G = example_stellar_graph_2()
    ntc = NodeTargetConverter(G, target="tar")

    labels = ntc.get_targets_for_ids([1, 3, 4])
    assert list(labels) == [1, 3, 4]

    labels = ntc.get_targets_for_ids([1, 2])
    assert list(labels) == [1, None]


def test_target_converter_partial_category():
    G = example_stellar_graph_2()
    ntc = NodeTargetConverter(G, target="tar", target_type="categorical")
    assert ntc.target_category_values == [1, 3, 4]

    labels = ntc.get_targets_for_ids([1, 3, 4])
    assert list(labels) == [0, 1, 2]

    labels = ntc.get_targets_for_ids([1, 2])
    assert list(labels) == [0, None]


def test_target_converter_1hot():
    G = example_stellar_graph_2()
    ntc = NodeTargetConverter(G, target="tar", target_type="1hot")
    assert ntc.target_category_values == [1, 3, 4]

    labels = ntc.get_targets_for_ids([1, 3, 4])
    assert np.all(labels == np.eye(3))
