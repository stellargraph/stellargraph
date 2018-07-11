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

GraphSAGELinkMapper(
        G: nx.Graph,
        ids: List[Any],
        link_labels: List[Any] or np.ndarray,
        batch_size: int,
        num_samples: List[int],
        feature_size: Optional[int] = None,
        name: AnyStr = None,
    )
g
"""
from stellar.mapper.link_mappers import *

import networkx as nx
import random
import numpy as np
import itertools as it
import pytest


def test_LinkMapper_constructor():
    n_feat = 4

    # test graph
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)
    edge_labels = [0] * G.number_of_edges()

    mapper = GraphSAGELinkMapper(
        G, G.edges(), edge_labels, batch_size=2, num_samples=[2, 2]
    )
    assert mapper.batch_size == 2
    assert mapper.data_size == 3
    assert len(mapper.ids) == 3

    G = nx.DiGraph()
    G.add_edges_from(elist)
    edge_labels = [0] * G.number_of_edges()
    mapper = GraphSAGELinkMapper(
        G, G.edges(), edge_labels, batch_size=2, num_samples=[2, 2]
    )
    assert mapper.batch_size == 2
    assert mapper.data_size == 4
    assert len(mapper.ids) == 4


def test_GraphSAGELinkMapper_1():
    n_feat = 4
    n_batch = 2

    # test graph
    G = nx.Graph()
    elist = [
        (1, 2),
        (2, 3),
        (1, 4),
        (4, 2),
    ]  # make sure the number of edges is divisible by n_batch, for this test to succeed
    G.add_edges_from(elist)
    data_size = G.number_of_edges()
    edge_labels = [0] * data_size

    # Add example features
    for v in G.nodes():
        G.node[v]["feature"] = np.ones(n_feat)

    mapper = GraphSAGELinkMapper(
        G, G.edges(), edge_labels, batch_size=n_batch, num_samples=[2, 2]
    )

    assert len(mapper) == 2

    for hop in range(2):
        nf, nl = mapper[hop]
        assert len(nf) == 3 * 2
        for ii in range(2):
            assert nf[ii].shape == (min(n_batch, data_size), 1, n_feat)
            assert nf[ii + 2].shape == (min(n_batch, data_size), 2, n_feat)
            assert nf[ii + 2 * 2].shape == (min(n_batch, data_size), 2 * 2, n_feat)
            assert nl == [0] * min(n_batch, data_size)

    with pytest.raises(IndexError):
        nf, nl = mapper[2]


def test_GraphSAGELinkMapper_2():
    n_feat = 4
    n_batch = 2

    # test graph
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)
    data_size = G.number_of_edges()
    edge_labels = [0] * data_size

    # Add example features
    for v in G.nodes():
        G.node[v]["feature"] = np.ones(n_feat)

    with pytest.raises(RuntimeWarning):
        GraphSAGELinkMapper(
            G,
            G.edges(),
            edge_labels,
            batch_size=n_batch,
            num_samples=[2, 2],
            feature_size=8,
        )


def test_GraphSAGELinkMapper_3():
    n_feat = 4
    n_batch = 2

    # test graph
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)
    data_size = G.number_of_edges()
    edge_labels = [0] * data_size

    # Add example features
    for v in G.nodes():
        G.node[v]["feature"] = np.ones(n_feat)

    mapper = GraphSAGELinkMapper(
        G, G.edges(), edge_labels, batch_size=n_batch, num_samples=[0]
    )

    assert len(mapper) == 2

    for ii in range(1):
        nf, nl = mapper[ii]
        assert len(nf) == 2 * 2
        for _ in range(len(nf)):
            assert nf[_].shape == (min(n_batch, data_size), 1, n_feat)
        assert nl == [0] * min(n_batch, data_size)


def test_GraphSAGELinkMapper_no_samples():
    n_feat = 4
    n_batch = 2

    # test graph
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (4, 2)]
    G.add_edges_from(elist)
    data_size = G.number_of_edges()
    edge_labels = [0] * data_size

    # Add example features
    for v in G.nodes():
        G.node[v]["feature"] = np.ones(n_feat)

    mapper = GraphSAGELinkMapper(
        G, G.edges(), edge_labels, batch_size=n_batch, num_samples=[]
    )

    assert len(mapper) == 2
    with pytest.raises(ValueError):
        nf, nl = mapper[0]
