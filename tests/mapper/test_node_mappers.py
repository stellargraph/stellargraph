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
g
"""
from stellar.mapper.node_mappers import *
from stellar.data.explorer import SampledBreadthFirstWalk, UniformRandomWalk

import networkx as nx
import random
import numpy as np
import itertools as it
import pytest


def create_test_sampler(G):
    def neigh_samples(head_nodes, current_per_hop):
        if len(current_per_hop) < 1:
            return list(head_nodes)

        num_for_hop, next_samples = current_per_hop[0], current_per_hop[1:]
        neighbour_samples = it.chain(
            *[random.choices(list(G.neighbors(hn)), k=num_for_hop) for hn in head_nodes]
        )
        return head_nodes + neigh_samples(list(neighbour_samples), next_samples)

    class Sampler(SampledBreadthFirstWalk):
        def run(self, nodes=[], n=1, n_size=[]):
            return [neigh_samples([hn], n_size) for hn in nodes]

    return Sampler(G)


def test_graphsage_constructor():
    n_feat = 4

    # test graph
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    mapper = GraphSAGENodeMapper(
        G, G.nodes(), create_test_sampler(G), batch_size=2, num_samples=[2, 2]
    )
    assert mapper.batch_size == 2
    assert mapper.data_size == 4
    assert len(mapper.ids) == 4


def test_graphsage_1():
    n_feat = 4
    n_batch = 2

    # test graph
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add example features
    for v in G.nodes():
        G.node[v]["feature"] = np.ones(n_feat)

    mapper = GraphSAGENodeMapper(
        G, G.nodes(), create_test_sampler(G), batch_size=n_batch, num_samples=[2, 2]
    )

    assert len(mapper) == 2

    for ii in range(2):
        nf, nl = mapper[ii]
        assert len(nf) == 3
        assert nf[0].shape == (n_batch, 1, n_feat)
        assert nf[1].shape == (n_batch, 2, n_feat)
        assert nf[2].shape == (n_batch, 2 * 2, n_feat)
        assert nl is None

    with pytest.raises(IndexError):
        nf, nl = mapper[2]


def test_graphsage_wrong_sampler():
    n_feat = 4
    n_batch = 2

    # test graph
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add example features
    for v in G.nodes():
        G.node[v]["feature"] = np.ones(n_feat)

    with pytest.raises(TypeError):
        mapper = GraphSAGENodeMapper(
            G, G.nodes(), UniformRandomWalk(G), batch_size=n_batch, num_samples=[2, 2]
        )


def test_graphsage_2():
    n_feat = 4
    n_batch = 2

    # test graph
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add example features
    for v in G.nodes():
        G.node[v]["feature"] = np.ones(n_feat)

    with pytest.raises(RuntimeWarning):
        GraphSAGENodeMapper(
            G,
            G.nodes(),
            create_test_sampler(G),
            batch_size=n_batch,
            num_samples=[2, 2],
            feature_size=8,
        )


def test_graphsage_3():
    n_feat = 4
    n_batch = 2

    # test graph
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add example features
    for v in G.nodes():
        G.node[v]["feature"] = np.ones(n_feat)

    mapper = GraphSAGENodeMapper(
        G, G.nodes(), create_test_sampler(G), batch_size=n_batch, num_samples=[0]
    )

    assert len(mapper) == 2

    for ii in range(1):
        nf, nl = mapper[ii]
        assert len(nf) == 2
        assert nf[0].shape == (n_batch, 1, n_feat)
        assert nf[1].shape == (n_batch, 1, n_feat)
        assert nl is None


def test_graphsage_no_samples():
    n_feat = 4
    n_batch = 2

    # test graph
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add example features
    for v in G.nodes():
        G.node[v]["feature"] = np.ones(n_feat)

    mapper = GraphSAGENodeMapper(
        G, G.nodes(), create_test_sampler(G), batch_size=n_batch, num_samples=[]
    )

    assert len(mapper) == 2
    nf, nl = mapper[0]

    assert len(nf) == 1
    assert nl is None
