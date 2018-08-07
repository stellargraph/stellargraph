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
from stellar.data.stellargraph import *

import networkx as nx
import random
import numpy as np
import itertools as it
import pytest


def example_Graph_1(feature_size=None):
    G = StellarGraph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = np.ones(feature_size)
    return G


def example_DiGraph_1(feature_size=None):
    G = StellarDiGraph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = np.ones(feature_size)
    return G


def example_Graph_2(feature_size=None):
    G = StellarGraph()
    elist = [(1, 2), (2, 3), (1, 4), (4, 2)]
    G.add_edges_from(elist)

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = np.ones(feature_size)
    return G


def example_HIN_1(feature_size_by_type=None):
    G = StellarGraph()
    G.add_nodes_from([0, 1, 2, 3], label="movie")
    G.add_nodes_from([4, 5], label="user")
    G.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label="rating")
    G.add_edges_from([(4, 5)], label="friend")

    # Add example features
    if feature_size_by_type is not None:
        for v, vdata in G.nodes(data=True):
            nt = vdata["label"]
            vdata["feature"] = int(v) * np.ones(feature_size_by_type[nt], dtype="int")
    return G


def example_HIN_homo(feature_size_by_type=None):
    G = StellarGraph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5], label="user")
    G.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label="friend")

    # Add example features
    if feature_size_by_type is not None:
        for v, vdata in G.nodes(data=True):
            nt = vdata["label"]
            vdata["feature"] = int(v) * np.ones(feature_size_by_type[nt], dtype="int")
    return G


class Test_GraphSAGELinkMapper:
    """
    Tests of GraphSAGELinkMapper class
    """

    n_feat = 4
    batch_size = 2
    num_samples = [2, 2]

    def test_LinkMapper_constructor(self):

        G = example_Graph_1(self.n_feat)
        edge_labels = [0] * G.number_of_edges()

        mapper = GraphSAGELinkMapper(
            G,
            G.edges(),
            edge_labels,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
        )
        assert mapper.batch_size == self.batch_size
        assert mapper.data_size == G.number_of_edges()
        assert len(mapper.ids) == G.number_of_edges()

        G = example_DiGraph_1(self.n_feat)
        edge_labels = [0] * G.number_of_edges()
        mapper = GraphSAGELinkMapper(
            G,
            G.edges(),
            edge_labels,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
        )
        assert mapper.batch_size == self.batch_size
        assert mapper.data_size == G.number_of_edges()
        assert len(mapper.ids) == G.number_of_edges()

    def test_GraphSAGELinkMapper_1(self):

        G = example_Graph_2(self.n_feat)
        data_size = G.number_of_edges()
        edge_labels = [0] * data_size

        mapper = GraphSAGELinkMapper(
            G,
            G.edges(),
            edge_labels,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
        )

        assert len(mapper) == 2

        for batch in range(len(mapper)):
            nf, nl = mapper[batch]
            assert len(nf) == 3 * 2
            for ii in range(2):
                assert nf[ii].shape == (min(self.batch_size, data_size), 1, self.n_feat)
                assert nf[ii + 2].shape == (
                    min(self.batch_size, data_size),
                    2,
                    self.n_feat,
                )
                assert nf[ii + 2 * 2].shape == (
                    min(self.batch_size, data_size),
                    2 * 2,
                    self.n_feat,
                )
                assert nl == [0] * min(self.batch_size, data_size)

        with pytest.raises(IndexError):
            nf, nl = mapper[2]

    def test_GraphSAGELinkMapper_2(self):

        G = example_Graph_1(self.n_feat)
        data_size = G.number_of_edges()
        edge_labels = [0] * data_size

        with pytest.raises(RuntimeWarning):
            GraphSAGELinkMapper(
                G,
                G.edges(),
                edge_labels,
                batch_size=self.batch_size,
                num_samples=self.num_samples,
                feature_size=2 * self.n_feat,
            )

    def test_GraphSAGELinkMapper_3(self):

        G = example_Graph_1(self.n_feat)
        data_size = G.number_of_edges()
        edge_labels = [0] * data_size

        mapper = GraphSAGELinkMapper(
            G, G.edges(), edge_labels, batch_size=self.batch_size, num_samples=[0]
        )

        assert len(mapper) == 2

        for ii in range(1):
            nf, nl = mapper[ii]
            assert len(nf) == 2 * 2
            for _ in range(len(nf)):
                assert nf[_].shape == (min(self.batch_size, data_size), 1, self.n_feat)
            assert nl == [0] * min(self.batch_size, data_size)

    def test_GraphSAGELinkMapper_no_samples(self):
        """
        The SampledBFS sampler, created inside the mapper, currently throws a ValueError when the num_samples list is empty.
        This might change in the future, so this test might have to be re-written.

        """
        G = example_Graph_2(self.n_feat)
        data_size = G.number_of_edges()
        edge_labels = [0] * data_size

        mapper = GraphSAGELinkMapper(
            G, G.edges(), edge_labels, batch_size=self.batch_size, num_samples=[]
        )

        assert len(mapper) == 2
        with pytest.raises(ValueError):
            nf, nl = mapper[0]


class Test_HinSAGELinkMapper(object):
    """
    Tests of HinSAGELinkMapper class
    """

    n_feat = {"user": 5, "movie": 10}
    batch_size = 2
    num_samples = [2, 3]

    def test_LinkMapper_constructor(self):

        # Constructor with a homogeneous graph:
        G = example_HIN_homo(self.n_feat)
        links = [(1, 4), (1, 5), (0, 4), (5, 0)]  # ('user', 'user') links
        link_labels = [0] * len(links)

        mapper = HinSAGELinkMapper(
            G,
            links,
            link_labels,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            feature_size_by_type=self.n_feat,
        )

        assert mapper.batch_size == self.batch_size
        assert mapper.data_size == len(links)
        assert len(mapper.ids) == len(links)
        assert mapper.head_node_types == ("user", "user")

        # Constructor with a heterogeneous graph:
        G = example_HIN_1(self.n_feat)
        links = [(1, 4), (1, 5), (0, 4), (0, 5)]  # ('movie', 'user') links
        link_labels = [0] * len(links)

        mapper = HinSAGELinkMapper(
            G,
            links,
            link_labels,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            feature_size_by_type=self.n_feat,
        )

        assert mapper.batch_size == self.batch_size
        assert mapper.data_size == len(links)
        assert len(mapper.ids) == len(links)
        assert mapper.data_size == len(link_labels)
        assert mapper.head_node_types == ("movie", "user")

    def test_LinkMapper_constructor_multiple_link_types(self):
        G = example_HIN_1(self.n_feat)
        links = [
            (1, 4),
            (1, 5),
            (0, 4),
            (5, 0),
        ]  # first 3 are ('movie', 'user') links, the last is ('user', 'movie') link
        link_labels = [0] * len(links)

        with pytest.raises(AssertionError):
            HinSAGELinkMapper(
                G,
                links,
                link_labels,
                batch_size=self.batch_size,
                num_samples=self.num_samples,
                feature_size_by_type=self.n_feat,
            )

    def test_HinSAGELinkMapper_1(self):
        G = example_HIN_1(self.n_feat)
        links = [(1, 4), (1, 5), (0, 4), (0, 5)]  # ('movie', 'user') links
        data_size = len(links)
        link_labels = [0] * data_size

        mapper = HinSAGELinkMapper(
            G,
            links,
            link_labels,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            feature_size_by_type=self.n_feat,
        )

        assert len(mapper) == 2

        for batch in range(len(mapper)):
            nf, nl = mapper[batch]
            assert len(nf) == 10
            assert nf[0].shape == (self.batch_size, 1, self.n_feat["movie"])
            assert nf[1].shape == (self.batch_size, 1, self.n_feat["user"])
            assert nf[2].shape == (
                self.batch_size,
                self.num_samples[0],
                self.n_feat["user"],
            )
            assert nf[3].shape == (
                self.batch_size,
                self.num_samples[0],
                self.n_feat["user"],
            )
            assert nf[4].shape == (
                self.batch_size,
                self.num_samples[0],
                self.n_feat["movie"],
            )
            assert nf[5].shape == (
                self.batch_size,
                np.multiply(*self.num_samples),
                self.n_feat["user"],
            )
            assert nf[6].shape == (
                self.batch_size,
                np.multiply(*self.num_samples),
                self.n_feat["movie"],
            )
            assert nf[7].shape == (
                self.batch_size,
                np.multiply(*self.num_samples),
                self.n_feat["user"],
            )
            assert nf[8].shape == (
                self.batch_size,
                np.multiply(*self.num_samples),
                self.n_feat["movie"],
            )
            assert nf[9].shape == (
                self.batch_size,
                np.multiply(*self.num_samples),
                self.n_feat["user"],
            )

        with pytest.raises(IndexError):
            nf, nl = mapper[2]
