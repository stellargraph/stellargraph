# -*- coding: utf-8 -*-
#
# Copyright 2017-2020 Data61, CSIRO
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


"""
Full-batch Generator Tests

"""
from stellargraph.core.graph import *
from stellargraph.core.graph_networkx import NetworkXStellarGraph
from stellargraph.mapper import (
    FullBatchGenerator,
    FullBatchLinkGenerator,
    FullBatchNodeGenerator,
)

import networkx as nx
import numpy as np
import random
import pytest
import pandas as pd
import scipy.sparse as sps


def create_graph_features():
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c"])
    G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
    G = G.to_undirected()
    return G, np.array([[1, 1], [1, 0], [0, 1]])


def example_graph_1(feature_size=None):
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_nodes_from([1, 2, 3, 4], label="default")
    G.add_edges_from(elist, label="default")

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.nodes[v]["feature"] = np.ones(feature_size)
        return StellarGraph(G, node_features="feature")

    else:
        return StellarGraph(G)


def example_graph_2(feature_size=None):
    G = nx.Graph()
    elist = [(1, 2), (1, 3), (1, 4), (3, 2), (3, 5)]
    G.add_nodes_from([1, 2, 3, 4, 5], label="default")
    G.add_edges_from(elist, label="default")

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.nodes[v]["feature"] = int(v) * np.ones(feature_size, dtype="int")
        return StellarGraph(G, node_features="feature")

    else:
        return StellarGraph(G)


def example_graph_3(feature_size=None, n_edges=20, n_nodes=6, n_isolates=1):
    G = nx.Graph()
    n_noniso = n_nodes - n_isolates
    edges = [
        (random.randint(0, n_noniso - 1), random.randint(0, n_noniso - 1))
        for _ in range(n_edges)
    ]
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges, label="default")

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.nodes[v]["feature"] = int(v) * np.ones(feature_size, dtype="int")
        return StellarGraph(G, node_features="feature")

    else:
        return StellarGraph(G)


def example_digraph_2(feature_size=None):
    G = nx.DiGraph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_edges_from(elist)

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.nodes[v]["feature"] = np.ones(feature_size)
        return StellarDiGraph(G, node_features="feature")

    else:
        return StellarDiGraph(G)


def example_hin_1(feature_size_by_type=None):
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3], label="A")
    G.add_nodes_from([4, 5, 6], label="B")
    G.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label="R")
    G.add_edges_from([(4, 5)], label="F")

    # Add example features
    if feature_size_by_type is not None:
        for v, vdata in G.nodes(data=True):
            nt = vdata["label"]
            vdata["feature"] = int(v) * np.ones(feature_size_by_type[nt], dtype="int")
        return StellarGraph(G, node_features="feature")

    else:
        return StellarGraph(G)


def example_hin_2(feature_size_by_type=None):
    nodes_type_1 = [0, 1, 2, 3]
    nodes_type_2 = [4, 5]

    # Create isolated graphs
    G = nx.Graph()
    G.add_nodes_from(nodes_type_1, label="t1")
    G.add_nodes_from(nodes_type_2, label="t2")
    G.add_edges_from([(0, 4), (1, 4), (2, 5), (3, 5)], label="e1")

    # Add example features
    if feature_size_by_type is not None:
        for v, vdata in G.nodes(data=True):
            nt = vdata["label"]
            vdata["feature"] = int(v) * np.ones(feature_size_by_type[nt], dtype="int")

        G = StellarGraph(G, node_features="feature")

    else:
        G = StellarGraph(G)

    return G, nodes_type_1, nodes_type_2


def example_hin_3(feature_size_by_type=None):
    nodes_type_1 = [0, 1, 2]
    nodes_type_2 = [4, 5, 6]

    # Create isolated graphs
    G = nx.Graph()
    G.add_nodes_from(nodes_type_1, label="t1")
    G.add_nodes_from(nodes_type_2, label="t2")
    G.add_edges_from([(0, 4), (1, 5)], label="e1")
    G.add_edges_from([(0, 2)], label="e2")

    # Node 2 has no edges of type 1
    # Node 1 has no edges of type 2
    # Node 6 has no edges

    # Add example features
    if feature_size_by_type is not None:
        for v, vdata in G.nodes(data=True):
            nt = vdata["label"]
            vdata["feature"] = (int(v) + 10) * np.ones(
                feature_size_by_type[nt], dtype="int"
            )

        G = StellarGraph(G, node_features="feature")

    else:
        G = StellarGraph(G)

    return G, nodes_type_1, nodes_type_2


class Test_FullBatchGenerator:
    """
    Tests of FullBatchGenerator class
    """

    G = example_graph_3(feature_size=4, n_nodes=6, n_isolates=1, n_edges=20)

    def test_generator_constructor(self):
        # Test constructing abstract base class
        with pytest.raises(TypeError):
            generator = FullBatchGenerator(self.G)


class Test_FullBatchNodeGenerator:
    """
    Tests of FullBatchNodeGenerator class
    """

    n_feat = 4
    target_dim = 5

    G = example_graph_3(feature_size=n_feat, n_nodes=6, n_isolates=1, n_edges=20)
    N = len(G.nodes())

    def test_generator_constructor(self):
        generator = FullBatchNodeGenerator(self.G)
        assert generator.Aadj.shape == (self.N, self.N)
        assert generator.features.shape == (self.N, self.n_feat)

    def test_generator_constructor_wrong_G_type(self):
        with pytest.raises(TypeError):
            generator = FullBatchNodeGenerator(nx.Graph())

    def test_generator_constructor_hin(self):
        feature_sizes = {"t1": 1, "t2": 1}
        Ghin, nodes_type_1, nodes_type_2 = example_hin_3(feature_sizes)
        with pytest.raises(TypeError):
            generator = FullBatchNodeGenerator(Ghin)

    def generator_flow(
        self,
        G,
        node_ids,
        node_targets,
        sparse=False,
        method="none",
        k=1,
        teleport_probability=0.1,
    ):
        generator = FullBatchNodeGenerator(
            G,
            sparse=sparse,
            method=method,
            k=k,
            teleport_probability=teleport_probability,
        )
        n_nodes = G.number_of_nodes()

        gen = generator.flow(node_ids, node_targets)
        if sparse:
            [X, tind, A_ind, A_val], y = gen[0]
            A_sparse = sps.coo_matrix(
                (A_val[0], (A_ind[0, :, 0], A_ind[0, :, 1])), shape=(n_nodes, n_nodes)
            )
            A_dense = A_sparse.toarray()

        else:
            [X, tind, A], y = gen[0]
            A_dense = A[0]

        assert np.allclose(X, gen.features)  # X should be equal to gen.features
        assert tind.shape[1] == len(node_ids)

        if node_targets is not None:
            assert np.allclose(y, node_targets)

        # Check that the diagonals are one
        if method == "self_loops":
            assert np.allclose(A_dense.diagonal(), 1)

        return A_dense, tind, y

    def test_generator_flow_notargets(self):
        node_ids = list(self.G.nodes())[:3]
        _, tind, y = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="none"
        )
        assert np.allclose(tind, range(3))
        _, tind, y = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="none"
        )
        assert np.allclose(tind, range(3))

        node_ids = list(self.G.nodes())
        _, tind, y = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="none"
        )
        assert np.allclose(tind, range(len(node_ids)))
        _, tind, y = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="none"
        )
        assert np.allclose(tind, range(len(node_ids)))

    def test_generator_flow_withtargets(self):
        node_ids = list(self.G.nodes())[:3]
        node_targets = np.ones((len(node_ids), self.target_dim)) * np.arange(3)[:, None]
        _, tind, y = self.generator_flow(self.G, node_ids, node_targets, sparse=True)
        assert np.allclose(tind, range(3))
        assert np.allclose(y, node_targets[:3])
        _, tind, y = self.generator_flow(self.G, node_ids, node_targets, sparse=False)
        assert np.allclose(tind, range(3))
        assert np.allclose(y, node_targets[:3])

        node_ids = list(self.G.nodes())[::-1]
        node_targets = (
            np.ones((len(node_ids), self.target_dim))
            * np.arange(len(node_ids))[:, None]
        )
        _, tind, y = self.generator_flow(self.G, node_ids, node_targets)
        assert np.allclose(tind, range(len(node_ids))[::-1])
        assert np.allclose(y, node_targets)

    def test_generator_flow_targets_as_list(self):
        generator = FullBatchNodeGenerator(self.G)
        node_ids = list(self.G.nodes())[:3]
        node_targets = [1] * len(node_ids)
        gen = generator.flow(node_ids, node_targets)

        inputs, y = gen[0]
        assert y.shape == (1, 3)
        assert np.sum(y) == 3

    def test_generator_flow_targets_not_iterator(self):
        generator = FullBatchNodeGenerator(self.G)
        node_ids = list(self.G.nodes())[:3]
        node_targets = 1
        with pytest.raises(TypeError):
            generator.flow(node_ids, node_targets)

    def test_fullbatch_generator_init_1(self):
        G, feats = create_graph_features()
        nodes = G.nodes()
        node_features = pd.DataFrame.from_dict(
            {n: f for n, f in zip(nodes, feats)}, orient="index"
        )
        G = StellarGraph(G, node_type_name="node", node_features=node_features)

        generator = FullBatchNodeGenerator(G, method=None)
        assert np.array_equal(feats, generator.features)

    def test_fullbatch_generator_init_3(self):
        G, feats = create_graph_features()
        nodes = G.nodes()
        node_features = pd.DataFrame.from_dict(
            {n: f for n, f in zip(nodes, feats)}, orient="index"
        )
        G = StellarGraph(G, node_type_name="node", node_features=node_features)

        func = "Not callable"

        with pytest.raises(ValueError):
            generator = FullBatchNodeGenerator(G, "test", transform=func)

    def test_fullbatch_generator_transform(self):
        G, feats = create_graph_features()
        nodes = G.nodes()
        node_features = pd.DataFrame.from_dict(
            {n: f for n, f in zip(nodes, feats)}, orient="index"
        )
        G = StellarGraph(G, node_type_name="node", node_features=node_features)

        def func(features, A, **kwargs):
            return features, A.dot(A)

        generator = FullBatchNodeGenerator(G, "test", transform=func)
        assert generator.name == "test"

        A = G.to_adjacency_matrix().toarray()
        assert np.array_equal(A.dot(A), generator.Aadj.toarray())

    def test_generator_methods(self):
        node_ids = list(self.G.nodes())
        Aadj = self.G.to_adjacency_matrix().toarray()
        Aadj_selfloops = Aadj + np.eye(*Aadj.shape) - np.diag(Aadj.diagonal())
        Dtilde = np.diag(Aadj_selfloops.sum(axis=1) ** (-0.5))
        Agcn = Dtilde.dot(Aadj_selfloops).dot(Dtilde)
        Appnp = 0.1 * np.linalg.inv(np.eye(Agcn.shape[0]) - ((1 - 0.1) * Agcn))

        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="none"
        )
        assert np.allclose(A_dense, Aadj)
        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="none"
        )
        assert np.allclose(A_dense, Aadj)

        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="self_loops"
        )
        assert np.allclose(A_dense, Aadj_selfloops)
        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="self_loops"
        )
        assert np.allclose(A_dense, Aadj_selfloops)

        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="gat"
        )
        assert np.allclose(A_dense, Aadj_selfloops)
        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="gat"
        )
        assert np.allclose(A_dense, Aadj_selfloops)

        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="gcn"
        )
        assert np.allclose(A_dense, Agcn)
        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="gcn"
        )
        assert np.allclose(A_dense, Agcn)

        # Check other pre-processing options
        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="sgc", k=2
        )
        assert np.allclose(A_dense, Agcn.dot(Agcn))
        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="sgc", k=2
        )
        assert np.allclose(A_dense, Agcn.dot(Agcn))

        A_dense, _, _ = self.generator_flow(
            self.G,
            node_ids,
            None,
            sparse=False,
            method="ppnp",
            teleport_probability=0.1,
        )
        assert np.allclose(A_dense, Appnp)

        ppnp_sparse_failed = False
        try:
            A_dense, _, _ = self.generator_flow(
                self.G,
                node_ids,
                None,
                sparse=True,
                method="ppnp",
                teleport_probability=0.1,
            )
        except ValueError as e:
            ppnp_sparse_failed = True

        assert ppnp_sparse_failed


class Test_FullBatchLinkGenerator:
    """
    Tests of FullBatchNodeGenerator class
    """

    n_feat = 4
    target_dim = 5

    G = example_graph_3(feature_size=n_feat, n_nodes=6, n_isolates=1, n_edges=20)
    N = len(G.nodes())

    def test_generator_constructor(self):
        generator = FullBatchLinkGenerator(self.G)
        assert generator.Aadj.shape == (self.N, self.N)
        assert generator.features.shape == (self.N, self.n_feat)

    def test_generator_constructor_wrong_G_type(self):
        with pytest.raises(TypeError):
            generator = FullBatchLinkGenerator(nx.Graph())

    def test_generator_constructor_hin(self):
        feature_sizes = {"t1": 1, "t2": 1}
        Ghin, nodes_type_1, nodes_type_2 = example_hin_3(feature_sizes)
        with pytest.raises(TypeError):
            generator = FullBatchLinkGenerator(Ghin)

    def generator_flow(
        self,
        G,
        link_ids,
        link_targets,
        sparse=False,
        method="none",
        k=1,
        teleport_probability=0.1,
    ):
        generator = FullBatchLinkGenerator(
            G,
            sparse=sparse,
            method=method,
            k=k,
            teleport_probability=teleport_probability,
        )
        n_nodes = G.number_of_nodes()

        gen = generator.flow(link_ids, link_targets)
        if sparse:
            [X, tind, A_ind, A_val], y = gen[0]
            A_sparse = sps.coo_matrix(
                (A_val[0], (A_ind[0, :, 0], A_ind[0, :, 1])), shape=(n_nodes, n_nodes)
            )
            A_dense = A_sparse.toarray()

        else:
            [X, tind, A], y = gen[0]
            A_dense = A[0]

        assert np.allclose(X, gen.features)  # X should be equal to gen.features
        assert isinstance(tind, np.ndarray)
        assert tind.ndim == 3
        assert tind.shape[1] == len(link_ids)
        assert tind.shape[2] == 2

        if link_targets is not None:
            assert np.allclose(y, link_targets)

        # Check that the diagonals are one
        if method == "self_loops":
            assert np.allclose(A_dense.diagonal(), 1)

        return A_dense, tind, y

    def test_generator_flow_notargets(self):
        link_ids = list(self.G.edges())[:3]

        _, tind, y = self.generator_flow(
            self.G, link_ids, None, sparse=False, method="none"
        )
        assert np.allclose(tind.reshape((3, 2)), link_ids)
        _, tind, y = self.generator_flow(
            self.G, link_ids, None, sparse=True, method="none"
        )
        assert np.allclose(tind.reshape((3, 2)), link_ids)

    def test_generator_flow_withtargets(self):
        link_ids = list(self.G.edges())[:3]
        link_targets = np.ones((len(link_ids), self.target_dim)) * np.arange(3)[:, None]
        _, tind, y = self.generator_flow(self.G, link_ids, link_targets, sparse=True)
        assert np.allclose(tind.reshape((3, 2)), link_ids)
        assert np.allclose(y, link_targets[:3])

        _, tind, y = self.generator_flow(self.G, link_ids, link_targets, sparse=False)
        assert np.allclose(tind.reshape((3, 2)), link_ids)
        assert np.allclose(y, link_targets[:3])

    def test_generator_flow_targets_as_list(self):
        generator = FullBatchLinkGenerator(self.G)
        link_ids = list(self.G.edges())[:3]
        link_targets = [1] * len(link_ids)
        gen = generator.flow(link_ids, link_targets)

        inputs, y = gen[0]
        assert y.shape == (1, 3)
        assert np.sum(y) == 3

    def test_generator_flow_targets_not_iterator(self):
        generator = FullBatchLinkGenerator(self.G)
        link_ids = list(self.G.edges())[:3]
        link_targets = 1
        with pytest.raises(TypeError):
            generator.flow(link_ids, link_targets)

    def test_fullbatch_generator_transform(self):
        def func(features, A, **kwargs):
            return features, A.dot(A)

        generator = FullBatchNodeGenerator(self.G, "test", transform=func)
        assert generator.name == "test"

        A = self.G.to_adjacency_matrix().toarray()
        assert np.array_equal(A.dot(A), generator.Aadj.toarray())

    def test_generator_methods(self):
        link_ids = list(self.G.edges())[:10]
        Aadj = self.G.to_adjacency_matrix().toarray()
        Aadj_selfloops = Aadj + np.eye(*Aadj.shape) - np.diag(Aadj.diagonal())
        Dtilde = np.diag(Aadj_selfloops.sum(axis=1) ** (-0.5))
        Agcn = Dtilde.dot(Aadj_selfloops).dot(Dtilde)
        Appnp = 0.1 * np.linalg.inv(np.eye(Agcn.shape[0]) - ((1 - 0.1) * Agcn))

        A_dense, _, _ = self.generator_flow(
            self.G, link_ids, None, sparse=True, method="none"
        )
        assert np.allclose(A_dense, Aadj)
        A_dense, _, _ = self.generator_flow(
            self.G, link_ids, None, sparse=False, method="none"
        )
        assert np.allclose(A_dense, Aadj)

        A_dense, _, _ = self.generator_flow(
            self.G, link_ids, None, sparse=True, method="self_loops"
        )
        assert np.allclose(A_dense, Aadj_selfloops)
        A_dense, _, _ = self.generator_flow(
            self.G, link_ids, None, sparse=False, method="self_loops"
        )
        assert np.allclose(A_dense, Aadj_selfloops)

        A_dense, _, _ = self.generator_flow(
            self.G, link_ids, None, sparse=True, method="gat"
        )
        assert np.allclose(A_dense, Aadj_selfloops)
        A_dense, _, _ = self.generator_flow(
            self.G, link_ids, None, sparse=False, method="gat"
        )
        assert np.allclose(A_dense, Aadj_selfloops)

        A_dense, _, _ = self.generator_flow(
            self.G, link_ids, None, sparse=True, method="gcn"
        )
        assert np.allclose(A_dense, Agcn)
        A_dense, _, _ = self.generator_flow(
            self.G, link_ids, None, sparse=False, method="gcn"
        )
        assert np.allclose(A_dense, Agcn)

        # Check other pre-processing options
        A_dense, _, _ = self.generator_flow(
            self.G, link_ids, None, sparse=True, method="sgc", k=2
        )
        assert np.allclose(A_dense, Agcn.dot(Agcn))
        A_dense, _, _ = self.generator_flow(
            self.G, link_ids, None, sparse=False, method="sgc", k=2
        )
        assert np.allclose(A_dense, Agcn.dot(Agcn))

        A_dense, _, _ = self.generator_flow(
            self.G,
            link_ids,
            None,
            sparse=False,
            method="ppnp",
            teleport_probability=0.1,
        )
        assert np.allclose(A_dense, Appnp)

        ppnp_sparse_failed = False
        try:
            A_dense, _, _ = self.generator_flow(
                self.G,
                link_ids,
                None,
                sparse=True,
                method="ppnp",
                teleport_probability=0.1,
            )
        except ValueError as e:
            ppnp_sparse_failed = True

        assert ppnp_sparse_failed
