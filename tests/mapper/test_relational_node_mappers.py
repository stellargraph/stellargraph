# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
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

from stellargraph.core.graph import *
from stellargraph.mapper.full_batch_generators import RelationalFullBatchNodeGenerator

import networkx as nx
import numpy as np
import pytest
import pandas as pd
import scipy.sparse as sps
from ..test_utils.graphs import (
    relational_create_graph_features as create_graph_features,
    example_hin_1,
)


class Test_RelationalFullBatchNodeGenerator:
    """
    Tests of FullBatchNodeGenerator class
    """

    n_feat = 2
    target_dim = 5

    G, features = create_graph_features()
    N = len(G.nodes())
    edge_types = sorted(set(e[-1] for e in G.edges(include_edge_type=True)))
    num_relationships = len(edge_types)

    def test_generator_constructor(self):

        generator = RelationalFullBatchNodeGenerator(self.G)
        assert len(generator.As) == self.num_relationships
        assert all((A.shape == (self.N, self.N)) for A in generator.As)

        assert generator.features.shape == (self.N, self.n_feat)

    def test_generator_constructor_wrong_G_type(self):
        with pytest.raises(TypeError):
            generator = RelationalFullBatchNodeGenerator(self.G.to_networkx())

    def generator_flow(self, G, node_ids, node_targets, sparse=False):
        generator = RelationalFullBatchNodeGenerator(G, sparse=sparse)
        n_nodes = len(G.nodes())

        gen = generator.flow(node_ids, node_targets)
        if sparse:
            [X, tind, *As], y = gen[0]
            As_indices = As[: self.num_relationships]
            As_values = As[self.num_relationships :]

            As_sparse = [
                sps.coo_matrix(
                    (A_val[0], (A_ind[0, :, 0], A_ind[0, :, 1])),
                    shape=(n_nodes, n_nodes),
                )
                for A_ind, A_val in zip(As_indices, As_values)
            ]
            As_dense = [A.toarray() for A in As_sparse]

        else:
            [X, tind, *As], y = gen[0]
            As_dense = As

        assert np.allclose(X, gen.features)  # X should be equal to gen.features
        assert tind.shape[1] == len(node_ids)

        if node_targets is not None:
            assert np.allclose(y, node_targets)

        return As_dense, tind, y

    def test_generator_flow_notargets(self):
        node_ids = list(self.G.nodes())[:3]
        _, tind, y = self.generator_flow(self.G, node_ids, None, sparse=False)
        assert np.allclose(tind, range(3))

        _, tind, y = self.generator_flow(self.G, node_ids, None, sparse=True)
        assert np.allclose(tind, range(3))

        node_ids = list(self.G.nodes())
        _, tind, y = self.generator_flow(self.G, node_ids, None, sparse=False)
        assert np.allclose(tind, range(len(node_ids)))

        _, tind, y = self.generator_flow(self.G, node_ids, None, sparse=True)
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
        generator = RelationalFullBatchNodeGenerator(self.G)
        node_ids = list(self.G.nodes())[:3]
        node_targets = [1] * len(node_ids)
        gen = generator.flow(node_ids, node_targets)

        inputs, y = gen[0]
        assert y.shape == (1, 3)
        assert np.sum(y) == 3

    def test_generator_flow_targets_not_iterator(self):
        generator = RelationalFullBatchNodeGenerator(self.G)
        node_ids = list(self.G.nodes())[:3]
        node_targets = 1
        with pytest.raises(TypeError):
            generator.flow(node_ids, node_targets)

    def test_fullbatch_generator_init_1(self):
        G, feats = create_graph_features()

        generator = RelationalFullBatchNodeGenerator(G, name="test")
        assert generator.name == "test"
        assert np.array_equal(feats, generator.features)

    def test_fullbatch_generator_init_3(self):
        G, _ = create_graph_features()

        func = "Not callable"

        with pytest.raises(TypeError):
            generator = RelationalFullBatchNodeGenerator(G, "test", transform=func)

    def test_fullbatch_generator_init_heterogeneous_nodes(self):
        G = example_hin_1(feature_sizes={})
        with pytest.raises(
            ValueError, match="G: expected one node type, found 'A', 'B'"
        ):
            RelationalFullBatchNodeGenerator(G)

    def test_fullbatch_generator_transform(self):
        G, _ = create_graph_features(is_directed=True)

        def func(features, A, **kwargs):
            return features, A.dot(A)

        generator = RelationalFullBatchNodeGenerator(G, "test", transform=func)
        assert generator.name == "test"

        As = []
        edge_types = sorted(set(e[-1] for e in G.edges(include_edge_type=True)))
        node_list = list(G.nodes())
        node_index = dict(zip(node_list, range(len(node_list))))
        for edge_type in edge_types:
            col_index = [
                node_index[n1]
                for n1, n2, etype in G.edges(include_edge_type=True)
                if etype == edge_type
            ]
            row_index = [
                node_index[n2]
                for n1, n2, etype in G.edges(include_edge_type=True)
                if etype == edge_type
            ]
            data = np.ones(len(col_index), np.float64)

            A = sps.coo_matrix(
                (data, (row_index, col_index)), shape=(len(node_list), len(node_list))
            )

            As.append(A)

        assert all(
            np.array_equal(A_1.dot(A_1).todense(), A_2.todense())
            for A_1, A_2 in zip(As, generator.As)
        )

    def test_weighted(self):
        G, _ = create_graph_features(edge_weights=True)
        generator = RelationalFullBatchNodeGenerator(
            G, weighted=True, transform=lambda f, A: (f, A)
        )
        np.testing.assert_array_equal(
            generator.As[0].todense(), [[0, 2.0, 0], [2.0, 0, 0.5], [0, 0.5, 0]]
        )
        np.testing.assert_array_equal(
            generator.As[1].todense(), [[0, 0.0, 1.0], [0.0, 0, 0.0], [1.0, 0.0, 0]]
        )
