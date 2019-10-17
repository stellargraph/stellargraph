from stellargraph.core.graph import *
from stellargraph.mapper.node_mappers import RelationalFullBatchNodeGenerator

import networkx as nx
import numpy as np
import pytest
import pandas as pd
import scipy.sparse as sps


def create_graph_features():
    G = nx.MultiDiGraph()
    G.add_nodes_from(["a", "b", "c"])
    G.add_edges_from([("a", "b", "r1"), ("b", "c", "r1"), ("a", "c", "r2")])
    return G, np.array([[1, 1], [1, 0], [0, 1]])


class Test_RelationalFullBatchNodeGenerator:
    """
    Tests of FullBatchNodeGenerator class
    """

    n_feat = 2
    target_dim = 5

    gnx, features = create_graph_features()
    nodes = list(gnx.nodes)
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarDiGraph(gnx, node_features=node_features)
    N = len(G.nodes())
    edge_types = sorted(set(e[-1] for e in G.edges))
    num_relationships = len(edge_types)

    def test_generator_constructor(self):

        generator = RelationalFullBatchNodeGenerator(self.G)
        assert len(generator.As) == self.num_relationships
        assert all((A.shape == (self.N, self.N)) for A in generator.As)

        assert generator.features.shape == (self.N, self.n_feat)

    def test_generator_constructor_wrong_G_type(self):
        with pytest.raises(TypeError):
            generator = RelationalFullBatchNodeGenerator(nx.Graph(self.G))

    def generator_flow(self, G, node_ids, node_targets, sparse=False):
        generator = RelationalFullBatchNodeGenerator(G, sparse=sparse)
        n_nodes = len(G)

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
        nodes = G.nodes()
        node_features = pd.DataFrame.from_dict(
            {n: f for n, f in zip(nodes, feats)}, orient="index"
        )
        G = StellarGraph(G, node_type_name="node", node_features=node_features)

        generator = RelationalFullBatchNodeGenerator(G, name="test")
        assert generator.name == "test"
        assert np.array_equal(feats, generator.features)

    def test_fullbatch_generator_init_3(self):
        G, feats = create_graph_features()
        nodes = G.nodes()
        node_features = pd.DataFrame.from_dict(
            {n: f for n, f in zip(nodes, feats)}, orient="index"
        )
        G = StellarGraph(G, node_type_name="node", node_features=node_features)

        func = "Not callable"

        with pytest.raises(TypeError):
            generator = RelationalFullBatchNodeGenerator(G, "test", transform=func)

    def test_fullbatch_generator_transform(self):
        G, feats = create_graph_features()
        nodes = G.nodes()
        node_features = pd.DataFrame.from_dict(
            {n: f for n, f in zip(nodes, feats)}, orient="index"
        )
        G = StellarDiGraph(G, node_type_name="node", node_features=node_features)

        def func(features, A, **kwargs):
            return features, A.dot(A)

        generator = RelationalFullBatchNodeGenerator(G, "test", transform=func)
        assert generator.name == "test"

        As = []
        edge_types = sorted(set(e[-1] for e in G.edges))
        node_list = list(G.nodes)
        node_index = dict(zip(node_list, range(len(node_list))))
        for edge_type in edge_types:
            col_index = [
                node_index[n1] for n1, n2, etype in G.edges if etype == edge_type
            ]
            row_index = [
                node_index[n2] for n1, n2, etype in G.edges if etype == edge_type
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
