from stellargraph.layer.ppnp import *
from stellargraph.mapper.node_mappers import FullBatchNodeGenerator
from stellargraph.core.graph import StellarGraph
from stellargraph.core.utils import PPNP_Aadj_feats_op

import networkx as nx
import pandas as pd
import numpy as np
from tensorflow import keras
import pytest


def create_graph_features():
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c"])
    G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
    G = G.to_undirected()
    return G, np.array([[1, 1], [1, 0], [0, 1]])


def test_PPNP_edge_cases():
    G, features = create_graph_features()
    adj = nx.to_scipy_sparse_matrix(G)
    features, adj = PPNP_Aadj_feats_op(features, adj)

    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_features=node_features)

    ppnp_sparse_failed = False
    try:
        generator = FullBatchNodeGenerator(G, sparse=True, method="ppnp")
    except ValueError as e:
        ppnp_sparse_failed = True
    assert(ppnp_sparse_failed)

    generator = FullBatchNodeGenerator(G, sparse=False, method="ppnp")

    try:
        ppnpModel = PPNP([2, 2], ["relu"], generator=generator, dropout=0.5)
    except ValueError as e:
        error = e
    assert str(error) == "The number of layers should equal the number of activations"

    try:
        ppnpModel = PPNP([2], ["relu"], generator=[0, 1], dropout=0.5)
    except TypeError as e:
        error = e
    assert str(error) == "Generator should be a instance of FullBatchNodeGenerator"


def test_PPNP_apply_dense():
    G, features = create_graph_features()
    adj = nx.to_scipy_sparse_matrix(G)
    features, adj = PPNP_Aadj_feats_op(features, adj)
    adj = adj[None, :, :]

    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_features=node_features)

    generator = FullBatchNodeGenerator(G, sparse=False, method="ppnp")
    ppnpModel = PPNP([2], ["relu"], generator=generator, dropout=0.5)

    x_in, x_out = ppnpModel.node_model()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, adj])
    assert preds_1.shape == (1, 2, 2)

    # Check fit_generator method
    preds_2 = model.predict_generator(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)