from stellargraph.layer.appnp import *
from stellargraph.mapper.node_mappers import FullBatchNodeGenerator
from stellargraph.core.graph import StellarGraph
from stellargraph.core.utils import GCN_Aadj_feats_op

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


def test_APPNP_edge_cases():
    G, features = create_graph_features()
    adj = nx.to_scipy_sparse_matrix(G)
    features, adj = GCN_Aadj_feats_op(features, adj)
    adj = adj.todense()[None, :, :]
    n_nodes = features.shape[0]

    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_features=node_features)

    generator = FullBatchNodeGenerator(G, sparse=False, method="gcn")

    try:
        appnpModel = APPNP([2, 2], ["relu"], generator=generator, dropout=0.5)
    except ValueError as e:
        error = e
    assert str(error) == "The number of layers should equal the number of activations"

    try:
        appnpModel = APPNP([2], ["relu"], generator=[0, 1], dropout=0.5)
    except TypeError as e:
        error = e
    assert str(error) == "Generator should be a instance of FullBatchNodeGenerator"

    try:
        appnpModel = APPNP([2], ["relu"], generator=generator, dropout=0., approx_iter=-1)
    except ValueError as e:
        error = e
    assert str(error) == "approx_iter should be a positive integer"

    try:
        appnpModel = APPNP([2], ["relu"], generator=generator, dropout=0., approx_iter=1.2)
    except ValueError as e:
        error = e
    assert str(error) == "approx_iter should be a positive integer"

    try:
        appnpModel = APPNP([2], ["relu"], generator=generator, dropout=0., transport_probability=1.2)
    except ValueError as e:
        error = e
    assert str(error) == "transport_probability should be between 0 and 1 (inclusive)"

    try:
        appnpModel = APPNP([2], ["relu"], generator=generator, dropout=0., transport_probability=1.2)
    except ValueError as e:
        error = e
    assert str(error) == "transport_probability should be between 0 and 1 (inclusive)"


def test_APPNP_apply_dense():
    G, features = create_graph_features()
    adj = nx.to_scipy_sparse_matrix(G)
    features, adj = GCN_Aadj_feats_op(features, adj)
    adj = adj.todense()[None, :, :]
    n_nodes = features.shape[0]

    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_features=node_features)

    generator = FullBatchNodeGenerator(G, sparse=False, method="gcn")
    appnpModel = APPNP([2], ["relu"], generator=generator, dropout=0.5)

    x_in, x_out = appnpModel.node_model()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, adj])
    assert preds_1.shape == (1, 2, 2)

    # Check fit_generator method
    preds_2 = model.predict_generator(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_APPNP_apply_sparse():

    G, features = create_graph_features()
    adj = nx.to_scipy_sparse_matrix(G)
    features, adj = GCN_Aadj_feats_op(features, adj)
    adj = adj.tocoo()
    A_indices = np.expand_dims(np.hstack((adj.row[:, None], adj.col[:, None])), 0)
    A_values = np.expand_dims(adj.data, 0)

    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_features=node_features)

    generator = FullBatchNodeGenerator(G, sparse=True, method="gcn")
    appnpnModel = APPNP([2], ["relu"], generator=generator, dropout=0.5)

    x_in, x_out = appnpnModel.node_model()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, A_indices, A_values])
    assert preds_1.shape == (1, 2, 2)

    # Check fit_generator method
    preds_2 = model.predict_generator(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_APPNP_apply_propagate_model_dense():
    G, features = create_graph_features()
    adj = nx.to_scipy_sparse_matrix(G)
    features, adj = GCN_Aadj_feats_op(features, adj)
    adj = adj.todense()[None, :, :]
    n_nodes = features.shape[0]

    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_features=node_features)

    generator = FullBatchNodeGenerator(G, sparse=False, method="gcn")
    appnpnModel = APPNP([2], ["relu"], generator=generator, dropout=0.5)

    fully_connected_model = keras.Sequential()
    fully_connected_model.add(Dense(2))

    x_in, x_out = appnpnModel.propagate_model(fully_connected_model)
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, adj])
    assert preds_1.shape == (1, 2, 2)

    # Check fit_generator method
    preds_2 = model.predict_generator(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_APPNP_apply_propagate_model_sparse():

    G, features = create_graph_features()
    adj = nx.to_scipy_sparse_matrix(G)
    features, adj = GCN_Aadj_feats_op(features, adj)
    adj = adj.tocoo()
    A_indices = np.expand_dims(np.hstack((adj.row[:, None], adj.col[:, None])), 0)
    A_values = np.expand_dims(adj.data, 0)

    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_features=node_features)

    generator = FullBatchNodeGenerator(G, sparse=True, method="gcn")
    appnpnModel = APPNP([2], ["relu"], generator=generator, dropout=0.5)

    fully_connected_model = keras.Sequential()
    fully_connected_model.add(Dense(2))

    x_in, x_out = appnpnModel.propagate_model(fully_connected_model)
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, A_indices, A_values])
    assert preds_1.shape == (1, 2, 2)

    # Check fit_generator method
    preds_2 = model.predict_generator(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)
