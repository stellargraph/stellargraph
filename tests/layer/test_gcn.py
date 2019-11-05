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
GCN tests

"""

from stellargraph.layer.gcn import *
from stellargraph.mapper import FullBatchNodeGenerator
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


def test_GraphConvolution_config():
    gcn_layer = GraphConvolution(units=16)
    conf = gcn_layer.get_config()

    assert conf["units"] == 16
    assert conf["activation"] == "linear"
    assert conf["use_bias"] == True
    assert conf["kernel_initializer"]["class_name"] == "GlorotUniform"
    assert conf["bias_initializer"]["class_name"] == "Zeros"
    assert conf["kernel_regularizer"] == None
    assert conf["bias_regularizer"] == None
    assert conf["kernel_constraint"] == None
    assert conf["bias_constraint"] == None


def test_GraphConvolution_init():
    gcn_layer = GraphConvolution(units=16, activation="relu")

    assert gcn_layer.units == 16
    assert gcn_layer.use_bias == True
    assert gcn_layer.get_config()["activation"] == "relu"


def test_GraphConvolution_dense():
    G, features = create_graph_features()

    # We need to specify the batch shape as one for the GraphConvolutional logic to work
    x_t = Input(batch_shape=(1,) + features.shape, name="X")
    A_t = Input(batch_shape=(1, 3, 3), name="A")
    output_indices_t = Input(batch_shape=(1, None), dtype="int32", name="outind")

    # Note we add a batch dimension of 1 to model inputs
    adj = nx.to_numpy_array(G)[None, :, :]
    out_indices = np.array([[0, 1]], dtype="int32")
    x = features[None, :, :]

    # For dense matrix, remove batch dimension
    A_mat = Lambda(lambda A: K.squeeze(A, 0))(A_t)

    # Test with final_layer=False
    out = GraphConvolution(2, final_layer=False)([x_t, output_indices_t, A_mat])
    model = keras.Model(inputs=[x_t, A_t, output_indices_t], outputs=out)
    preds = model.predict([x, adj, out_indices], batch_size=1)
    assert preds.shape == (1, 3, 2)

    # Now try with final_layer=True
    out = GraphConvolution(2, final_layer=True)([x_t, output_indices_t, A_mat])
    model = keras.Model(inputs=[x_t, A_t, output_indices_t], outputs=out)
    preds = model.predict([x, adj, out_indices], batch_size=1)
    assert preds.shape == (1, 2, 2)

    # Check for errors with batch size != 1
    # We need to specify the batch shape as one for the GraphConvolutional logic to work
    x_t = Input(batch_shape=(2,) + features.shape)
    output_indices_t = Input(batch_shape=(2, None), dtype="int32")
    with pytest.raises(ValueError):
        out = GraphConvolution(2)([x_t, A_t, output_indices_t])


def test_GraphConvolution_sparse():
    G, features = create_graph_features()
    n_nodes = features.shape[0]

    # We need to specify the batch shape as one for the GraphConvolutional logic to work
    x_t = Input(batch_shape=(1,) + features.shape)
    A_ind = Input(batch_shape=(1, None, 2), dtype="int64")
    A_val = Input(batch_shape=(1, None), dtype="float32")
    output_indices_t = Input(batch_shape=(1, None), dtype="int32")

    # Test with final_layer=False
    A_mat = SqueezedSparseConversion(shape=(n_nodes, n_nodes), dtype=A_val.dtype)(
        [A_ind, A_val]
    )
    out = GraphConvolution(2, final_layer=False)([x_t, output_indices_t, A_mat])

    # Note we add a batch dimension of 1 to model inputs
    adj = nx.to_scipy_sparse_matrix(G, format="coo")
    A_indices = np.expand_dims(np.hstack((adj.row[:, None], adj.col[:, None])), 0)
    A_values = np.expand_dims(adj.data, 0)
    out_indices = np.array([[0, 1]], dtype="int32")
    x = features[None, :, :]

    model = keras.Model(inputs=[x_t, output_indices_t, A_ind, A_val], outputs=out)
    preds = model.predict([x, out_indices, A_indices, A_values], batch_size=1)
    assert preds.shape == (1, 3, 2)

    # Now try with final_layer=True
    out = GraphConvolution(2, final_layer=True)([x_t, output_indices_t, A_mat])
    model = keras.Model(inputs=[x_t, output_indices_t, A_ind, A_val], outputs=out)
    preds = model.predict([x, out_indices, A_indices, A_values], batch_size=1)
    assert preds.shape == (1, 2, 2)


def test_GCN_init():
    G, features = create_graph_features()
    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_type_name="node", node_features=node_features)

    generator = FullBatchNodeGenerator(G)
    gcnModel = GCN([2], generator, activations=["relu"], dropout=0.5)

    assert gcnModel.layer_sizes == [2]
    assert gcnModel.activations == ["relu"]
    assert gcnModel.dropout == 0.5


def test_GCN_apply_dense():
    G, features = create_graph_features()
    adj = nx.to_numpy_array(G)[None, :, :]
    n_nodes = features.shape[0]

    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_features=node_features)

    generator = FullBatchNodeGenerator(G, sparse=False, method="none")
    gcnModel = GCN([2], generator, activations=["relu"], dropout=0.5)

    x_in, x_out = gcnModel.node_model()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, adj])
    assert preds_1.shape == (1, 2, 2)

    # Check fit_generator method
    preds_2 = model.predict_generator(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_GCN_apply_sparse():

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
    gcnModel = GCN(
        layer_sizes=[2], activations=["relu"], generator=generator, dropout=0.5
    )

    x_in, x_out = gcnModel.node_model()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, A_indices, A_values])
    assert preds_1.shape == (1, 2, 2)

    # Check fit_generator method
    preds_2 = model.predict_generator(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_GCN_activations():
    G, features = create_graph_features()
    adj = nx.to_numpy_array(G)[None, :, :]
    n_nodes = features.shape[0]

    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_features=node_features)

    generator = FullBatchNodeGenerator(G, sparse=False, method="none")

    gcn = GCN([2], generator)
    assert gcn.activations == ["relu"]

    gcn = GCN([2, 2], generator)
    assert gcn.activations == ["relu", "relu"]

    gcn = GCN([2], generator, activations=["linear"])
    assert gcn.activations == ["linear"]

    with pytest.raises(ValueError):
        # More regularisers than layers
        gcn = GCN([2], generator, activations=["relu", "linear"])

    with pytest.raises(ValueError):
        # Fewer regularisers than layers
        gcn = GCN([2, 2], generator, activations=["relu"])

    with pytest.raises(ValueError):
        # Unknown regularisers
        gcn = GCN([2], generator, activations=["bleach"])


def test_GCN_regularisers():
    G, features = create_graph_features()
    adj = nx.to_numpy_array(G)[None, :, :]
    n_nodes = features.shape[0]

    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_features=node_features)

    generator = FullBatchNodeGenerator(G, sparse=False, method="none")

    gcn = GCN([2], generator)

    gcn = GCN([2], generator, kernel_initializer="ones")

    gcn = GCN([2], generator, kernel_initializer=initializers.ones())

    with pytest.raises(ValueError):
        gcn = GCN([2], generator, kernel_initializer="fred")

    gcn = GCN([2], generator, bias_initializer="zeros")

    gcn = GCN([2], generator, bias_initializer=initializers.zeros())

    with pytest.raises(ValueError):
        gcn = GCN([2], generator, bias_initializer="barney")
