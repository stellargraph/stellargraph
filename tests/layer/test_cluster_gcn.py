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
Cluster-GCN tests

"""
from tensorflow.keras import backend as K
from stellargraph.layer.cluster_gcn import *
from stellargraph.mapper.node_mappers import ClusterNodeGenerator
from stellargraph.core.graph import StellarGraph

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


def test_ClusterGraphConvolution_config():
    cluster_gcn_layer = ClusterGraphConvolution(units=16)
    conf = cluster_gcn_layer.get_config()

    assert conf["units"] == 16
    assert conf["activation"] == "linear"
    assert conf["use_bias"] == True
    assert conf["kernel_initializer"]["class_name"] == "GlorotUniform"
    assert conf["bias_initializer"]["class_name"] == "Zeros"
    assert conf["kernel_regularizer"] == None
    assert conf["bias_regularizer"] == None
    assert conf["kernel_constraint"] == None
    assert conf["bias_constraint"] == None


def test_ClusterGraphConvolution_init():
    cluster_gcn_layer = ClusterGraphConvolution(units=16, activation="relu")

    assert cluster_gcn_layer.units == 16
    assert cluster_gcn_layer.use_bias == True
    assert cluster_gcn_layer.get_config()["activation"] == "relu"


def test_GraphConvolution_dense():
    G, features = create_graph_features()

    # We need to specify the batch shape as one for the ClusterGraphConvolutional logic to work
    x_t = Input(batch_shape=(1,) + features.shape, name="X")
    A_t = Input(batch_shape=(1, 3, 3), name="A")
    output_indices_t = Input(batch_shape=(1, None), dtype="int32", name="outind")

    # Note we add a batch dimension of 1 to model inputs
    adj = nx.to_numpy_array(G)[None, :, :]
    out_indices = np.array([[0, 1]], dtype="int32")
    x = features[None, :, :]

    # Remove batch dimension
    A_mat = Lambda(lambda A: K.squeeze(A, 0))(A_t)

    # Test with final_layer=False
    out = ClusterGraphConvolution(2, final_layer=False)([x_t, output_indices_t, A_mat])
    model = keras.Model(inputs=[x_t, A_t, output_indices_t], outputs=out)
    preds = model.predict([x, adj, out_indices], batch_size=1)
    assert preds.shape == (1, 3, 2)

    # Now try with final_layer=True
    out = ClusterGraphConvolution(2, final_layer=True)([x_t, output_indices_t, A_mat])
    # The final layer removes the batch dimension and causes the call to predict to fail.
    # We are going to manually added the batch dimension before calling predict.
    out = K.expand_dims(out, 0)
    model = keras.Model(inputs=[x_t, A_t, output_indices_t], outputs=out)
    print(f"x_t: {x_t.shape} A_t: {A_t.shape} output_indices_t: {output_indices_t.shape}")
    preds = model.predict([x, adj, out_indices], batch_size=1)
    assert preds.shape == (1, 2, 2)

    # Check for errors with batch size != 1
    # We need to specify the batch shape as one for the ClusterGraphConvolutional logic to work
    x_t = Input(batch_shape=(2,) + features.shape)
    output_indices_t = Input(batch_shape=(2, None), dtype="int32")
    with pytest.raises(ValueError):
        out = ClusterGraphConvolution(2)([x_t, A_t, output_indices_t])
