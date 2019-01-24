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
from stellargraph.mapper.gcn_mappers import *
from stellargraph.core.graph import StellarGraph

import networkx as nx
import pandas as pd
import numpy as np
import keras


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
    assert conf["support"] == 1
    assert conf["activation"] == "linear"
    assert conf["use_bias"] == True
    assert conf["kernel_initializer"]["class_name"] == "VarianceScaling"
    assert conf["kernel_initializer"]["config"]["distribution"] == "uniform"
    assert conf["bias_initializer"]["class_name"] == "Zeros"
    assert conf["kernel_regularizer"] == None
    assert conf["bias_regularizer"] == None
    assert conf["activity_regularizer"] == None
    assert conf["kernel_constraint"] == None
    assert conf["bias_constraint"] == None


def test_GraphConvolution_init():
    gcn_layer = GraphConvolution(units=16, support=2, activation="relu")

    assert gcn_layer.units == 16
    assert gcn_layer.support == 2
    assert gcn_layer.use_bias == True
    assert gcn_layer.get_config()["activation"] == "relu"


def test_GraphConvolution_apply():
    G, features = create_graph_features()

    x_in = Input(shape=(features.shape[1],))
    A = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
    out = GraphConvolution(2, 1)([x_in] + A)

    adj = nx.adjacency_matrix(G)

    model = keras.Model(inputs=[x_in] + A, outputs=out)
    preds = model.predict([features, adj], batch_size=adj.shape[0])
    assert preds.shape == (3, 2)


def test_GCN_init():
    G, features = create_graph_features()
    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_type_name="node", node_features=node_features)

    generator = FullBatchNodeGenerator(G)
    gcnModel = GCN([2], ["relu"], generator=generator, dropout=0.5)

    assert gcnModel.layer_sizes == [2]
    assert gcnModel.activations == ["relu"]
    assert gcnModel.dropout == 0.5


def test_GCN_apply():
    G, features = create_graph_features()
    adj = nx.adjacency_matrix(G)

    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarGraph(G, node_type_name="node", node_features=node_features)

    generator = FullBatchNodeGenerator(G)
    gcnModel = GCN([2], ["relu"], generator=generator, dropout=0.5)

    x_in, x_out = gcnModel.node_model()
    model = keras.Model(inputs=x_in, outputs=x_out)
    preds = model.predict([features, adj], batch_size=adj.shape[0])

    assert preds.shape == (3, 2)
