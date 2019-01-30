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
GAT tests
"""
from stellargraph.core.graph import StellarGraph
from stellargraph.mapper.node_mappers import FullBatchNodeGenerator
from stellargraph.layer.graph_attention import *

import keras
from keras.layers import Input
import numpy as np
import networkx as nx
import pytest


def example_graph_1(feature_size=None):
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_nodes_from([1, 2, 3, 4], label="default")
    G.add_edges_from(elist, label="default")

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = np.ones(feature_size)
        return StellarGraph(G, node_features="feature")

    else:
        return StellarGraph(G)


class Test_GraphAttention_layer:
    N = 10
    F_in = 5
    F_out = 2
    attn_heads = 8
    activation = "relu"
    """
    Tests of GraphAttention layer
    """

    def test_constructor(self):
        # attn_heads_reduction = "concat":
        layer = GraphAttention(
            F_out=self.F_out,
            attn_heads=self.attn_heads,
            attn_heads_reduction="concat",
            activation=self.activation,
        )
        assert layer.F_out == self.F_out
        assert layer.attn_heads == self.attn_heads
        assert layer.output_dim == self.F_out * self.attn_heads
        assert layer.activation == keras.activations.get(self.activation)

        # attn_heads_reduction = "average":
        layer = GraphAttention(
            F_out=self.F_out,
            attn_heads=self.attn_heads,
            attn_heads_reduction="average",
            activation=self.activation,
        )
        assert layer.output_dim == self.F_out

        # attn_heads_reduction = "ave":
        with pytest.raises(ValueError):
            GraphAttention(
                F_out=self.F_out,
                attn_heads=self.attn_heads,
                attn_heads_reduction="ave",
                activation=self.activation,
            )

    def test_apply_concat(self):
        gat = GraphAttention(
            F_out=self.F_out,
            attn_heads=self.attn_heads,
            attn_heads_reduction="concat",
            activation=self.activation,
            kernel_initializer="ones",
        )
        x_inp = [Input(shape=(self.F_in,)), Input(shape=(self.N,))]
        x_out = gat(x_inp)

        model = keras.Model(inputs=x_inp, outputs=x_out)
        assert model.output_shape[-1] == self.F_out * self.attn_heads

        X = np.ones((self.N, self.F_in))  # features
        A = np.eye(self.N)  # adjacency matrix with self-loops only
        expected = np.ones((self.N, self.F_out * self.attn_heads)) * self.F_in
        actual = model.predict([X, A])
        assert expected == pytest.approx(actual)

    def test_apply_average(self):
        gat = GraphAttention(
            F_out=self.F_out,
            attn_heads=self.attn_heads,
            attn_heads_reduction="average",
            activation=self.activation,
            kernel_initializer="ones",
        )
        x_inp = [Input(shape=(self.F_in,)), Input(shape=(self.N,))]
        x_out = gat(x_inp)

        model = keras.Model(inputs=x_inp, outputs=x_out)
        assert model.output_shape[-1] == self.F_out

        X = np.ones((self.N, self.F_in))  # features
        A = np.eye(self.N)  # adjacency matrix with self-loops only
        expected = np.ones((self.N, self.F_out)) * self.F_in
        actual = model.predict([X, A])
        assert expected == pytest.approx(actual)

    def test_layer_config(self):
        layer = GraphAttention(
            F_out=self.F_out,
            attn_heads=self.attn_heads,
            attn_heads_reduction="concat",
            activation=self.activation,
        )
        conf = layer.get_config()

        assert conf["F_out"] == self.F_out
        assert conf["attn_heads"] == self.attn_heads
        assert conf["attn_heads_reduction"] == "concat"
        assert conf["activation"] == self.activation
        assert conf["output_dim"] == self.F_out*self.attn_heads
        assert conf["use_bias"] == True
        assert conf["kernel_initializer"]["class_name"] == "VarianceScaling"
        assert conf["kernel_initializer"]["config"]["distribution"] == "uniform"
        assert conf["bias_initializer"]["class_name"] == "Zeros"
        assert conf["kernel_regularizer"] == None
        assert conf["bias_regularizer"] == None
        assert conf["kernel_constraint"] == None
        assert conf["bias_constraint"] == None
