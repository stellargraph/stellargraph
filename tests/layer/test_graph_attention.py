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
    F = 5
    F_ = 2
    attn_heads = 8
    activation = "relu"
    """
    Tests of GraphAttention layer
    """
    def test_constructor(self):
        # attn_heads_reduction = "concat":
        layer = GraphAttention(F_=self.F_, attn_heads=self.attn_heads, attn_heads_reduction="concat",
                               activation=self.activation)
        assert layer.F_ == self.F_
        assert layer.attn_heads == self.attn_heads
        assert layer.output_dim == self.F_ *self.attn_heads
        assert layer.activation == keras.activations.get(self.activation)

        # attn_heads_reduction = "average":
        layer = GraphAttention(F_=self.F_, attn_heads=self.attn_heads, attn_heads_reduction="average",
                               activation=self.activation)
        assert layer.output_dim == self.F_

        # attn_heads_reduction = "ave":
        with pytest.raises(ValueError):
            GraphAttention(F_=self.F_, attn_heads=self.attn_heads, attn_heads_reduction="ave",
                           activation=self.activation)

    def test_apply(self):
        gat = GraphAttention(F_=self.F_, attn_heads=self.attn_heads, attn_heads_reduction="concat",
                             activation=self.activation, kernel_initializer="ones")
        x_inp = [Input(shape=(self.F,)), Input(shape=(self.N,))]
        x_out = gat(x_inp)

        model = keras.Model(inputs=x_inp, outputs=x_out)
        assert model.output_shape[-1] == self.F_ *self.attn_heads

        X = np.ones((self.N, self.F))  # features
        A = np.eye(self.N)   # adjacency matrix with self-loops only
        expected = np.ones((self.N, self.F_ *self.attn_heads) ) *self.F
        actual = model.predict([X, A])
        assert expected == pytest.approx(actual)