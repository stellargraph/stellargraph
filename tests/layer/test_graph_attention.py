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
from stellargraph.mapper.node_mappers import (
    FullBatchNodeGenerator,
    GraphSAGENodeGenerator,
)
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

    def test_apply_average_with_neighbours(self):
        gat = GraphAttention(
            F_out=self.F_out,
            attn_heads=1,
            attn_heads_reduction="average",
            activation=self.activation,
            kernel_initializer="ones",
            attn_kernel_initializer="zeros",
        )
        x_inp = [Input(shape=(self.F_in,)), Input(shape=(self.N,))]
        x_out = gat(x_inp)

        model = keras.Model(inputs=x_inp, outputs=x_out)
        assert model.output_shape[-1] == self.F_out

        X = np.zeros((self.N, self.F_in))  # features
        for i in range(X.shape[0]):
            X[i, :] += i
        A = np.eye(self.N)  # adjacency matrix with self-loops only
        A[0, 1] = A[1, 0] = 1.0  # add undirected link between nodes 0 and 1

        expected = (X * self.F_in)[:, : self.F_out]
        expected[:2,] = np.ones((2, self.F_out)) * (self.F_in / 2)
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
        assert conf["use_bias"] == True
        assert conf["kernel_initializer"]["class_name"] == "VarianceScaling"
        assert conf["kernel_initializer"]["config"]["distribution"] == "uniform"
        assert conf["bias_initializer"]["class_name"] == "Zeros"
        assert conf["kernel_regularizer"] == None
        assert conf["bias_regularizer"] == None
        assert conf["kernel_constraint"] == None
        assert conf["bias_constraint"] == None


class Test_GAT:
    N = 10
    F_in = 5
    F_out = 2
    attn_heads = 8
    layer_sizes = [4, 16]
    activations = ["relu", "linear"]
    """
    Tests of GAT class
    """

    def test_constructor(self):
        G = example_graph_1(feature_size=self.F_in)
        gen = FullBatchNodeGenerator(G)
        # test error if no activations are passed:
        with pytest.raises(TypeError):
            gat = GAT(layer_sizes=self.layer_sizes, generator=gen, bias=True)

        # test error where layer_sizes is not a list:
        with pytest.raises(TypeError):
            gat = GAT(
                layer_sizes=10,
                activations=self.activations,
                attn_heads=self.attn_heads,
                generator=gen,
                bias=True,
            )

        # test error where layer_sizes values are not valid
        with pytest.raises(ValueError):
            gat = GAT(
                layer_sizes=[4, 0],
                activations=self.activations,
                attn_heads=self.attn_heads,
                generator=gen,
                bias=True,
            )

        # test for incorrect length of att_heads list:
        with pytest.raises(ValueError):
            gat = GAT(
                layer_sizes=self.layer_sizes,
                activations=self.activations,
                attn_heads=[8, 8, 1],
                generator=gen,
                bias=True,
            )

        # test for invalid values in att_heads list:
        with pytest.raises(ValueError):
            gat = GAT(
                layer_sizes=self.layer_sizes,
                activations=self.activations,
                attn_heads=[8, 0],
                generator=gen,
                bias=True,
            )

        # test for invalid type of att_heads argument:
        with pytest.raises(TypeError):
            gat = GAT(
                layer_sizes=self.layer_sizes,
                activations=self.activations,
                attn_heads=8.0,
                generator=gen,
                bias=True,
            )

        # test error where activations is not a list:
        with pytest.raises(TypeError):
            gat = GAT(
                layer_sizes=self.layer_sizes,
                activations="relu",
                generator=gen,
                bias=True,
            )

        # test attn_heads_reduction errors:
        with pytest.raises(TypeError):
            gat = GAT(
                layer_sizes=self.layer_sizes,
                activations=self.activations,
                attn_heads=self.attn_heads,
                attn_heads_reduction="concat",
                generator=gen,
                bias=True,
            )
        with pytest.raises(ValueError):
            gat = GAT(
                layer_sizes=self.layer_sizes,
                activations=self.activations,
                attn_heads=self.attn_heads,
                attn_heads_reduction=["concat", "concat", "average"],
                generator=gen,
                bias=True,
            )
        with pytest.raises(ValueError):
            gat = GAT(
                layer_sizes=self.layer_sizes,
                activations=self.activations,
                attn_heads=self.attn_heads,
                attn_heads_reduction=["concat", "sum"],
                generator=gen,
                bias=True,
            )

        # test error where len(activations) is not equal to len(layer_sizes):
        with pytest.raises(ValueError):
            gat = GAT(
                layer_sizes=self.layer_sizes,
                activations=["relu"],
                generator=gen,
                bias=True,
            )

        # Default attention heads reductions:
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            generator=gen,
            bias=True,
        )

        assert gat.activations == self.activations
        assert gat.attn_heads_reduction == ["concat", "average"]
        assert gat.generator == gen

        # User-specified attention heads reductions:
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            attn_heads_reduction=["concat", "concat"],
            generator=gen,
            bias=True,
        )

        assert gat.attn_heads_reduction == ["concat", "concat"]

    def test_gat_node_model_constructor(self):
        G = example_graph_1(feature_size=self.F_in)
        gen = FullBatchNodeGenerator(G)
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            generator=gen,
            bias=True,
        )

        assert len(gat.node_model()) == 2
        x_in, x_out = gat.node_model()
        assert len(x_in) == 2
        assert int(x_in[0].shape[-1]) == self.F_in
        assert x_in[1]._keras_shape == (None, G.number_of_nodes())
        assert int(x_out.shape[-1]) == self.layer_sizes[-1]

    def test_gat_node_model_constructor_no_generator(self):
        G = example_graph_1(feature_size=self.F_in)
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            bias=True,
        )

        with pytest.raises(RuntimeError):
            x_in, x_out = gat.node_model()

        x_in, x_out = gat.node_model(num_nodes=1000, feature_size=self.F_in)
        assert len(x_in) == 2
        assert int(x_in[0].shape[-1]) == self.F_in
        assert x_in[1]._keras_shape == (None, 1000)
        assert int(x_out.shape[-1]) == self.layer_sizes[-1]

    def test_gat_node_model_constructor_wrong_generator(self):
        G = example_graph_1(feature_size=self.F_in)
        gen = GraphSAGENodeGenerator(G, self.N, [5, 10])

        # test error where generator is of the wrong type for GAT:
        with pytest.raises(ValueError):
            gat = GAT(
                layer_sizes=self.layer_sizes,
                activations=self.activations,
                attn_heads=self.attn_heads,
                bias=True,
                generator=gen,
            )

    def test_gat_node_model_l2norm(self):
        G = example_graph_1(feature_size=self.F_in)
        gen = FullBatchNodeGenerator(G)
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            generator=gen,
            bias=True,
            normalize="l2",
        )

        gat._layers[1].kernel_initializer = keras.initializers.get("ones")
        gat._layers[1].attn_kernel_initializer = keras.initializers.get("ones")
        gat._layers[3].kernel_initializer = keras.initializers.get("ones")
        gat._layers[3].attn_kernel_initializer = keras.initializers.get("ones")

        assert len(gat.node_model()) == 2
        x_in, x_out = gat.node_model()
        assert len(x_in) == 2
        assert int(x_in[0].shape[-1]) == self.F_in
        assert x_in[1]._keras_shape == (None, G.number_of_nodes())
        assert int(x_out.shape[-1]) == self.layer_sizes[-1]

        model = keras.Model(inputs=x_in, outputs=x_out)

        X = gen.features
        A = gen.Aadj
        actual = model.predict([X, A])
        expected = np.ones((G.number_of_nodes(), self.layer_sizes[-1])) * (
            1.0 / G.number_of_nodes()
        )
        assert expected == pytest.approx(actual)

    def test_gat_node_model_no_norm(self):
        G = example_graph_1(feature_size=self.F_in)
        gen = FullBatchNodeGenerator(G)
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            generator=gen,
            bias=True,
            normalize=None,
        )

        gat._layers[1].kernel_initializer = keras.initializers.get("ones")
        gat._layers[1].attn_kernel_initializer = keras.initializers.get("ones")
        gat._layers[3].kernel_initializer = keras.initializers.get("ones")
        gat._layers[3].attn_kernel_initializer = keras.initializers.get("ones")

        assert len(gat.node_model()) == 2
        x_in, x_out = gat.node_model()
        assert len(x_in) == 2
        assert int(x_in[0].shape[-1]) == self.F_in
        assert x_in[1]._keras_shape == (None, G.number_of_nodes())
        assert int(x_out.shape[-1]) == self.layer_sizes[-1]

        model = keras.Model(inputs=x_in, outputs=x_out)

        X = gen.features
        A = gen.Aadj
        actual = model.predict([X, A])
        expected = np.ones((G.number_of_nodes(), self.layer_sizes[-1])) * (
            self.F_in * self.layer_sizes[0] * self.attn_heads
        )
        assert expected == pytest.approx(actual)

    def test_gat_node_model_wrong_norm(self):
        G = example_graph_1(feature_size=self.F_in)
        gen = FullBatchNodeGenerator(G)
        with pytest.raises(ValueError):
            gat = GAT(
                layer_sizes=self.layer_sizes,
                activations=self.activations,
                attn_heads=self.attn_heads,
                generator=gen,
                bias=True,
                normalize="whatever",
            )

    def test_gat_serialize(self):
        G = example_graph_1(feature_size=self.F_in)
        gen = FullBatchNodeGenerator(G)
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            generator=gen,
            bias=True,
            normalize="l2",
        )

        x_in, x_out = gat.node_model()
        model = keras.Model(inputs=x_in, outputs=x_out)

        # Save model
        model_json = model.to_json()

        # Set all weights to one
        model_weights = [np.ones_like(w) for w in model.get_weights()]

        # Load model from json & set all weights
        model2 = keras.models.model_from_json(
            model_json, custom_objects={"GraphAttention": GraphAttention}
        )
        model2.set_weights(model_weights)

        # Test loaded model
        X = gen.features
        A = gen.Aadj
        actual = model2.predict([X, A])
        expected = np.ones((G.number_of_nodes(), self.layer_sizes[-1])) * (
            1.0 / G.number_of_nodes()
        )
        assert expected == pytest.approx(actual)
