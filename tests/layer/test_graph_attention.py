# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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
import pytest
import scipy.sparse as sps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from stellargraph.mapper import (
    FullBatchNodeGenerator,
    FullBatchLinkGenerator,
    GraphSAGENodeGenerator,
)
from stellargraph.layer import *
from ..test_utils.graphs import example_graph
from .. import test_utils


pytestmark = test_utils.ignore_stellargraph_experimental_mark


class Test_GraphAttention:
    """
    Tests of GraphAttention layer
    """

    N = 10
    F_in = 5
    F_out = 2
    attn_heads = 8
    activation = "relu"
    layer = GraphAttention

    def get_inputs(self):
        x_inp = [
            Input(batch_shape=(1, self.N, self.F_in)),
            Input(batch_shape=(1, self.N, self.N)),
        ]

        # duplicate input here for Test_GraphAttentionSparse to work
        return x_inp, x_inp

    def get_matrix(self, edges=[]):
        # adjacency matrix with self-loops only
        A = np.eye(self.N)
        for e, v in edges:
            A[e[0], e[1]] = v
        return [A[None, :, :]]

    def test_constructor(self):
        # attn_heads_reduction = "concat":
        layer = self.layer(
            units=self.F_out,
            attn_heads=self.attn_heads,
            attn_heads_reduction="concat",
            activation=self.activation,
        )
        assert layer.units == self.F_out
        assert layer.attn_heads == self.attn_heads
        assert layer.output_dim == self.F_out * self.attn_heads
        assert layer.activation == keras.activations.get(self.activation)

        # attn_heads_reduction = "average":
        layer = self.layer(
            units=self.F_out,
            attn_heads=self.attn_heads,
            attn_heads_reduction="average",
            activation=self.activation,
        )
        assert layer.output_dim == self.F_out

        # attn_heads_reduction = "ave":
        with pytest.raises(ValueError):
            self.layer(
                units=self.F_out,
                attn_heads=self.attn_heads,
                attn_heads_reduction="ave",
                activation=self.activation,
            )

    def test_apply_concat(self):
        gat = self.layer(
            units=self.F_out,
            attn_heads=self.attn_heads,
            attn_heads_reduction="concat",
            activation=self.activation,
            kernel_initializer="ones",
        )
        x_inp, layer_inp = self.get_inputs()

        # Instantiate layer with squeezed matrix
        x_out = gat(layer_inp)

        model = keras.Model(inputs=x_inp, outputs=x_out)

        assert model.output_shape[-1] == self.F_out * self.attn_heads

        As = self.get_matrix()
        X = np.ones((1, self.N, self.F_in))  # features

        expected = np.ones((self.N, self.F_out * self.attn_heads)) * self.F_in
        actual = model.predict([X] + As)

        assert np.allclose(actual.squeeze(), expected)

    def test_apply_average(self):
        gat = self.layer(
            units=self.F_out,
            attn_heads=self.attn_heads,
            attn_heads_reduction="average",
            activation=self.activation,
            kernel_initializer="ones",
            attn_kernel_initializer="zeros",
            bias_initializer="zeros",
        )
        x_inp, layer_inp = self.get_inputs()

        # Instantiate layer with squeezed matrix
        x_out = gat(layer_inp)

        model = keras.Model(inputs=x_inp, outputs=x_out)
        assert model.output_shape[-1] == self.F_out

        X = np.ones((1, self.N, self.F_in))  # features
        for i in range(self.N):
            X[:, i, :] = i + 1

        As = self.get_matrix()

        expected = (X * self.F_in)[..., : self.F_out]
        actual = model.predict([X] + As)

        assert np.allclose(actual.squeeze(), expected)

    def test_apply_average_with_neighbours(self):
        gat_saliency = self.layer(
            units=self.F_out,
            attn_heads=self.attn_heads,
            attn_heads_reduction="average",
            activation=self.activation,
            kernel_initializer="ones",
            attn_kernel_initializer="zeros",
            bias_initializer="zeros",
            saliency_map_support=True,
        )

        gat_origin = self.layer(
            units=self.F_out,
            attn_heads=self.attn_heads,
            attn_heads_reduction="average",
            activation=self.activation,
            kernel_initializer="ones",
            attn_kernel_initializer="zeros",
            bias_initializer="zeros",
            saliency_map_support=False,
        )

        x_inp, layer_inp = self.get_inputs()

        # Instantiate layer with squeezed matrix
        x_out_saliency = gat_saliency(layer_inp)
        x_out_origin = gat_origin(layer_inp)

        model_origin = keras.Model(inputs=x_inp, outputs=x_out_origin)
        model_saliency = keras.Model(inputs=x_inp, outputs=x_out_saliency)
        assert model_origin.output_shape[-1] == self.F_out
        assert model_saliency.output_shape[-1] == self.F_out

        X = np.zeros((1, self.N, self.F_in))  # features
        for i in range(self.N):
            X[:, i, :] = i

        As = self.get_matrix([((0, 1), 1), ((1, 0), 1)])

        expected = (X * self.F_in)[..., : self.F_out]
        expected[:, :2] = self.F_in / 2
        actual_origin = model_origin.predict([X] + As)
        actual_saliency = model_saliency.predict([X] + As)
        assert np.allclose(expected, actual_origin)
        assert np.allclose(expected, actual_saliency)

    def test_layer_config(self):
        layer = self.layer(
            units=self.F_out,
            attn_heads=self.attn_heads,
            attn_heads_reduction="concat",
            activation=self.activation,
        )
        conf = layer.get_config()

        assert conf["units"] == self.F_out
        assert conf["attn_heads"] == self.attn_heads
        assert conf["attn_heads_reduction"] == "concat"
        assert conf["activation"] == self.activation
        assert conf["use_bias"] == True
        assert conf["kernel_initializer"]["class_name"] == "GlorotUniform"
        assert conf["bias_initializer"]["class_name"] == "Zeros"
        assert conf["kernel_regularizer"] == None
        assert conf["bias_regularizer"] == None
        assert conf["kernel_constraint"] == None
        assert conf["bias_constraint"] == None


class Test_GraphAttentionSparse(Test_GraphAttention):
    """
    Tests of GraphAttentionSparse layer
    """

    N = 10
    F_in = 5
    F_out = 2
    attn_heads = 8
    activation = "relu"
    layer = GraphAttentionSparse

    def get_inputs(self):
        x_inp = [
            Input(batch_shape=(1, self.N, self.F_in)),
            Input(batch_shape=(1, None, 2), dtype="int64"),
            Input(batch_shape=(1, None), dtype="float32"),
        ]

        A_mat = SqueezedSparseConversion(shape=(self.N, self.N))(x_inp[1:])

        # For dense matrix, remove batch dimension
        layer_inp = x_inp[:1] + [A_mat]

        return x_inp, layer_inp

    def get_matrix(self, edges=[]):
        # adjacency matrix with self-loops + edges
        A_sparse = sps.eye(self.N, format="lil")
        for e, v in edges:
            A_sparse[e[0], e[1]] = v
        # Extract indices & values to feed to tensorflow
        A_sparse = A_sparse.tocoo()
        A_indices = np.expand_dims(
            np.hstack((A_sparse.row[:, None], A_sparse.col[:, None])), 0
        )
        A_values = np.expand_dims(A_sparse.data, 0)
        return [A_indices, A_values]


class Test_GAT:
    """
    Tests of GAT class
    """

    N = 10
    F_in = 5
    F_out = 2
    attn_heads = 8
    layer_sizes = [4, 16]
    activations = ["relu", "linear"]
    sparse = False
    method = "gat"

    def test_constructor(self):
        G = example_graph(feature_size=self.F_in)
        gen = FullBatchNodeGenerator(G, sparse=self.sparse, method=self.method)
        # test default if no activations are passed:
        gat = GAT(layer_sizes=self.layer_sizes, generator=gen, bias=True)
        assert gat.activations == ["elu", "elu"]

        # test error if too many activations:
        with pytest.raises(ValueError):
            gat = GAT(layer_sizes=[10], activations=self.activations, generator=gen)

        # test error if too few activations:
        with pytest.raises(ValueError):
            gat = GAT(layer_sizes=[10, 10], activations=["relu"], generator=gen)

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

    def test_gat_build_constructor(self):
        G = example_graph(feature_size=self.F_in)
        gen = FullBatchNodeGenerator(G, sparse=self.sparse, method=self.method)
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            generator=gen,
            bias=True,
        )

        assert len(gat.in_out_tensors()) == 2
        x_in, x_out = gat.in_out_tensors()
        assert len(x_in) == 4 if self.sparse else 3
        assert int(x_in[0].shape[-1]) == self.F_in
        assert K.int_shape(x_in[-1]) == (1, G.number_of_nodes(), G.number_of_nodes())
        assert int(x_out.shape[-1]) == self.layer_sizes[-1]

    def test_gat_build_linkmodel_constructor(self):
        G = example_graph(feature_size=self.F_in)
        gen = FullBatchLinkGenerator(G, sparse=self.sparse, method=self.method)
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            generator=gen,
            bias=True,
        )

        assert len(gat.in_out_tensors()) == 2
        x_in, x_out = gat.in_out_tensors()
        assert len(x_in) == 4 if self.sparse else 3
        assert int(x_in[0].shape[-1]) == self.F_in
        assert int(x_out.shape[-1]) == self.layer_sizes[-1]

    def test_gat_build_constructor_no_generator(self):
        G = example_graph(feature_size=self.F_in)
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            bias=True,
            num_nodes=1000,
            num_features=self.F_in,
            multiplicity=1,
        )
        assert gat.use_sparse == False

        x_in, x_out = gat.in_out_tensors()
        assert len(x_in) == 4 if self.sparse else 3
        assert int(x_in[0].shape[-1]) == self.F_in
        assert int(x_out.shape[-1]) == self.layer_sizes[-1]

    def test_gat_build_constructor_wrong_generator(self):
        G = example_graph(feature_size=self.F_in)
        gen = GraphSAGENodeGenerator(G, self.N, [5, 10])

        # test error where generator is of the wrong type for GAT:
        with pytest.raises(TypeError):
            gat = GAT(
                layer_sizes=self.layer_sizes,
                activations=self.activations,
                attn_heads=self.attn_heads,
                bias=True,
                generator=gen,
            )

    def test_gat_build_l2norm(self):
        G = example_graph(feature_size=self.F_in)
        gen = FullBatchNodeGenerator(G, sparse=self.sparse, method=self.method)
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            generator=gen,
            bias=True,
            normalize="l2",
            kernel_initializer="ones",
            attn_kernel_initializer="ones",
        )

        x_in, x_out = gat.in_out_tensors()

        model = keras.Model(inputs=x_in, outputs=x_out)

        ng = gen.flow(G.nodes())
        actual = model.predict(ng)
        expected = np.ones((G.number_of_nodes(), self.layer_sizes[-1])) * (
            1.0 / G.number_of_nodes()
        )

        assert np.allclose(expected, actual[0])

    def test_gat_build_no_norm(self):
        G = example_graph(feature_size=self.F_in)
        gen = FullBatchNodeGenerator(G, sparse=self.sparse, method=self.method)
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            generator=gen,
            bias=True,
            normalize=None,
            kernel_initializer="ones",
            attn_kernel_initializer="ones",
        )

        x_in, x_out = gat.in_out_tensors()

        model = keras.Model(inputs=x_in, outputs=x_out)

        ng = gen.flow(G.nodes())
        actual = model.predict(ng)

        expected = np.ones((G.number_of_nodes(), self.layer_sizes[-1])) * (
            self.F_in
            * self.layer_sizes[0]
            * self.attn_heads
            * np.max(G.node_features(G.nodes()))
        )
        assert np.allclose(expected, actual[0])

    def test_gat_build_wrong_norm(self):
        G = example_graph(feature_size=self.F_in)
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
        G = example_graph(feature_size=self.F_in)
        gen = FullBatchNodeGenerator(G, sparse=self.sparse, method=self.method)
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            generator=gen,
            bias=True,
            normalize="l2",
        )

        x_in, x_out = gat.in_out_tensors()
        model = keras.Model(inputs=x_in, outputs=x_out)

        ng = gen.flow(G.nodes())

        # Save model
        model_json = model.to_json()

        # Set all weights to one
        model_weights = [np.ones_like(w) for w in model.get_weights()]

        # Load model from json & set all weights
        model2 = keras.models.model_from_json(
            model_json,
            custom_objects={
                "GraphAttention": GraphAttention,
                "GatherIndices": GatherIndices,
            },
        )
        model2.set_weights(model_weights)

        # Test deserialized model
        actual = model2.predict(ng)
        expected = np.ones((G.number_of_nodes(), self.layer_sizes[-1])) * (
            1.0 / G.number_of_nodes()
        )
        assert np.allclose(expected, actual[0])

    def test_kernel_and_bias_defaults(self):
        graph = example_graph(feature_size=self.F_in)
        gen = FullBatchNodeGenerator(graph, sparse=self.sparse, method=self.method)
        gat = GAT(
            layer_sizes=self.layer_sizes,
            activations=self.activations,
            attn_heads=self.attn_heads,
            generator=gen,
        )
        for layer in gat._layers:
            if isinstance(layer, GraphAttention):
                assert isinstance(
                    layer.kernel_initializer, tf.initializers.GlorotUniform
                )
                assert isinstance(layer.bias_initializer, tf.initializers.Zeros)
                assert isinstance(
                    layer.attn_kernel_initializer, tf.initializers.GlorotUniform
                )
                assert layer.kernel_regularizer is None
                assert layer.bias_regularizer is None
                assert layer.attn_kernel_regularizer is None
                assert layer.kernel_constraint is None
                assert layer.bias_constraint is None
                assert layer.attn_kernel_constraint is None


def TestGATsparse(Test_GAT):
    sparse = True
    method = "gat"
