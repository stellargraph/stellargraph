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

import numpy as np
import tensorflow as tf
from stellargraph.layer.graph_classification import *
from stellargraph.layer import SortPooling
from stellargraph.mapper import PaddedGraphGenerator, FullBatchNodeGenerator
import pytest
from ..test_utils.graphs import example_graph_random


graphs = [
    example_graph_random(feature_size=4, n_nodes=6),
    example_graph_random(feature_size=4, n_nodes=5),
    example_graph_random(feature_size=4, n_nodes=3),
]

generator = PaddedGraphGenerator(graphs=graphs)


def test_init():

    model = GCNSupervisedGraphClassification(
        layer_sizes=[16], activations=["relu"], generator=generator
    )

    assert len(model.layer_sizes) == 1
    assert len(model.activations) == 1
    assert model.layer_sizes[0] == 16
    assert model.activations[0] == "relu"

    with pytest.raises(
        TypeError, match="generator: expected.*PaddedGraphGenerator, found NoneType"
    ):
        GCNSupervisedGraphClassification(
            layer_sizes=[16], activations=["relu"], generator=None
        )

    with pytest.raises(
        TypeError,
        match="generator: expected.*PaddedGraphGenerator, found FullBatchNodeGenerator",
    ):
        GCNSupervisedGraphClassification(
            layer_sizes=[16],
            activations=["relu"],
            generator=FullBatchNodeGenerator(graphs[0]),
        )

    with pytest.raises(
        ValueError,
        match="expected.*number of layers.*same as.*number of activations,found 2.*vs.*1",
    ):
        GCNSupervisedGraphClassification(
            layer_sizes=[16, 32], activations=["relu"], generator=generator
        )

    with pytest.raises(
        ValueError,
        match="expected.*number of layers.*same as.*number of activations,found 1.*vs.*2",
    ):
        GCNSupervisedGraphClassification(
            layer_sizes=[32], activations=["relu", "elu"], generator=generator
        )


def test_in_out_tensors():
    layer_sizes = [16, 8]
    activations = ["relu", "relu"]

    model = GCNSupervisedGraphClassification(
        layer_sizes=layer_sizes, activations=activations, generator=generator
    )

    x_in, x_out = model.in_out_tensors()

    assert len(x_in) == 3
    assert len(x_in[0].shape) == 3
    assert x_in[0].shape[-1] == 4  # the node feature dimensionality
    assert len(x_out.shape) == 2
    assert x_out.shape[-1] == layer_sizes[-1]


def test_stateful():
    layer_sizes = [16, 2]
    activations = ["elu", "elu"]
    targets = np.array([[0, 1], [0, 1], [1, 0]])
    train_graphs = [0, 1, 2]

    gcn_graph_model = GCNSupervisedGraphClassification(
        generator=generator, activations=activations, layer_sizes=layer_sizes
    )

    train_gen = generator.flow(graphs=train_graphs, targets=targets)

    model_1 = tf.keras.Model(*gcn_graph_model.in_out_tensors())
    model_2 = tf.keras.Model(*gcn_graph_model.in_out_tensors())

    # check embeddings are equal before training
    embeddings_1 = model_1.predict(train_gen)
    embeddings_2 = model_2.predict(train_gen)

    assert np.array_equal(embeddings_1, embeddings_2)

    model_1.compile(loss=tf.nn.softmax_cross_entropy_with_logits, optimizer="Adam")
    model_1.fit(train_gen)

    # check embeddings are still equal after training one model
    embeddings_1 = model_1.predict(train_gen)
    embeddings_2 = model_2.predict(train_gen)

    assert np.array_equal(embeddings_1, embeddings_2)

    model_2.compile(loss=tf.nn.softmax_cross_entropy_with_logits, optimizer="Adam")
    model_2.fit(train_gen)

    # check embeddings are still equal after training both models
    embeddings_1 = model_1.predict(train_gen)
    embeddings_2 = model_2.predict(train_gen)

    assert np.array_equal(embeddings_1, embeddings_2)


@pytest.mark.parametrize("pooling", ["default", "custom"])
def test_pooling(pooling):

    # no GCN layers, to just test the pooling directly
    if pooling == "default":
        gcn_graph_model = GCNSupervisedGraphClassification(
            layer_sizes=[], activations=[], generator=generator
        )

        def expected_values(array):
            return array.mean(axis=0)

    else:
        # shift the features to make it a bit more interesting
        shift = 10

        def shifted_sum_pooling(tensor, mask):
            mask_floats = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
            return tf.math.reduce_sum(tf.multiply(mask_floats, shift + tensor), axis=1)

        gcn_graph_model = GCNSupervisedGraphClassification(
            layer_sizes=[],
            activations=[],
            generator=generator,
            pooling=shifted_sum_pooling,
        )

        def expected_values(array):
            return (shift + array).sum(axis=0)

    train_graphs = [0, 1, 2]
    train_gen = generator.flow(graphs=train_graphs, batch_size=2, shuffle=False)
    model = tf.keras.Model(*gcn_graph_model.in_out_tensors())

    predictions = model.predict(train_gen)
    assert predictions.shape == (3, 4)

    expected = np.vstack(
        [
            expected_values(graphs[iloc].node_features(node_type="n-0"))
            for iloc in train_graphs
        ]
    )
    np.testing.assert_almost_equal(predictions, expected)


def test_pool_all_layers():
    gcn_graph_model = GCNSupervisedGraphClassification(
        layer_sizes=[5, 7, 11, 1],
        activations=["relu", "relu", "relu", "relu"],
        generator=generator,
        pool_all_layers=True,
    )

    train_graphs = [0, 1, 2]
    train_gen = generator.flow(graphs=train_graphs, batch_size=2)
    model = tf.keras.Model(*gcn_graph_model.in_out_tensors())

    predictions = model.predict(train_gen)
    assert predictions.shape == (3, 5 + 7 + 11 + 1)


def test_dgcnn_smoke():
    # this is entirely implemented in terms of GCNSupervisedGraphClassification, and so it's enough
    # to validate that the functionality is composed correctly.
    dgcnn = DeepGraphCNN(
        layer_sizes=[2, 3, 4],
        activations=["relu", "relu", "relu"],
        # one graph is perfect, one graph requires padding and one requires truncation
        k=5,
        generator=generator,
    )

    # validate the expectations of the implementation
    assert isinstance(dgcnn, GCNSupervisedGraphClassification)
    assert isinstance(dgcnn.pooling, SortPooling)
    assert dgcnn.pool_all_layers == True

    # check it gives output of the expected shape
    model = tf.keras.Model(*dgcnn.in_out_tensors())

    preds = model.predict(generator.flow([0, 1, 2]))
    assert preds.shape == (3, (2 + 3 + 4) * 5, 1)
