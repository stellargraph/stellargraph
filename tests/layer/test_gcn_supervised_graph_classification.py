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
from stellargraph.layer.graph_classification import *
from stellargraph.mapper import GraphGenerator, FullBatchNodeGenerator
import pytest
from ..test_utils.graphs import example_graph_random


graphs = [
    example_graph_random(feature_size=4, n_nodes=6),
    example_graph_random(feature_size=4, n_nodes=5),
    example_graph_random(feature_size=4, n_nodes=3),
]

generator = GraphGenerator(graphs=graphs)


def test_init():

    model = GCNSupervisedGraphClassification(
        layer_sizes=[16], activations=["relu"], generator=generator
    )

    assert len(model.layer_sizes) == 1
    assert len(model.activations) == 1
    assert model.layer_sizes[0] == 16
    assert model.activations[0] == "relu"

    with pytest.raises(
        TypeError, match="generator: expected.*GraphGenerator, found NoneType"
    ):
        GCNSupervisedGraphClassification(
            layer_sizes=[16], activations=["relu"], generator=None
        )

    with pytest.raises(
        TypeError,
        match="generator: expected.*GraphGenerator, found FullBatchNodeGenerator",
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

    train_gen = generator.flow(graph_ilocs=train_graphs, targets=targets)

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
