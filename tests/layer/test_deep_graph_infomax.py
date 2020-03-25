# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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

from stellargraph.layer import (
    DeepGraphInfomax,
    GCN,
    APPNP,
    GAT,
    PPNP,
    GraphSAGE,
    DirectedGraphSAGE,
)

from stellargraph.mapper import (
    FullBatchNodeGenerator,
    CorruptedGenerator,
    GraphSAGENodeGenerator,
    DirectedGraphSAGENodeGenerator,
    FullBatchLinkGenerator
)
from ..test_utils.graphs import example_graph_random
import tensorflow as tf
import pytest
import numpy as np


@pytest.mark.parametrize("model_type", [GCN, APPNP, GAT, PPNP])
@pytest.mark.parametrize("sparse", [False, True])
def test_dgi(model_type, sparse):

    if sparse and model_type is PPNP:
        pytest.skip("PPNP doesn't support sparse=True")

    G = example_graph_random()
    emb_dim = 16

    generator = FullBatchNodeGenerator(G, sparse=sparse)
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())

    base_model = model_type(
        generator=generator, activations=["relu"], layer_sizes=[emb_dim]
    )
    infomax = DeepGraphInfomax(base_model)

    model = tf.keras.Model(*infomax.in_out_tensors())
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer="Adam")
    model.fit(gen)

    emb_model = tf.keras.Model(*infomax.embedding_model())
    embeddings = emb_model.predict(generator.flow(G.nodes()))

    assert embeddings.shape == (len(G.nodes()), emb_dim)


def test_dgi_stateful():
    G = example_graph_random()
    emb_dim = 16

    generator = FullBatchNodeGenerator(G)
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())

    infomax = DeepGraphInfomax(
        GCN(generator=generator, activations=["relu"], layer_sizes=[emb_dim])
    )

    model_1 = tf.keras.Model(*infomax.in_out_tensors())
    model_2 = tf.keras.Model(*infomax.in_out_tensors())

    # check embeddings are equal before training
    embeddings_1 = tf.keras.Model(*infomax.embedding_model()).predict(
        generator.flow(G.nodes())
    )
    embeddings_2 = tf.keras.Model(*infomax.embedding_model()).predict(
        generator.flow(G.nodes())
    )

    assert np.array_equal(embeddings_1, embeddings_2)

    model_1.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer="Adam")
    model_1.fit(gen)

    # check embeddings are still equal after training one model
    embeddings_1 = tf.keras.Model(*infomax.embedding_model()).predict(
        generator.flow(G.nodes())
    )
    embeddings_2 = tf.keras.Model(*infomax.embedding_model()).predict(
        generator.flow(G.nodes())
    )

    assert np.array_equal(embeddings_1, embeddings_2)

    model_2.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer="Adam")
    model_2.fit(gen)

    # check embeddings are still equal after training both models
    embeddings_1 = tf.keras.Model(*infomax.embedding_model()).predict(
        generator.flow(G.nodes())
    )
    embeddings_2 = tf.keras.Model(*infomax.embedding_model()).predict(
        generator.flow(G.nodes())
    )

    assert np.array_equal(embeddings_1, embeddings_2)


@pytest.mark.parametrize("model_type", [GCN, APPNP, GAT])
def test_dgi_link_model(model_type):
    G = example_graph_random()
    emb_dim = 16

    generator = FullBatchLinkGenerator(G)

    with pytest.warns(
        UserWarning,
        match=r"base_model: expected a node model .* found a link model \(multiplicity = 2\)",
    ):
        infomax = DeepGraphInfomax(
            model_type(generator=generator, activations=["relu"], layer_sizes=[emb_dim])
        )

    # build should work
    _ = infomax.in_out_tensors()


@pytest.mark.parametrize("is_directed", [False, True])
def test_dgi_graphsage(is_directed):

    G = example_graph_random(is_directed=is_directed)

    if is_directed:
        generator = DirectedGraphSAGENodeGenerator(
            G, batch_size=5, in_samples=[2, 3], out_samples=[4, 1]
        )
        base_model = DirectedGraphSAGE(generator=generator, layer_sizes=[4, 4])
    else:
        generator = GraphSAGENodeGenerator(G, batch_size=5, num_samples=[2, 3])
        base_model = GraphSAGE(generator=generator, layer_sizes=[4, 4])

    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())

    infomax = DeepGraphInfomax(base_model)

    model = tf.keras.Model(*infomax.build())
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer="Adam")
    model.fit(gen)

    emb_model = tf.keras.Model(*infomax.embedding_model(model))
    embeddings = emb_model.predict(generator.flow(G.nodes()))

    assert embeddings.shape == (len(G.nodes()), 4)
