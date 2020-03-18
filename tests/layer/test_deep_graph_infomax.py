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

from stellargraph.layer import DeepGraphInfomax, GCN, APPNP, GAT, PPNP
from stellargraph.mapper import FullBatchNodeGenerator, CorruptedGenerator
from ..test_utils.graphs import example_graph_random
import tensorflow as tf
import pytest
import numpy as np


@pytest.mark.parametrize("model_type", [GCN, APPNP, GAT, PPNP])
def test_dgi_dense(model_type):

    G = example_graph_random()
    emb_dim = 16

    generator = FullBatchNodeGenerator(G, sparse=False)
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())

    base_model = model_type(
        generator=generator, activations=["relu"], layer_sizes=[emb_dim]
    )
    infomax = DeepGraphInfomax(base_model)

    model = tf.keras.Model(*infomax.build())
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer="Adam")
    model.fit(gen)

    emb_model = tf.keras.Model(*infomax.embedding_model(model))
    embeddings = emb_model.predict(generator.flow(G.nodes()))

    assert embeddings.shape == (len(G.nodes()), emb_dim)


@pytest.mark.parametrize("model_type", [GCN, APPNP, GAT])
def test_dgi_sparse(model_type):

    G = example_graph_random()
    emb_dim = 16

    generator = FullBatchNodeGenerator(G, sparse=True)
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())

    base_model = model_type(
        generator=generator, activations=["relu"], layer_sizes=[emb_dim]
    )
    infomax = DeepGraphInfomax(base_model)

    model = tf.keras.Model(*infomax.build())
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer="Adam")
    model.fit(gen)

    emb_model = tf.keras.Model(*infomax.embedding_model(model))
    embeddings = emb_model.predict(generator.flow(G.nodes()))

    assert embeddings.shape == (len(G.nodes()), emb_dim)


def test_dgi_embedding_wrong_model():
    G = example_graph_random()
    emb_dim = 16

    generator = FullBatchNodeGenerator(G)

    infomax_1 = DeepGraphInfomax(
        GCN(generator=generator, activations=["relu"], layer_sizes=[emb_dim])
    )
    infomax_2 = DeepGraphInfomax(
        GCN(generator=generator, activations=["relu"], layer_sizes=[emb_dim])
    )

    model_1 = tf.keras.Model(*infomax_1.build())

    # check case when infomax_2.build() has not been called
    with pytest.raises(ValueError, match="model: *."):
        infomax_2.embedding_model(model_1)

    # check case when infomax_2.build() has been called
    model_2 = tf.keras.Model(*infomax_2.build())
    with pytest.raises(ValueError, match="model: *."):
        infomax_2.embedding_model(model_1)

    with pytest.raises(ValueError, match="model: *."):
        infomax_1.embedding_model(model_2)


def test_dgi_stateful():
    G = example_graph_random()
    emb_dim = 16

    generator = FullBatchNodeGenerator(G)
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())

    infomax = DeepGraphInfomax(
        GCN(generator=generator, activations=["relu"], layer_sizes=[emb_dim])
    )

    model_1 = tf.keras.Model(*infomax.build())
    model_2 = tf.keras.Model(*infomax.build())

    # check embeddings are equal before training
    embeddings_1 = tf.keras.Model(*infomax.embedding_model(model_1)).predict(
        generator.flow(G.nodes())
    )
    embeddings_2 = tf.keras.Model(*infomax.embedding_model(model_2)).predict(
        generator.flow(G.nodes())
    )

    assert np.array_equal(embeddings_1, embeddings_2)

    model_1.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer="Adam")
    model_1.fit(gen)

    # check embeddings are still equal after training one model
    embeddings_1 = tf.keras.Model(*infomax.embedding_model(model_1)).predict(
        generator.flow(G.nodes())
    )
    embeddings_2 = tf.keras.Model(*infomax.embedding_model(model_2)).predict(
        generator.flow(G.nodes())
    )

    assert np.array_equal(embeddings_1, embeddings_2)

    model_2.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer="Adam")
    model_2.fit(gen)

    # check embeddings are still equal after training both models
    embeddings_1 = tf.keras.Model(*infomax.embedding_model(model_1)).predict(
        generator.flow(G.nodes())
    )
    embeddings_2 = tf.keras.Model(*infomax.embedding_model(model_2)).predict(
        generator.flow(G.nodes())
    )

    assert np.array_equal(embeddings_1, embeddings_2)
