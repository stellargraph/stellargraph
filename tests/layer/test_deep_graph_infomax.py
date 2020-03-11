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

from stellargraph.layer import DeepGraphInfoMax, GCN
from stellargraph.mapper import FullBatchNodeGenerator, CorruptedGenerator
from ..test_utils.graphs import example_graph_random
import tensorflow as tf
import pytest


def test_dgi():

    G = example_graph_random()
    emb_dim = 16

    generator = FullBatchNodeGenerator(G)
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())

    gcn = GCN(generator=generator, activations=["relu"], layer_sizes=[emb_dim])
    infomax = DeepGraphInfoMax(gcn)

    model = tf.keras.Model(*infomax.build())
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer="Adam")
    model.fit(gen)

    emb_model = tf.keras.Model(*infomax.embedding_model(model))
    embeddings = emb_model.predict(generator.flow(G.nodes()))

    assert embeddings.shape == (len(G.nodes()), emb_dim)


def test_dgi_embedding_model_wrong_model():
    G = example_graph_random()
    emb_dim = 16

    generator = FullBatchNodeGenerator(G)
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())

    infomax_1 = DeepGraphInfoMax(
        GCN(generator=generator, activations=["relu"], layer_sizes=[emb_dim])
    )
    infomax_2 = DeepGraphInfoMax(
        GCN(generator=generator, activations=["relu"], layer_sizes=[emb_dim])
    )

    model_1 = tf.keras.Model(*infomax_1.build())

    # check case when infomax_2.build() has not been called
    with pytest.raises(ValueError, match="model: *."):
        emb_model = tf.keras.Model(*infomax_2.embedding_model(model_1))

    # check case when infomax_2.build() has been called
    model_2 = tf.keras.Model(*infomax_2.build())
    with pytest.raises(ValueError, match="model: *."):
        infomax_2.embedding_model(model_1)

    with pytest.raises(ValueError, match="model: *."):
        infomax_1.embedding_model(model_2)

    emb_model_1 = tf.keras.Model(*infomax_1.embedding_model(model_1))
    emb_model_2 = tf.keras.Model(*infomax_2.embedding_model(model_2))
