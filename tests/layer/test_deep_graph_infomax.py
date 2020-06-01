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

from stellargraph.layer import *
from stellargraph.mapper import *

from ..test_utils.graphs import example_graph_random
from .. import require_gpu
import tensorflow as tf
import pytest
import numpy as np


def _model_data(model_type, sparse):
    emb_dim = 16

    sparse_support = (GCN, APPNP, GAT, RGCN)
    if sparse and model_type not in sparse_support:
        pytest.skip(f"{model_type.__name__} doesn't support/use sparse=True")

    if model_type in (GCN, APPNP, GAT, PPNP):
        G = example_graph_random()
        generator = FullBatchNodeGenerator(G, sparse=sparse)
        model = model_type(
            generator=generator, activations=["relu"], layer_sizes=[emb_dim]
        )
        nodes = G.nodes()
    elif model_type is GraphSAGE:
        G = example_graph_random()
        generator = GraphSAGENodeGenerator(G, batch_size=5, num_samples=[2, 3])
        model = GraphSAGE(generator=generator, layer_sizes=[4, emb_dim])
        nodes = G.nodes()
    elif model_type is DirectedGraphSAGE:
        G = example_graph_random(is_directed=True)
        generator = DirectedGraphSAGENodeGenerator(
            G, batch_size=5, in_samples=[2, 3], out_samples=[4, 1]
        )
        model = DirectedGraphSAGE(generator=generator, layer_sizes=[4, emb_dim])
        nodes = G.nodes()
    elif model_type is HinSAGE:
        head_node_type = "n-1"
        node_types = 2
        G = example_graph_random(
            {nt: nt + 3 for nt in range(node_types)},
            node_types=node_types,
            edge_types=2,
        )
        generator = HinSAGENodeGenerator(
            G, batch_size=5, num_samples=[2, 2], head_node_type=head_node_type
        )
        model = HinSAGE(generator=generator, layer_sizes=[4, emb_dim])
        nodes = G.nodes(node_type=head_node_type)
    elif model_type is RGCN:
        G = example_graph_random(10, edge_types=3)
        generator = RelationalFullBatchNodeGenerator(G, sparse=sparse)
        model = RGCN([4, emb_dim], generator)
        nodes = G.nodes()

    return generator, model, nodes


@pytest.mark.parametrize(
    "model_type", [GCN, APPNP, GAT, PPNP, GraphSAGE, DirectedGraphSAGE, HinSAGE, RGCN]
)
@pytest.mark.parametrize("sparse", [False, True])
def test_dgi(model_type, sparse):
    base_generator, base_model, nodes = _model_data(model_type, sparse)
    corrupted_generator = CorruptedGenerator(base_generator)
    gen = corrupted_generator.flow(nodes)

    infomax = DeepGraphInfomax(base_model, corrupted_generator)

    model = tf.keras.Model(*infomax.in_out_tensors())
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer="Adam")
    model.fit(gen)

    emb_model = tf.keras.Model(*base_model.in_out_tensors())
    embeddings = emb_model.predict(base_generator.flow(nodes))

    if isinstance(
        base_generator, (FullBatchNodeGenerator, RelationalFullBatchNodeGenerator)
    ):
        assert embeddings.shape == (1, len(nodes), 16)
    else:
        assert embeddings.shape == (len(nodes), 16)


@pytest.mark.skipif(require_gpu, reason="tf on GPU is non-deterministic")
def test_dgi_stateful():
    G = example_graph_random()
    emb_dim = 16

    generator = FullBatchNodeGenerator(G)
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())

    infomax = DeepGraphInfomax(
        GCN(generator=generator, activations=["relu"], layer_sizes=[emb_dim]),
        corrupted_generator,
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


def test_dgi_deprecated_no_generator():
    G = example_graph_random()
    generator = FullBatchNodeGenerator(G)

    with pytest.warns(
        DeprecationWarning, match="The 'corrupted_generator' parameter should be set"
    ):
        DeepGraphInfomax(
            GCN(generator=generator, activations=["relu"], layer_sizes=[4]),
        )
