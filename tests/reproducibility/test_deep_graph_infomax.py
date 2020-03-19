# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .fixtures import assert_reproducible
from stellargraph.layer import DeepGraphInfomax, GCN, APPNP, GAT, PPNP
from stellargraph.mapper import FullBatchNodeGenerator, CorruptedGenerator
from ..test_utils.graphs import example_graph_random
import tensorflow as tf
import pytest
import numpy as np


def dgi(generator, gen, model_type):

    tf.random.set_seed(1234)
    np.random.seed(1234)

    emb_dim = 4

    base_model = model_type(
        generator=generator, activations=["relu"], layer_sizes=[emb_dim]
    )
    infomax = DeepGraphInfomax(base_model)

    model = tf.keras.Model(*infomax.build())
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer="Adam")
    model.fit(gen)
    return model


@pytest.mark.parametrize("model_type", [GCN, APPNP, GAT])
def test_dgi_sparse(model_type):
    G = example_graph_random()
    generator = FullBatchNodeGenerator(G, sparse=True)
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())

    assert_reproducible(lambda: dgi(generator, gen, model_type), num_iter=3)


@pytest.mark.parametrize("model_type", [GCN, APPNP, GAT, PPNP])
def test_dgi_dense(model_type):
    G = example_graph_random()
    generator = FullBatchNodeGenerator(G, sparse=False)
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())

    assert_reproducible(lambda: dgi(generator, gen, model_type), num_iter=3)
