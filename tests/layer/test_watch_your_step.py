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

from stellargraph.layer import AttentiveWalk, WatchYourStep, get_embeddings
import numpy as np
from ..test_utils.graphs import barbell
from stellargraph.mapper import AdjacencyPowerGenerator
from stellargraph.losses import graph_log_likelihood
import pytest
from tensorflow.keras import Model


def test_AttentiveWalk_config():
    att_wlk = AttentiveWalk(walk_length=10)
    conf = att_wlk.get_config()

    assert conf["walk_length"] == 10
    assert conf["attention_initializer"]["class_name"] == "GlorotUniform"
    assert conf["attention_regularizer"] is None
    assert conf["attention_constraint"] is None


def test_AttentiveWalk():

    random_partial_powers = np.random.random((2, 5, 31))
    att_wlk = AttentiveWalk(walk_length=5, attention_initializer="ones")

    output = att_wlk(random_partial_powers).numpy()

    assert np.allclose(output, random_partial_powers.mean(axis=1))


def test_WatchYourStep_init(barbell):
    generator = AdjacencyPowerGenerator(barbell, num_powers=5)
    wys = WatchYourStep(generator)

    assert wys.num_powers == 5
    assert wys.n_nodes == len(barbell.nodes())
    assert wys.num_walks == 80
    assert wys.embedding_dimension == 64
    assert wys.attention_initializer == "glorot_uniform"
    assert wys.attention_constraint is None
    assert wys.attention_regularizer is None


def test_WatchYourStep_bad_init(barbell):

    generator = AdjacencyPowerGenerator(barbell, num_powers=5)

    with pytest.raises(TypeError, match="num_walks: expected.* found float"):
        wys = WatchYourStep(generator, num_walks=10.0)

    with pytest.raises(ValueError, match="num_walks: expected.* found 0"):
        wys = WatchYourStep(generator, num_walks=0)

    with pytest.raises(TypeError, match="embedding_dimension: expected.* found float"):
        wys = WatchYourStep(generator, embedding_dimension=10.0)

    with pytest.raises(ValueError, match="embedding_dimension: expected.* found 1"):
        wys = WatchYourStep(generator, embedding_dimension=1)


def test_WatchYourStep(barbell):

    generator = AdjacencyPowerGenerator(barbell, num_powers=5)
    gen = generator.flow(batch_size=4)
    wys = WatchYourStep(generator)

    x_in, x_out = wys.build()

    model = Model(inputs=x_in, outputs=x_out)
    model.compile(optimizer="adam", loss=graph_log_likelihood)
    model.fit(gen, epochs=1, steps_per_epoch=int(len(barbell.nodes()) // 4))

    embs = get_embeddings(model)

    assert embs.shape == (len(barbell.nodes()), wys.embedding_dimension)

    model2 = Model(*wys.build())
    assert np.array_equal(get_embeddings(model2), embs)


def test_WatchYourStep_embeddings(barbell):
    generator = AdjacencyPowerGenerator(barbell, num_powers=5)
    wys = WatchYourStep(generator, embeddings_initializer="ones")
    x_in, x_out = wys.build()

    model = Model(inputs=x_in, outputs=x_out)
    model.compile(optimizer="adam", loss=graph_log_likelihood)
    embs = get_embeddings(model)

    assert (embs == 1).all()
