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
GraphSAGE tests

"""


from stellargraph.layer.graphsage import *
import keras
import numpy as np
import pytest


def test_mean_agg_constructor():
    agg = MeanAggregator(2)
    assert agg.output_dim == 2
    assert agg.half_output_dim == 1
    assert not agg.has_bias
    assert agg.act.__name__ == "relu"


def test_mean_agg_constructor_1():
    agg = MeanAggregator(output_dim=4, bias=True, act=lambda x: x + 1)
    assert agg.output_dim == 4
    assert agg.half_output_dim == 2
    assert agg.has_bias
    assert agg.act(2) == 3


def test_mean_agg_apply():
    agg = MeanAggregator(4, act=lambda x: x)
    agg._initializer = "ones"
    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(1, 2, 2))
    out = agg([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)
    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[[2, 2], [3, 3]]]])
    actual = model.predict([x1, x2])
    expected = np.array([[[2, 2, 5, 5]]])
    assert expected == pytest.approx(actual)


def test_graphsage_constructor():
    gs = GraphSAGE(layer_sizes=[4], n_samples=[2], input_dim=2)
    assert gs.dims == [2, 4]
    assert gs.n_samples == [2]
    assert gs.n_layers == 1
    assert gs.bias
    assert len(gs._aggs) == 1


def test_graphsage_constructor_1():
    gs = GraphSAGE(
        layer_sizes=[4, 6, 8], n_samples=[2, 4, 6], input_dim=2, bias=True, dropout=0.5
    )
    assert gs.dims == [2, 4, 6, 8]
    assert gs.n_samples == [2, 4, 6]
    assert gs.n_layers == 3
    assert gs.bias
    assert len(gs._aggs) == 3


def test_graphsage_apply():
    gs = GraphSAGE(layer_sizes=[4], n_samples=[2], bias=False, input_dim=2)
    gs._normalization = lambda x: x
    for agg in gs._aggs:
        agg._initializer = "ones"

    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(2, 2))
    out = gs([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)

    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[2, 2], [3, 3]]])

    actual = model.predict([x1, x2])
    expected = np.array([[[2, 2, 5, 5]]])
    assert expected == pytest.approx(actual)


def test_graphsage_apply_1():
    gs = GraphSAGE(layer_sizes=[2, 2, 2], n_samples=[2, 2, 2], bias=False, input_dim=2)
    gs._normalization = lambda z: z
    for agg in gs._aggs:
        agg._initializer = "ones"

    inp = [keras.Input(shape=(i, 2)) for i in [1, 2, 4, 8]]
    out = gs(inp)
    model = keras.Model(inputs=inp, outputs=out)

    x = [
        np.array([[[1, 1]]]),
        np.array([[[2, 2], [2, 2]]]),
        np.array([[[3, 3], [3, 3], [3, 3], [3, 3]]]),
        np.array([[[4, 4], [4, 4], [4, 4], [4, 4], [5, 5], [5, 5], [5, 5], [5, 5]]]),
    ]

    actual = model.predict(x)
    expected = np.array([[[16, 25]]])
    assert expected == pytest.approx(actual)
