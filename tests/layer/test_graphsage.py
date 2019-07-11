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
from stellargraph.core.graph import StellarGraph
from stellargraph.mapper.node_mappers import GraphSAGENodeGenerator
from stellargraph.layer.graphsage import *

import keras
import numpy as np
import networkx as nx
import pytest

from keras.engine import saving


def example_graph_1(feature_size=None):
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_nodes_from([1, 2, 3, 4], label="default")
    G.add_edges_from(elist, label="default")

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = np.ones(feature_size)
        return StellarGraph(G, node_features="feature")

    else:
        return StellarGraph(G)


# MaxPooling aggregator tests


def test_maxpool_agg_constructor():
    agg = MaxPoolingAggregator(2, bias=False)
    assert agg.output_dim == 2
    assert agg.half_output_dim == 1
    assert agg.hidden_dim == 2
    assert not agg.has_bias
    assert agg.act.__name__ == "relu"
    assert agg.hidden_act.__name__ == "relu"

    # Check config
    config = agg.get_config()
    assert config["output_dim"] == 2
    assert config["bias"] == False
    assert config["act"] == "relu"


def test_maxpool_agg_constructor_1():
    agg = MaxPoolingAggregator(output_dim=4, bias=True, act=lambda x: x + 1)
    assert agg.output_dim == 4
    assert agg.half_output_dim == 2
    assert agg.hidden_dim == 4
    assert agg.has_bias
    assert agg.act(2) == 3

    # Test for output dim not divisible by 2
    with pytest.raises(ValueError):
        MaxPoolingAggregator(output_dim=3)


def test_maxpool_agg_apply():
    agg = MaxPoolingAggregator(2, bias=True, act="linear")
    agg._initializer = "ones"

    # Self features
    inp1 = keras.Input(shape=(1, 2))
    # Neighbour features
    inp2 = keras.Input(shape=(1, 2, 2))

    # Numerical test values
    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[[2, 2], [3, 3]]]])
    # Agg output:
    # neigh_agg = max(relu(x2 · ones(2x2)) + 1, axis=1)
    #   = max([[5,5],[7,7]]) = [[7,7]]
    # from_self = K.dot(x1, ones) = [[2]]
    # from_neigh = K.dot(neigh_agg, ones) = [[14]]

    out = agg([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)
    actual = model.predict([x1, x2])
    expected = np.array([[[2, 14]]])

    assert expected == pytest.approx(actual)


def test_maxpool_agg_zero_neighbours():
    agg = MaxPoolingAggregator(4, bias=False, act="linear")
    agg._initializer = "ones"

    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(1, 0, 2))

    out = agg([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)

    x1 = np.array([[[1, 1]]])
    x2 = np.zeros((1, 1, 0, 2))

    actual = model.predict([x1, x2])
    expected = np.array([[[2, 2, 2, 2]]])
    assert expected == pytest.approx(actual)


# MeanPooling aggregator tests


def test_meanpool_agg_constructor():
    agg = MeanPoolingAggregator(2, bias=False)
    assert agg.output_dim == 2
    assert agg.half_output_dim == 1
    assert agg.hidden_dim == 2
    assert not agg.has_bias
    assert agg.act.__name__ == "relu"
    assert agg.hidden_act.__name__ == "relu"

    # Check config
    config = agg.get_config()
    assert config["output_dim"] == 2
    assert config["bias"] == False
    assert config["act"] == "relu"


def test_meanpool_agg_constructor_1():
    agg = MeanPoolingAggregator(output_dim=4, bias=True, act=lambda x: x + 1)
    assert agg.output_dim == 4
    assert agg.half_output_dim == 2
    assert agg.hidden_dim == 4
    assert agg.has_bias
    assert agg.act(2) == 3

    # Test for output dim not divisible by 2
    with pytest.raises(ValueError):
        MeanPoolingAggregator(output_dim=3)


def test_meanpool_agg_apply():
    agg = MeanPoolingAggregator(2, bias=True, act="linear")
    agg._initializer = "ones"

    # Self features
    inp1 = keras.Input(shape=(1, 2))
    # Neighbour features
    inp2 = keras.Input(shape=(1, 2, 2))

    # Numerical test values
    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[[2, 2], [3, 3]]]])
    # Agg output:
    # neigh_agg = mean(relu(x2 · ones(2x2)) + 1, axis=1)
    #   = mean([[5,5],[7,7]]) = [[6,6]]
    # from_self = K.dot(x1, ones) = [[2]]
    # from_neigh = K.dot(neigh_agg, ones) = [[12]]

    out = agg([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)
    actual = model.predict([x1, x2])
    expected = np.array([[[2, 12]]])

    assert expected == pytest.approx(actual)


def test_meanpool_agg_zero_neighbours():
    agg = MeanPoolingAggregator(4, bias=False, act="linear")
    agg._initializer = "ones"

    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(1, 0, 2))

    out = agg([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)

    x1 = np.array([[[1, 1]]])
    x2 = np.zeros((1, 1, 0, 2))

    actual = model.predict([x1, x2])
    expected = np.array([[[2, 2, 2, 2]]])
    assert expected == pytest.approx(actual)


# Mean aggregator tests
def test_mean_agg_constructor():
    agg = MeanAggregator(2)
    assert agg.output_dim == 2
    assert agg.half_output_dim == 1
    assert not agg.has_bias

    # Check config
    config = agg.get_config()
    assert config["output_dim"] == 2
    assert config["bias"] == False
    assert config["act"] == "relu"


def test_mean_agg_constructor_1():
    agg = MeanAggregator(output_dim=4, bias=True, act=lambda x: x + 1)
    assert agg.output_dim == 4
    assert agg.half_output_dim == 2
    assert agg.has_bias
    assert agg.act(2) == 3

    # Test for output dim not divisible by 2
    with pytest.raises(ValueError):
        MeanAggregator(output_dim=3)


def test_mean_agg_apply():
    agg = MeanAggregator(4, bias=True, act=lambda x: x)
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


def test_mean_agg_zero_neighbours():
    agg = MeanAggregator(4, bias=False, act=lambda x: x)
    agg._initializer = "ones"

    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(1, 0, 2))

    out = agg([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)

    x1 = np.array([[[1, 1]]])
    x2 = np.zeros((1, 1, 0, 2))

    actual = model.predict([x1, x2])
    expected = np.array([[[2, 2, 2, 2]]])
    assert expected == pytest.approx(actual)


# Attentional aggregator tests
def test_attn_agg_constructor():
    agg = AttentionalAggregator(2, bias=False)
    assert agg.output_dim == 2
    assert not agg.has_bias
    assert agg.act.__name__ == "relu"
    # assert agg.attn_act.__name__ == "relu"

    # Check config
    config = agg.get_config()
    assert config["output_dim"] == 2
    assert config["bias"] == False
    assert config["act"] == "relu"


def test_attn_agg_constructor_1():
    agg = AttentionalAggregator(output_dim=4, bias=True, act=lambda x: x + 1)
    assert agg.output_dim == 4
    assert agg.has_bias
    assert agg.act(2) == 3


def test_attn_agg_apply():
    agg = AttentionalAggregator(2, bias=True, act="linear")
    agg._initializer = "ones"
    agg.attn_act = keras.activations.get("relu")

    # Self features
    inp1 = keras.Input(shape=(1, 2))
    # Neighbour features
    inp2 = keras.Input(shape=(1, 2, 2))

    # Numerical test values
    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[[2, 2], [3, 3]]]])

    # Agg output:
    # hs = relu(x1 · ones(2x2)) = [2,2]
    # hn = relu(x2 · ones(2x2)) =  [[2,2], [4,4],[6,6]]
    # attn_u = ones(2) · hs +  ones(2) · hn = [8, 12, 16]
    # attn = softmax(attn_u) = [3.3e-4, 1.8e-4, 9.81e-1]
    # hout =  attn · hn = [5.96, 5.96]

    out = agg([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)
    actual = model.predict([x1, x2])
    expected = np.array([[[5.963, 5.963]]])

    assert expected == pytest.approx(actual, rel=1e-4)


def test_attn_agg_zero_neighbours():
    agg = AttentionalAggregator(4, bias=False, act="linear")
    agg._initializer = "ones"

    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(1, 0, 2))

    out = agg([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)

    x1 = np.array([[[1, 1]]])
    x2 = np.zeros((1, 1, 0, 2))

    actual = model.predict([x1, x2])
    expected = np.array([[[2, 2, 2, 2]]])
    assert expected == pytest.approx(actual)


def test_graphsage_constructor():
    gs = GraphSAGE(layer_sizes=[4], n_samples=[2], input_dim=2, normalize="l2")
    assert gs.dims == [2, 4]
    assert gs.n_samples == [2]
    assert gs.n_layers == 1
    assert gs.bias
    assert len(gs._aggs) == 1

    # Check incorrect normalization flag
    with pytest.raises(ValueError):
        GraphSAGE(layer_sizes=[4], n_samples=[2], input_dim=2, normalize=lambda x: x)

    with pytest.raises(ValueError):
        GraphSAGE(layer_sizes=[4], n_samples=[2], input_dim=2, normalize="unknown")

    # Check requirement for generator or n_samples
    with pytest.raises(RuntimeError):
        GraphSAGE(layer_sizes=[4])

    # Construction from generator
    G = example_graph_1(feature_size=3)
    gen = GraphSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2]).flow([1, 2])
    gs = GraphSAGE(layer_sizes=[4, 8], generator=gen, bias=True)

    assert gs.dims == [3, 4, 8]
    assert gs.n_samples == [2, 2]
    assert gs.n_layers == 2
    assert gs.bias
    assert len(gs._aggs) == 2


def test_graphsage_constructor_passing_aggregator():
    gs = GraphSAGE(
        layer_sizes=[4], n_samples=[2], input_dim=2, aggregator=MeanAggregator
    )
    assert gs.dims == [2, 4]
    assert gs.n_samples == [2]
    assert gs.n_layers == 1
    assert gs.bias
    assert len(gs._aggs) == 1

    with pytest.raises(TypeError):
        GraphSAGE(layer_sizes=[4], n_samples=[2], input_dim=2, aggregator=1)


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
    gs = GraphSAGE(
        layer_sizes=[4], n_samples=[2], bias=False, input_dim=2, normalize=None
    )
    for agg in gs._aggs:
        agg._initializer = "ones"

    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(2, 2))
    out = gs([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)


def test_graphsage_apply_1():
    gs = GraphSAGE(
        layer_sizes=[2, 2, 2],
        n_samples=[2, 2, 2],
        bias=False,
        input_dim=2,
        normalize=None,
    )
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

    # Use the node model:
    xinp, xout = gs.node_model(flatten_output=False)
    model2 = keras.Model(inputs=xinp, outputs=xout)

    expected = np.array([[[16, 25]]])
    assert pytest.approx(expected) == model2.predict(x)


def test_graphsage_serialize():
    gs = GraphSAGE(
        layer_sizes=[4], n_samples=[2], bias=False, input_dim=2, normalize=None
    )

    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(2, 2))
    out = gs([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)

    # Save model
    model_json = model.to_json()

    # Set all weights to one
    model_weights = [np.ones_like(w) for w in model.get_weights()]

    # Load model from json & set all weights
    model2 = keras.models.model_from_json(
        model_json, custom_objects={"MeanAggregator": MeanAggregator}
    )
    model2.set_weights(model_weights)

    # Test loaded model
    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[2, 2], [3, 3]]])
    actual = model2.predict([x1, x2])
    expected = np.array([[[2, 2, 5, 5]]])
    assert expected == pytest.approx(actual)


def test_graphsage_zero_neighbours():
    gs = GraphSAGE(
        layer_sizes=[2, 2], n_samples=[0, 0], bias=False, input_dim=2, normalize="none"
    )

    for agg in gs._aggs:
        agg._initializer = "ones"

    inp = [keras.Input(shape=(i, 2)) for i in [1, 0, 0]]
    out = gs(inp)
    model = keras.Model(inputs=inp, outputs=out)

    x = [np.array([[[1.5, 1]]]), np.zeros((1, 0, 2)), np.zeros((1, 0, 2))]

    actual = model.predict(x)
    expected = np.array([[[5, 5]]])
    assert actual == pytest.approx(expected)
