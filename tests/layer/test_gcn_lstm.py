# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
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
import pandas as pd
import pytest
from tensorflow.keras import Model
from stellargraph import StellarGraph, IndexedArray
from stellargraph.layer import GCN_LSTM
from stellargraph.layer import FixedAdjacencyGraphConvolution
from stellargraph.mapper import SlidingFeaturesNodeGenerator


def get_timeseries_graph_data():
    featuresX = np.random.rand(10, 5, 4)
    featuresY = np.random.rand(10, 5)
    adj = np.random.randint(0, 5, size=(5, 5))
    return featuresX, featuresY, adj


def test_GraphConvolution_config():

    _, _, a = get_timeseries_graph_data()

    gc_layer = FixedAdjacencyGraphConvolution(units=10, A=a, activation="relu")
    conf = gc_layer.get_config()

    assert conf["units"] == 10
    assert conf["activation"] == "relu"
    assert conf["use_bias"] == True
    assert conf["kernel_initializer"]["class_name"] == "GlorotUniform"
    assert conf["bias_initializer"]["class_name"] == "Zeros"
    assert conf["kernel_regularizer"] == None
    assert conf["bias_regularizer"] == None
    assert conf["kernel_constraint"] == None
    assert conf["bias_constraint"] == None


def test_gcn_lstm_model_parameters():
    fx, fy, a = get_timeseries_graph_data()

    gcn_lstm_model = GCN_LSTM(
        seq_len=fx.shape[-2],
        adj=a,
        gc_layer_sizes=[2, 2],
        gc_activations=["relu", "relu"],
        lstm_layer_sizes=[10],
        lstm_activations=["tanh"],
    )
    assert gcn_lstm_model.gc_activations == ["relu", "relu"]
    assert gcn_lstm_model.dropout == 0.5
    assert gcn_lstm_model.lstm_activations == ["tanh"]
    assert gcn_lstm_model.lstm_layer_sizes == [10]
    assert len(gcn_lstm_model.lstm_layer_sizes) == len(gcn_lstm_model.lstm_activations)


def test_gcn_lstm_activations():
    fx, fy, a = get_timeseries_graph_data()

    gcn_lstm_model = GCN_LSTM(
        seq_len=fx.shape[-2],
        adj=a,
        gc_layer_sizes=[10, 10, 10, 10, 10],
        lstm_layer_sizes=[8, 16, 32, 64],
    )
    # check when no activations provided, defaults to 'relu' and 'tanh' for gc and lstm respectively
    assert gcn_lstm_model.gc_activations == ["relu", "relu", "relu", "relu", "relu"]
    assert gcn_lstm_model.lstm_activations == ["tanh", "tanh", "tanh", "tanh"]

    gcn_lstm_model = GCN_LSTM(
        seq_len=fx.shape[-2],
        adj=a,
        gc_layer_sizes=[10],
        gc_activations=["relu"],
        lstm_layer_sizes=[8, 16, 32, 64],
    )

    assert gcn_lstm_model.lstm_activations == ["tanh", "tanh", "tanh", "tanh"]


def test_lstm_return_sequences():
    fx, fy, a = get_timeseries_graph_data()

    gcn_lstm_model = GCN_LSTM(
        seq_len=fx.shape[-2],
        adj=a,
        gc_layer_sizes=[16, 16, 16],
        gc_activations=["relu", "relu", "relu"],
        lstm_layer_sizes=[8, 16, 32],
        lstm_activations=["tanh"],
    )
    for layer in gcn_lstm_model._lstm_layers[:-1]:
        assert layer.return_sequences == True
    assert gcn_lstm_model._lstm_layers[-1].return_sequences == False


def test_gcn_lstm_layers():
    fx, fy, a = get_timeseries_graph_data()

    gcn_lstm_model = GCN_LSTM(
        seq_len=fx.shape[-2],
        adj=a,
        gc_layer_sizes=[8, 8, 16],
        gc_activations=["relu", "relu", "relu"],
        lstm_layer_sizes=[8, 16, 32],
        lstm_activations=["tanh"],
    )
    # check number of layers for gc and lstm
    assert len(gcn_lstm_model._gc_layers) == len(gcn_lstm_model.gc_layer_sizes)
    assert len(gcn_lstm_model._lstm_layers) == len(gcn_lstm_model.lstm_layer_sizes)


def test_gcn_lstm_model_input_output():
    fx, fy, a = get_timeseries_graph_data()

    gcn_lstm_model = GCN_LSTM(
        seq_len=fx.shape[-1],
        adj=a,
        gc_layer_sizes=[8, 8, 16],
        gc_activations=["relu", "relu", "relu"],
        lstm_layer_sizes=[8, 16, 32],
        lstm_activations=["tanh"],
    )

    # check model input and output tensors
    x_input, x_output = gcn_lstm_model.in_out_tensors()
    assert x_input.shape[1] == fx.shape[1]
    assert x_input.shape[2] == fx.shape[2]
    assert x_output.shape[1] == fx.shape[-2]


def test_gcn_lstm_model():
    fx, fy, a = get_timeseries_graph_data()

    gcn_lstm_model = GCN_LSTM(
        seq_len=fx.shape[-1],
        adj=a,
        gc_layer_sizes=[8, 8, 16],
        gc_activations=["relu", "relu", "relu"],
        lstm_layer_sizes=[8, 16, 32],
        lstm_activations=["tanh"],
    )

    x_input, x_output = gcn_lstm_model.in_out_tensors()
    model = Model(inputs=x_input, outputs=x_output)

    model.compile(optimizer="adam", loss="mae", metrics=["mse"])

    # check model training
    history = model.fit(fx, fy, epochs=5, batch_size=2, shuffle=True, verbose=0)

    assert history.params["epochs"] == 5
    assert len(history.history["loss"]) == 5


def test_gcn_lstm_model_prediction():
    fx, fy, a = get_timeseries_graph_data()

    gcn_lstm_model = GCN_LSTM(
        seq_len=fx.shape[-1],
        adj=a,
        gc_layer_sizes=[8, 8, 16],
        gc_activations=["relu", "relu", "relu"],
        lstm_layer_sizes=[8, 16, 32],
        lstm_activations=["tanh"],
    )

    x_input, x_output = gcn_lstm_model.in_out_tensors()
    model = Model(inputs=x_input, outputs=x_output)

    test_sample = np.random.rand(1, 5, 4)
    pred = model.predict(test_sample)

    # check 1 prediction for each node
    assert pred.shape == (1, 5)


@pytest.mark.parametrize("multivariate", [False, True])
def test_gcn_lstm_generator(multivariate):
    shape = (3, 7, 11) if multivariate else (3, 7)
    total_elems = np.product(shape)
    nodes = IndexedArray(
        np.arange(total_elems).reshape(shape) / total_elems, index=["a", "b", "c"]
    )
    edges = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"]})
    graph = StellarGraph(nodes, edges)

    gen = SlidingFeaturesNodeGenerator(graph, 2, batch_size=3)
    gcn_lstm = GCN_LSTM(None, None, [2], [4], generator=gen)

    model = Model(*gcn_lstm.in_out_tensors())

    model.compile("adam", loss="mse")

    history = model.fit(gen.flow(slice(0, 5), target_distance=1))

    predictions = model.predict(gen.flow(slice(5, 7)))

    model2 = Model(*gcn_lstm.in_out_tensors())
    predictions2 = model2.predict(gen.flow(slice(5, 7)))
    np.testing.assert_array_equal(predictions, predictions2)
