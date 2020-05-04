#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:02:32 2020

@author: hab031
"""

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

from stellargraph.layer import GraphConvolutionLSTM

import numpy as np
from tensorflow import keras
from tensorflow.keras import Model
from stellargraph.layer import GraphConvolutionLSTM
from stellargraph.layer import FixedAdjacencyGraphConvolution
import pytest


def get_timeseries_graph_data():
    featuresX = np.random.rand(10, 4, 5)
    featuresY = np.random.rand(10, 5)
    adj = np.random.randint(0, 5, size=(5, 5))
    return featuresX, featuresY, adj


def test_GraphConvolution_config():

    _, _, a = get_timeseries_graph_data()

    gc_layer = FixedAdjacencyGraphConvolution(units=10, A=a, activation="relu")
    conf = gc_layer.get_config()

    assert conf["units"] == (10,)
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

    gcn_lstm_model = GraphConvolutionLSTM(
        seq_len=fx.shape[-2],
        adj=a,
        gc_layers=2,
        gc_activations=["relu", "relu"],
        lstm_layer_size=[10],
        lstm_activations=["tanh"],
    )
    assert gcn_lstm_model.gc_activations == ["relu", "relu"]
    assert gcn_lstm_model.dropout == 0.5
    assert gcn_lstm_model.lstm_activations == ["tanh"]
    assert gcn_lstm_model.lstm_layer_size == [10]
    assert len(gcn_lstm_model.lstm_layer_size) == len(gcn_lstm_model.lstm_activations)


def test_gcn_lstm_activations():
    fx, fy, a = get_timeseries_graph_data()

    gcn_lstm_model = GraphConvolutionLSTM(
        seq_len=fx.shape[-2], adj=a, gc_layers=5, lstm_layer_size=[8, 16, 32, 64],
    )
    assert gcn_lstm_model.gc_activations == ["relu", "relu", "relu", "relu", "relu"]
    assert gcn_lstm_model.lstm_activations == ["tanh", "tanh", "tanh", "tanh"]

    with pytest.raises(ValueError):
        # More regularisers than layers
        gcn_lstm_model = GraphConvolutionLSTM(
            seq_len=fx.shape[-2],
            adj=a,
            gc_layers=2,
            gc_activations=["relu"],
            lstm_layer_size=[8, 16, 32, 64],
        )

    with pytest.raises(ValueError):
        # More regularisers than layers
        gcn_lstm_model = GraphConvolutionLSTM(
            seq_len=fx.shape[-2],
            adj=a,
            gc_layers=1,
            gc_activations=["relu"],
            lstm_layer_size=[32],
            lstm_activations=["tanh", "tanh"],
        )


def test_lstm_return_sequences():
    fx, fy, a = get_timeseries_graph_data()

    gcn_lstm_model = GraphConvolutionLSTM(
        seq_len=fx.shape[-2],
        adj=a,
        gc_layers=3,
        gc_activations=["relu", "relu", "relu"],
        lstm_layer_size=[8, 16, 32],
        lstm_activations=["tanh"],
    )
    n_layers = len(gcn_lstm_model._layers)
    n_gc_layers = len(gcn_lstm_model.gc_activations)
    for i in range(n_gc_layers, n_layers - 3):
        assert gcn_lstm_model._layers[i].return_sequences == True
    assert gcn_lstm_model._layers[n_layers - 3].return_sequences == False
