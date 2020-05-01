# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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
Cluster-GCN tests

"""
from tensorflow.keras import backend as K
from stellargraph.layer.cluster_gcn import *
from stellargraph.mapper import ClusterNodeGenerator
from stellargraph.core.graph import StellarGraph

import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pytest
from ..test_utils.graphs import create_graph_features


def test_ClusterGCN_init():
    G, features = create_graph_features()

    generator = ClusterNodeGenerator(G)
    cluster_gcn_model = ClusterGCN(
        layer_sizes=[2], generator=generator, activations=["relu"], dropout=0.5
    )

    assert cluster_gcn_model.layer_sizes == [2]
    assert cluster_gcn_model.activations == ["relu"]
    assert cluster_gcn_model.dropout == 0.5


def test_ClusterGCN_apply():

    G, _ = create_graph_features()

    generator = ClusterNodeGenerator(G)

    cluster_gcn_model = ClusterGCN(
        layer_sizes=[2], generator=generator, activations=["relu"], dropout=0.0
    )

    x_in, x_out = cluster_gcn_model.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    preds_2 = model.predict(generator.flow(["a", "b", "c"]))
    assert preds_2.shape == (1, 3, 2)


def test_ClusterGCN_activations():

    G, _ = create_graph_features()
    generator = ClusterNodeGenerator(G)

    # Test activations are set correctly
    cluster_gcn = ClusterGCN(layer_sizes=[2], generator=generator, activations=["relu"])
    assert cluster_gcn.activations == ["relu"]

    cluster_gcn = ClusterGCN(
        layer_sizes=[2, 2], generator=generator, activations=["relu", "relu"]
    )
    assert cluster_gcn.activations == ["relu", "relu"]

    cluster_gcn = ClusterGCN(
        layer_sizes=[2], generator=generator, activations=["linear"]
    )
    assert cluster_gcn.activations == ["linear"]

    with pytest.raises(TypeError):
        # activations for layers must be specified
        ClusterGCN(layer_sizes=[2], generator=generator)

    with pytest.raises(AssertionError):
        # More activations than layers
        ClusterGCN(layer_sizes=[2], generator=generator, activations=["relu", "linear"])

    with pytest.raises(AssertionError):
        # Fewer activations than layers
        ClusterGCN(layer_sizes=[2, 2], generator=generator, activations=["relu"])

    with pytest.raises(ValueError):
        # Unknown activation
        ClusterGCN(layer_sizes=[2], generator=generator, activations=["bleach"])


def test_ClusterGCN_regularisers():
    G, _ = create_graph_features()

    generator = ClusterNodeGenerator(G)

    cluster_gcn = ClusterGCN(
        layer_sizes=[2],
        activations=["relu"],
        generator=generator,
        kernel_regularizer=keras.regularizers.l2(),
    )

    with pytest.raises(ValueError):
        ClusterGCN(
            layer_sizes=[2],
            activations=["relu"],
            generator=generator,
            kernel_regularizer="fred",
        )

    cluster_gcn = ClusterGCN(
        layer_sizes=[2],
        activations=["relu"],
        generator=generator,
        bias_initializer="zeros",
    )

    cluster_gcn = ClusterGCN(
        layer_sizes=[2],
        activations=["relu"],
        generator=generator,
        bias_initializer=initializers.zeros(),
    )

    with pytest.raises(ValueError):
        ClusterGCN(
            layer_sizes=[2],
            activations=["relu"],
            generator=generator,
            bias_initializer="barney",
        )


def test_kernel_and_bias_defaults():
    graph, _ = create_graph_features()
    generator = ClusterNodeGenerator(graph)
    cluster_gcn = ClusterGCN(
        layer_sizes=[2, 2], activations=["relu", "relu"], generator=generator
    )
    for layer in cluster_gcn._layers:
        if isinstance(layer, GraphConvolution):
            assert isinstance(layer.kernel_initializer, tf.initializers.GlorotUniform)
            assert isinstance(layer.bias_initializer, tf.initializers.Zeros)
            assert layer.kernel_regularizer is None
            assert layer.bias_regularizer is None
            assert layer.kernel_constraint is None
            assert layer.bias_constraint is None
