# -*- coding: utf-8 -*-
#
# Copyright 2018-2019 Data61, CSIRO
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

import pytest
from stellargraph.utils.saliency_maps import *
import numpy as np
from stellargraph.layer import GraphAttention, GAT
from stellargraph import StellarGraph
from stellargraph.mapper import FullBatchNodeGenerator
from keras import Model, regularizers
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import networkx as nx
import keras.backend as K
import keras


def example_graph_1(feature_size=None):
    G = nx.Graph()
    elist = [(0, 1), (1, 2), (2, 3)]
    G.add_nodes_from([0, 1, 2, 3], label="default")
    G.add_edges_from(elist, label="default")

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = np.ones(feature_size)
        return StellarGraph(G, node_features="feature")

    else:
        return StellarGraph(G)


def create_GAT_model_dense(graph):
    generator = FullBatchNodeGenerator(graph, sparse=False, method="gat")
    train_gen = generator.flow([0, 1], np.array([[1, 0], [0, 1]]))

    layer_sizes = [2, 2]
    gat = GAT(
        layer_sizes=layer_sizes,
        attn_heads=2,
        generator=generator,
        bias=True,
        in_dropout=0,
        attn_dropout=0,
        activations=["elu", "softmax"],
        normalize=None,
        saliency_map_support=True,
    )

    for layer in gat._layers:
        layer._initializer = "ones"
    x_inp, x_out = gat.node_model()
    keras_model = Model(inputs=x_inp, outputs=x_out)
    return gat, keras_model, generator, train_gen


def create_GAT_model_sparse(graph):
    generator = FullBatchNodeGenerator(graph, sparse=True, method="gat")
    train_gen = generator.flow([0, 1], np.array([[1, 0], [0, 1]]))

    layer_sizes = [2, 2]
    gat = GAT(
        layer_sizes=layer_sizes,
        attn_heads=2,
        generator=generator,
        bias=True,
        in_dropout=0,
        attn_dropout=0,
        activations=["elu", "softmax"],
        normalize=None,
        saliency_map_support=True,
    )

    for layer in gat._layers:
        layer._initializer = "ones"
    x_inp, x_out = gat.node_model()
    keras_model = Model(inputs=x_inp, outputs=x_out)
    return gat, keras_model, generator, train_gen


def test_ig_saliency_map():
    graph = example_graph_1(feature_size=10)
    base_model, keras_model_gat, generator, train_gen = create_GAT_model_dense(graph)
    base_model_sp, keras_model_gat_sp, generator_sp, train_gen_sp = create_GAT_model_sparse(
        graph
    )

    keras_model_gat.compile(
        optimizer=Adam(lr=0.1), loss=categorical_crossentropy, weighted_metrics=["acc"]
    )

    keras_model_gat_sp.compile(
        optimizer=Adam(lr=0.1), loss=categorical_crossentropy, weighted_metrics=["acc"]
    )

    weights = keras_model_gat_sp.get_weights()
    keras_model_gat.set_weights(weights)
    ig_dense = IntegratedGradients(keras_model_gat, train_gen)
    ig_sparse = IntegratedGradients(keras_model_gat_sp, train_gen_sp)

    target_idx = 0
    class_of_interest = 0
    ig_node_importance_dense = ig_dense.get_node_importance(
        target_idx, class_of_interest, steps=500
    )
    ig_node_importance_sp = ig_sparse.get_node_importance(
        target_idx, class_of_interest, steps=500
    )
    assert pytest.approx(ig_node_importance_dense, ig_node_importance_sp)
    selected_nodes = np.array([[0, 1, 2]], dtype="int32")
    (X, _, A), _ = train_gen[0]
    prediction_dense = keras_model_gat.predict([X, selected_nodes, A]).squeeze()
    (X, _, A_index, A), _ = train_gen_sp[0]
    prediction_dense_sp = keras_model_gat_sp.predict(
        [X, selected_nodes, A_index, A]
    ).squeeze()
    assert pytest.approx(prediction_dense, prediction_dense_sp)

    ig_link_importance_dense = ig_dense.get_integrated_link_masks(
        target_idx, class_of_interest, A_baseline=None, steps=1
    )
    ig_link_importance_dense_nz = ig_link_importance_dense[
        np.nonzero(ig_link_importance_dense)
    ]
    ig_link_importance_sp = ig_sparse.get_integrated_link_masks(
        target_idx, class_of_interest, A_baseline=None, steps=1
    )
    ig_link_importance_sp_nz = ig_link_importance_sp[np.nonzero(ig_link_importance_sp)]
    assert pytest.approx(ig_link_importance_dense_nz, ig_link_importance_sp_nz)
