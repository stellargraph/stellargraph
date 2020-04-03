# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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
from stellargraph.interpretability.saliency_maps import *
import numpy as np
from stellargraph.layer import GCN
from stellargraph.mapper import FullBatchNodeGenerator
from tensorflow.keras import Model, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from ..test_utils.graphs import example_graph_1_saliency_maps as example_graph_1


def create_GCN_model_dense(graph):
    generator = FullBatchNodeGenerator(graph, sparse=False, method="gcn")
    train_gen = generator.flow([0, 1], np.array([[1, 0], [0, 1]]))

    layer_sizes = [2, 2]
    gcn = GCN(
        layer_sizes=layer_sizes,
        activations=["elu", "elu"],
        generator=generator,
        dropout=0.3,
        kernel_regularizer=regularizers.l2(5e-4),
    )

    for layer in gcn._layers:
        layer._initializer = "ones"
    x_inp, x_out = gcn.in_out_tensors()
    keras_model = Model(inputs=x_inp, outputs=x_out)
    return gcn, keras_model, generator, train_gen


def create_GCN_model_sparse(graph):
    generator = FullBatchNodeGenerator(graph, sparse=True, method="gcn")
    train_gen = generator.flow([0, 1], np.array([[1, 0], [0, 1]]))

    layer_sizes = [2, 2]
    gcn = GCN(
        layer_sizes=layer_sizes,
        activations=["elu", "elu"],
        generator=generator,
        dropout=0.3,
        kernel_regularizer=regularizers.l2(5e-4),
    )

    for layer in gcn._layers:
        layer._initializer = "ones"
    x_inp, x_out = gcn.in_out_tensors()
    keras_model = Model(inputs=x_inp, outputs=x_out)
    return gcn, keras_model, generator, train_gen


def test_ig_saliency_map():

    graph = example_graph_1(feature_size=4)
    base_model, keras_model_gcn, generator, train_gen = create_GCN_model_dense(graph)
    (
        base_model_sp,
        keras_model_gcn_sp,
        generator_sp,
        train_gen_sp,
    ) = create_GCN_model_sparse(graph)

    keras_model_gcn.compile(
        optimizer=Adam(lr=0.1), loss=categorical_crossentropy, weighted_metrics=["acc"]
    )

    keras_model_gcn_sp.compile(
        optimizer=Adam(lr=0.1), loss=categorical_crossentropy, weighted_metrics=["acc"]
    )

    weights = [
        np.array(
            [
                [0.43979216, -0.205199],
                [0.774606, 0.9521842],
                [-0.7586646, -0.41291213],
                [-0.80931616, 0.8148985],
            ],
            dtype="float32",
        ),
        np.array([0.0, 0.0], dtype="float32"),
        np.array([[1.0660936, -0.48291892], [1.2134176, 1.1863097]], dtype="float32"),
        np.array([0.0, 0.0], dtype="float32"),
    ]

    keras_model_gcn.set_weights(weights)
    keras_model_gcn_sp.set_weights(weights)

    ig_dense = IntegratedGradients(keras_model_gcn, train_gen)
    ig_sparse = IntegratedGradients(keras_model_gcn_sp, train_gen_sp)

    target_idx = 0
    class_of_interest = 0
    ig_node_importance_dense = ig_dense.get_node_importance(
        target_idx, class_of_interest, steps=50
    )
    ig_node_importance_sp = ig_sparse.get_node_importance(
        target_idx, class_of_interest, steps=50
    )

    ig_node_importance_ref = np.array([20.91, 18.29, 11.98, 5.98, 0])
    assert pytest.approx(ig_node_importance_dense, ig_node_importance_ref)
    assert pytest.approx(ig_node_importance_dense, ig_node_importance_sp)

    ig_link_importance_nz_ref = np.array(
        [0.2563, 0.2759, 0.2423, 0.0926, 0.1134, 0.0621, 0.0621, 0.0621]
    )

    ig_link_importance_dense = ig_dense.get_integrated_link_masks(
        target_idx, class_of_interest, adj_baseline=None, steps=50
    )
    ig_link_importance_dense_nz = ig_link_importance_dense[
        np.nonzero(ig_link_importance_dense)
    ]
    ig_link_importance_sp = ig_sparse.get_integrated_link_masks(
        target_idx, class_of_interest, adj_baseline=None, steps=50
    )
    ig_link_importance_sp_nz = ig_link_importance_sp[np.nonzero(ig_link_importance_sp)]

    assert pytest.approx(ig_link_importance_dense_nz, ig_link_importance_nz_ref)

    assert pytest.approx(ig_link_importance_dense_nz, ig_link_importance_sp_nz)


def test_saliency_init_parameters():
    graph = example_graph_1(feature_size=4)
    base_model, keras_model_gcn, generator, train_gen = create_GCN_model_dense(graph)
    (
        base_model_sp,
        keras_model_gcn_sp,
        generator_sp,
        train_gen_sp,
    ) = create_GCN_model_sparse(graph)

    keras_model_gcn.compile(
        optimizer=Adam(lr=0.1), loss=categorical_crossentropy, weighted_metrics=["acc"]
    )

    keras_model_gcn_sp.compile(
        optimizer=Adam(lr=0.1), loss=categorical_crossentropy, weighted_metrics=["acc"]
    )
    # Both TypeError and RuntimeError will be raised.
    # TypeError is raised due to the wrong generator type while RuntimeError is due to the wrong number of inputs for the model.
    with pytest.raises(TypeError) and pytest.raises(RuntimeError):
        IntegratedGradients(keras_model_gcn, train_gen_sp)

    with pytest.raises(TypeError) and pytest.raises(RuntimeError):
        IntegratedGradients(keras_model_gcn_sp, train_gen)
