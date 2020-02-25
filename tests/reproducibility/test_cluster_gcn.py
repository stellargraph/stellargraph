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

from stellargraph.mapper.mini_batch_node_generators import ClusterNodeGenerator
from stellargraph.layer.cluster_gcn import ClusterGCN
from stellargraph.random import set_seed
from .fixtures import assert_reproducible
from ..test_utils.graphs import petersen_graph
import numpy as np
import tensorflow as tf
import random
import pytest


def cluster_gcn_nai_model(generator, targets, optimizer, dropout):
    cluster_gcn = ClusterGCN(
        layer_sizes=[32, 32],
        activations=["relu", "relu"],
        generator=generator,
        dropout=dropout,
    )
    x_inp, x_out = cluster_gcn.build()
    pred = tf.keras.layers.Dense(units=targets.shape[1], activation="softmax")(x_out)
    model = tf.keras.Model(inputs=x_inp, outputs=pred)

    model.compile(optimizer=optimizer, loss=tf.losses.categorical_crossentropy)
    return model


def cluster_gcn_nai(
    graph,
    targets,
    num_clusters,
    clusters_per_batch,
    optimizer,
    epochs=4,
    dropout=0.0,
    seed=0,
    shuffle=True,
):
    set_seed(seed)
    tf.random.set_seed(seed)
    if shuffle:
        random.seed(seed)

    nodes = list(graph.nodes())
    generator = ClusterNodeGenerator(
        graph, clusters=num_clusters, q=clusters_per_batch, lam=0.1
    )
    model = cluster_gcn_nai_model(generator, targets, optimizer, dropout)
    train_gen = generator.flow(nodes, targets)
    model.fit_generator(
        train_gen, epochs=epochs, use_multiprocessing=False, workers=4, shuffle=shuffle
    )
    return model


@pytest.mark.parametrize("shuffle", [True, False])
def test_nai(petersen_graph, shuffle):
    target_size = 10
    targets = np.random.rand(petersen_graph.number_of_nodes(), target_size)
    assert_reproducible(
        lambda: cluster_gcn_nai(
            petersen_graph, targets, 4, 2, tf.optimizers.Adam(1e-3), shuffle=shuffle
        )
    )
