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


import numpy as np
import pytest
import random
import tensorflow as tf
from stellargraph.data.unsupervised_sampler import UnsupervisedSampler
from stellargraph.mapper.sampled_node_generators import GraphSAGENodeGenerator
from stellargraph.mapper.sampled_link_generators import GraphSAGELinkGenerator
from stellargraph.layer.graphsage import GraphSAGE
from stellargraph.layer.link_inference import link_classification
from stellargraph.random import set_seed
from ..test_utils.graphs import petersen_graph
from .fixtures import assert_reproducible
from .. import require_gpu


def unsup_gs_model(num_samples, generator, optimizer, bias, dropout, normalize):
    layer_sizes = [50] * len(num_samples)
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes,
        generator=generator,
        bias=bias,
        dropout=dropout,
        normalize=normalize,
    )
    # Build the model and expose input and output sockets of graphsage, for node pair inputs:
    x_inp, x_out = graphsage.in_out_tensors()
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)
    model = tf.keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy)

    return model


def unsup_gs(
    g,
    num_samples,
    optimizer,
    batch_size=4,
    epochs=4,
    bias=True,
    dropout=0.0,
    normalize="l2",
    number_of_walks=1,
    walk_length=5,
    seed=0,
    shuffle=True,
):
    set_seed(seed)
    tf.random.set_seed(seed)
    if shuffle:
        random.seed(seed)

    nodes = list(g.nodes())
    unsupervised_samples = UnsupervisedSampler(
        g, nodes=nodes, length=walk_length, number_of_walks=number_of_walks
    )
    generator = GraphSAGELinkGenerator(g, batch_size, num_samples)
    train_gen = generator.flow(unsupervised_samples)

    model = unsup_gs_model(num_samples, generator, optimizer, bias, dropout, normalize)

    model.fit(
        train_gen,
        epochs=epochs,
        verbose=1,
        use_multiprocessing=False,
        workers=4,
        shuffle=shuffle,
    )
    return model


def gs_nai_model(num_samples, generator, targets, optimizer, bias, dropout, normalize):
    layer_sizes = [50] * len(num_samples)
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes,
        generator=generator,
        bias=bias,
        dropout=dropout,
        normalize=normalize,
    )
    # Build the model and expose input and output sockets of graphsage, for node pair inputs:
    x_inp, x_out = graphsage.in_out_tensors()
    pred = tf.keras.layers.Dense(units=targets.shape[1], activation="softmax")(x_out)
    model = tf.keras.Model(inputs=x_inp, outputs=pred)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy)

    return model


def gs_nai(
    g,
    targets,
    num_samples,
    optimizer,
    batch_size=4,
    epochs=4,
    bias=True,
    dropout=0.0,
    normalize="l2",
    seed=0,
    shuffle=True,
):
    set_seed(seed)
    tf.random.set_seed(seed)
    if shuffle:
        random.seed(seed)

    nodes = list(g.nodes())
    generator = GraphSAGENodeGenerator(g, batch_size, num_samples)
    train_gen = generator.flow(nodes, targets, shuffle=True)

    model = gs_nai_model(
        num_samples, generator, targets, optimizer, bias, dropout, normalize
    )

    model.fit(
        train_gen,
        epochs=epochs,
        verbose=1,
        use_multiprocessing=False,
        workers=4,
        shuffle=shuffle,
    )
    return model


def gs_link_pred_model(num_samples, generator, optimizer, bias, dropout, normalize):
    layer_sizes = [50] * len(num_samples)
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes,
        generator=generator,
        bias=bias,
        dropout=dropout,
        normalize=normalize,
    )
    # Build the model and expose input and output sockets of graphsage, for node pair inputs:
    x_inp, x_out = graphsage.in_out_tensors()
    pred = link_classification(
        output_dim=1, output_act="relu", edge_embedding_method="ip"
    )(x_out)

    model = tf.keras.Model(inputs=x_inp, outputs=pred)

    model.compile(
        optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy,
    )

    return model


def gs_link_prediction(
    g,
    edge_ids,
    edge_labels,
    num_samples,
    optimizer,
    batch_size=4,
    epochs=4,
    bias=True,
    dropout=0.0,
    normalize="l2",
    seed=0,
    shuffle=True,
):
    set_seed(seed)
    tf.random.set_seed(seed)
    if shuffle:
        random.seed(seed)

    generator = GraphSAGELinkGenerator(g, batch_size, num_samples)
    train_gen = generator.flow(edge_ids, edge_labels, shuffle=True)

    model = gs_link_pred_model(
        num_samples, generator, optimizer, bias, dropout, normalize
    )

    model.fit(
        train_gen,
        epochs=epochs,
        verbose=1,
        use_multiprocessing=False,
        workers=4,
        shuffle=shuffle,
    )
    return model


@pytest.mark.parametrize("shuffle", [True, False])
def test_unsupervised(petersen_graph, shuffle):
    assert_reproducible(
        lambda: unsup_gs(
            petersen_graph,
            [2, 2],
            tf.optimizers.Adam(1e-3),
            epochs=4,
            walk_length=2,
            batch_size=4,
            shuffle=shuffle,
        )
    )


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.skipif(require_gpu, reason="tf on GPU is non-deterministic")
def test_nai(petersen_graph, shuffle):
    target_size = 10
    targets = np.random.rand(len(petersen_graph.nodes()), target_size)
    assert_reproducible(
        lambda: gs_nai(
            petersen_graph, targets, [2, 2], tf.optimizers.Adam(1e-3), shuffle=shuffle
        )
    )


# FIXME (#970): This test fails intermittently with shuffle=True
@pytest.mark.parametrize("shuffle", [False])
def test_link_prediction(petersen_graph, shuffle):
    num_examples = 10
    edge_ids = np.random.choice(petersen_graph.nodes(), size=(num_examples, 2))
    edge_labels = np.random.choice([0, 1], size=num_examples)
    assert_reproducible(
        lambda: gs_link_prediction(
            petersen_graph,
            edge_ids,
            edge_labels,
            [2, 2],
            tf.optimizers.Adam(1e-3),
            shuffle=shuffle,
        )
    )
