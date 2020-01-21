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
import tensorflow as tf
import random
from stellargraph.data.unsupervised_sampler import UnsupervisedSampler
from stellargraph.mapper.sampled_link_generators import GraphSAGELinkGenerator
from stellargraph.layer.graphsage import GraphSAGE
from stellargraph.layer.link_inference import link_classification
from stellargraph.utils.random import set_seed
from ..test_utils.graphs import petersen_graph
from .fixtures import assert_reproducible


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
    x_inp, x_out = graphsage.build()
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
):
    set_seed(seed)

    nodes = list(g.nodes())
    unsupervised_samples = UnsupervisedSampler(
        g, nodes=nodes, length=walk_length, number_of_walks=number_of_walks
    )
    generator = GraphSAGELinkGenerator(g, batch_size, num_samples)
    train_gen = generator.flow(unsupervised_samples)

    model = unsup_gs_model(num_samples, generator, optimizer, bias, dropout, normalize)

    model.fit_generator(
        train_gen,
        epochs=epochs,
        verbose=1,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
    )
    return model


def test_reproducibility(petersen_graph):
    assert_reproducible(
        lambda: unsup_gs(
            petersen_graph,
            [2],
            tf.optimizers.Adam(1e-3),
            epochs=4,
            walk_length=2,
            batch_size=4,
        )
    )
