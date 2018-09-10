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
Graph node classification using GraphSAGE.
Requires a EPGM graph as input.
This currently is only tested on the CORA dataset.

Example usage:
python epgm-example.py -g ../../tests/resources/data/cora/cora.epgm

"""
import os
import argparse
import pickle

import numpy as np
import pandas as pd

import keras
from keras import optimizers, losses, layers, metrics

from stellargraph.data.node_splitter import train_val_test_split
from stellargraph.data.loader import from_epgm
from stellargraph.data.converter import (
    NodeAttributeSpecification,
    OneHotCategoricalConverter,
    BinaryConverter,
)
from stellargraph.layer.graphsage import GraphSAGE, MeanAggregator
from stellargraph.mapper.node_mappers import GraphSAGENodeMapper


def train(
    G,
    layer_size,
    num_samples,
    batch_size=100,
    num_epochs=10,
    learning_rate=0.005,
    dropout=0.0,
):
    # Split nodes into train/test using stratification
    train_nodes, val_nodes, test_nodes, _ = train_val_test_split(
        G, train_size=140, test_size=1000, stratify=True
    )

    # Get targets for the mapper
    train_targets = G.get_target_for_nodes(train_nodes)
    val_targets = G.get_target_for_nodes(val_nodes)

    # Create mappers for GraphSAGE that input data from the graph to the model
    train_mapper = GraphSAGENodeMapper(
        G, train_nodes, batch_size, num_samples, targets=train_targets
    )
    val_mapper = GraphSAGENodeMapper(
        G, val_nodes, batch_size, num_samples, targets=val_targets
    )

    # GraphSAGE model
    model = GraphSAGE(
        layer_sizes=layer_size, mapper=train_mapper, bias=True, dropout=dropout
    )
    x_inp, x_out = model.default_model(flatten_output=True)

    # Final estimator layer
    prediction = layers.Dense(units=G.get_target_size("paper"), activation="softmax")(
        x_out
    )

    # Create Keras model for training
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=learning_rate),
        loss=losses.categorical_crossentropy,
        metrics=[metrics.categorical_accuracy],
    )

    # Train model
    history = model.fit_generator(
        train_mapper,
        epochs=num_epochs,
        validation_data=val_mapper,
        verbose=2,
        shuffle=True,
    )

    # Evaluate on test set and print metrics
    test_targets = G.get_target_for_nodes(test_nodes)
    test_mapper = GraphSAGENodeMapper(
        G, test_nodes, batch_size, num_samples, targets=test_targets
    )
    test_metrics = model.evaluate_generator(test_mapper)

    return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graph node classification using GraphSAGE"
    )
    parser.add_argument(
        "-g", "--graph", type=str, default=None, help="The graph stored in EPGM format."
    )
    args, cmdline_args = parser.parse_known_args()

    # Load graph
    graph_loc = os.path.expanduser(args.graph)
    G = from_epgm(graph_loc)

    # Convert node attribute to target values
    nts = NodeAttributeSpecification()
    nts.add_attribute("paper", "subject", OneHotCategoricalConverter)

    # Convert the rest of the node attributes to feature values
    nfs = NodeAttributeSpecification()
    nfs.add_all_attributes(
        G, "paper", BinaryConverter, ignored_attributes=["subject"]
    )

    # Learn feature and target conversion
    G.fit_attribute_spec(feature_spec=nfs, target_spec=nts)

    layer_size = [20, 20]
    neighbour_samples = [10, 10]
    batch_size = 20
    epochs = 50
    learningrate = 0.0025
    dropout = 0.25

    all_results = []
    for ii in range(10):
        tmetrics = train(
            G,
            layer_size,
            neighbour_samples,
            batch_size,
            epochs,
            learningrate,
            dropout,
        )

        all_results.append(tmetrics[1])

    print(all_results)
    print(np.mean(all_results))
    print(np.std(all_results))