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

usage: epgm-example.py [-h] [-c [CHECKPOINT]] [-n BATCH_SIZE] [-e EPOCHS]
                       [-s [NEIGHBOUR_SAMPLES [NEIGHBOUR_SAMPLES ...]]]
                       [-l [LAYER_SIZE [LAYER_SIZE ...]]] [-g GRAPH]
                       [-f FEATURES] [-t TARGET]

optional arguments:
  -h, --help            show this help message and exit
  -c [CHECKPOINT], --checkpoint [CHECKPOINT]
                        Load a saved checkpoint file
  -n BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs to train the model
  -d DROPOUT, --dropout DROPOUT
                        Dropout for the GraphSAGE model, between 0.0 and 1.0
  -r LEARNINGRATE, --learningrate LEARNINGRATE
                        Learning rate for training model
  -s [NEIGHBOUR_SAMPLES [NEIGHBOUR_SAMPLES ...]], --neighbour_samples [NEIGHBOUR_SAMPLES [NEIGHBOUR_SAMPLES ...]]
                        The number of nodes sampled at each layer
  -l [LAYER_SIZE [LAYER_SIZE ...]], --layer_size [LAYER_SIZE [LAYER_SIZE ...]]
                        The number of hidden features at each layer
  -g GRAPH, --graph GRAPH
                        The graph stored in EPGM format.
  -f FEATURES, --features FEATURES
                        The node features to use, stored as a pickled numpy
                        array.
  -t TARGET, --target TARGET
                        The target node attribute (categorical)
"""
import os
import argparse
import numpy as np
import pandas as pd

import keras
from keras import optimizers, losses, layers, metrics

from stellar.data.node_splitter import train_val_test_split
from stellar.data.loader import from_epgm
from stellar.data.converter import (
    NodeAttributeSpecification,
    OneHotCategoricalConverter,
    BinaryConverter,
)
from stellar.layer.graphsage import GraphSAGE, MeanAggregator
from stellar.mapper.node_mappers import GraphSAGENodeMapper


def train(
    G,
    layer_size,
    num_samples,
    batch_size=100,
    num_epochs=10,
    learning_rate=0.005,
    dropout=0.0,
):
    """
    Train the GraphSAGE model on the specified graph G
    with given parameters.

    Args:
        G: NetworkX graph file
        target_converter: Class to give numeric representations of node targets
        feature_converter: CLass to give numeric representations of the node features
        layer_size: A list of number of hidden nodes in each layer
        num_samples: Number of neighbours to sample at each layer
        batch_size: Size of batch for inference
        num_epochs: Number of epochs to train the model
        learning_rate: Initial Learning rate
        dropout: The dropout (0->1)
    """
    # Convert node attribute to target values
    nts = NodeAttributeSpecification()
    nts.add_attribute("paper", args.target, OneHotCategoricalConverter)

    # Convert the rest of the node attributes to feature values
    nfs = NodeAttributeSpecification()
    nfs.add_all_attributes(
        G, "paper", BinaryConverter, ignored_attributes=[args.target]
    )

    # Learn feature and target conversion
    G.fit_attribute_spec(feature_spec=nfs, target_spec=nts)

    # Split "user" nodes into train/test
    train_nodes, val_nodes, test_nodes, _ = train_val_test_split(
        G, train_size=160, test_size=1000, stratify=True
    )

    # Create mappers for GraphSAGE that input data from the graph to the model
    train_mapper = GraphSAGENodeMapper(G, train_nodes, batch_size, num_samples)
    val_mapper = GraphSAGENodeMapper(G, val_nodes, batch_size, num_samples)
    test_mapper = GraphSAGENodeMapper(G, test_nodes, batch_size, num_samples)
    all_mapper = GraphSAGENodeMapper(G, list(G), batch_size, num_samples)

    # GraphSAGE model
    model = GraphSAGE(
        output_dims=layer_size,
        n_samples=num_samples,
        input_dim=G.get_feature_size("paper"),
        bias=True,
        dropout=dropout,
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

    # Evaluate and print metrics
    test_metrics = model.evaluate_generator(test_mapper)

    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    all_nodes_metrics = model.evaluate_generator(all_mapper)
    print("\nAll-node Evaluation:")
    for name, val in zip(model.metrics_names, all_nodes_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # TODO: extract the GraphSAGE embeddings from x_out, and save/plot them

    # Save model
    str_numsamp = "_".join([str(x) for x in num_samples])
    str_layer = "_".join([str(x) for x in layer_size])
    model.save(
        "graphsage_n{}_l{}_d{}_r{}.h5".format(
            str_numsamp, str_layer, dropout, learning_rate
        )
    )


def test(G, model_file, batch_size):
    """
    Load the serialized model and evaluate on all nodes in the graph.

    Args:
        G: NetworkX graph file
        target_converter: Class to give numeric representations of node targets
        feature_converter: CLass to give numeric representations of the node features
        model_file: Location of Keras model to load
        batch_size: Size of batch for inference
    """
    #TODO: This needs to be written
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graph node classification using GraphSAGE"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        nargs="?",
        type=str,
        default=None,
        help="Load a saved checkpoint file",
    )
    parser.add_argument(
        "-n", "--batch_size", type=int, default=20, help="Batch size for training"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs to train the model",
    )
    parser.add_argument(
        "-d",
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout for the GraphSAGE model, between 0.0 and 1.0",
    )
    parser.add_argument(
        "-r",
        "--learningrate",
        type=float,
        default=0.01,
        help="Learning rate for training model",
    )
    parser.add_argument(
        "-s",
        "--neighbour_samples",
        type=int,
        nargs="*",
        default=[20, 10],
        help="The number of nodes sampled at each layer",
    )
    parser.add_argument(
        "-l",
        "--layer_size",
        type=int,
        nargs="*",
        default=[20, 20],
        help="The number of hidden features at each layer",
    )
    parser.add_argument(
        "-g", "--graph", type=str, default=None, help="The graph stored in EPGM format."
    )
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        default=None,
        help="The node features to use, stored as a pickled numpy array.",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="subject",
        help="The target node attribute (categorical)",
    )
    args, cmdline_args = parser.parse_known_args()

    # Load graph
    graph_loc = os.path.expanduser(args.graph)
    G = from_epgm(graph_loc)

    if args.checkpoint is None:
        train(
            G,
            args.layer_size,
            args.neighbour_samples,
            args.batch_size,
            args.epochs,
            args.learningrate,
            args.dropout,
        )
    else:
        test(G, args.checkpoint, args.batch_size)
