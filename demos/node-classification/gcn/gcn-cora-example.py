# -*- coding: utf-8 -*-
#
# Copyright 2018-2019 Data61, CSIRO
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

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from tensorflow import keras
from tensorflow.keras import optimizers, losses, layers, metrics, regularizers
from sklearn import feature_extraction, model_selection

import stellargraph as sg
from stellargraph.layer import GCN
from stellargraph.mapper import FullBatchNodeGenerator


def train(
    train_nodes,
    train_targets,
    val_nodes,
    val_targets,
    generator,
    dropout,
    layer_sizes,
    learning_rate,
    activations,
    num_epochs,
):
    """

    Train a GCN model on the specified graph G with given parameters, evaluate it, and save the model.
    Args:
        train_nodes: A list of train nodes
        train_targets: Labels of train nodes
        val_nodes: A list of validation nodes
        val_targets: Labels of validation nodes
        generator: A FullBatchNodeGenerator
        dropout: The dropout (0->1) Initial Learning rate
        layer_sizes: A list of number of hidden nodes in each layer
        learning_rate: Initial Learning rate
        activations: A list of number of activation functions in each layer
    """

    train_gen = generator.flow(train_nodes, train_targets)
    val_gen = generator.flow(val_nodes, val_targets)
    gcnModel = GCN(
        layer_sizes,
        generator,
        bias=True,
        dropout=dropout,
        kernel_regularizer=regularizers.l2(5e-4),
        activations=activations,
    )

    # Expose the input and output sockets of the model:
    x_inp, x_out = gcnModel.node_model()

    # Create Keras model for training
    model = keras.Model(inputs=x_inp, outputs=x_out)
    model.compile(
        loss=losses.categorical_crossentropy,
        metrics=[metrics.categorical_accuracy],
        optimizer=optimizers.Adam(lr=learning_rate),
    )

    # Train model
    history = model.fit_generator(
        train_gen, epochs=num_epochs, validation_data=val_gen, verbose=2, shuffle=False
    )

    return model


def test(test_nodes, test_targets, generator, model_file):
    """

    Test a GCN model on the specified graph G with given parameters, evaluate it.
    Args:
        test_nodes: A list of test nodes
        test_targets: Labels of test nodes
        generator: A FullBatchNodeGenerator
        val_targets: Labels of validation nodes
        model_file: A path to the saved model file after training
    """

    test_gen = generator.flow(test_nodes, test_targets)

    model = keras.models.load_model(model_file, custom_objects=sg.custom_keras_layers)
    model.compile(
        loss=losses.categorical_crossentropy,
        metrics=[metrics.categorical_accuracy],
        optimizer=optimizers.Adam(lr=0.01),
    )
    print(model.summary())

    # Evaluate on test set and print metrics
    test_metrics = model.evaluate_generator(test_gen)

    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))


def main(graph_loc, layer_sizes, activations, dropout, learning_rate, num_epochs):
    # Load edges in order 'cited-paper' <- 'citing-paper'
    edgelist = pd.read_csv(
        os.path.join(graph_loc, "cora.cites"),
        sep="\t",
        header=None,
        names=["target", "source"],
    )

    # Load node features
    # The CORA dataset contains binary attributes 'w_x' that correspond to whether the corresponding keyword
    # (out of 1433 keywords) is found in the corresponding publication.
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    # Also, there is a "subject" column
    column_names = feature_names + ["subject"]
    node_data = pd.read_csv(
        os.path.join(graph_loc, "cora.content"),
        sep="\t",
        header=None,
        names=column_names,
    )

    target_encoding = feature_extraction.DictVectorizer(sparse=False)
    node_targets = target_encoding.fit_transform(
        node_data[["subject"]].to_dict("records")
    )

    node_ids = node_data.index
    node_features = node_data[feature_names]

    Gnx = nx.from_pandas_edgelist(edgelist)

    # Convert to StellarGraph and prepare for ML
    G = sg.StellarGraph(Gnx, node_features=node_features)

    # Split nodes into train/test using stratification.
    train_nodes, test_nodes, train_targets, test_targets = model_selection.train_test_split(
        node_ids,
        node_targets,
        train_size=140,
        test_size=None,
        stratify=node_targets,
        random_state=55232,
    )

    # Split test set into test and validation
    val_nodes, test_nodes, val_targets, test_targets = model_selection.train_test_split(
        test_nodes, test_targets, train_size=300, test_size=None, random_state=523214
    )

    # We specify the method='gcn' to give the pre-processing required by the GCN algorithm.
    generator = FullBatchNodeGenerator(G, method="gcn")

    model = train(
        train_nodes,
        train_targets,
        val_nodes,
        val_targets,
        generator,
        dropout,
        layer_sizes,
        learning_rate,
        activations,
        num_epochs,
    )

    # Save the trained model
    save_str = "_h{}_l{}_d{}_r{}".format(
        "gcn", "".join([str(x) for x in layer_sizes]), str(dropout), str(learning_rate)
    )

    model.save("cora_gcn_model" + save_str + ".h5")

    # We must also save the target encoding to convert model predictions
    with open("cora_gcn_encoding" + save_str + ".pkl", "wb") as f:
        pickle.dump([target_encoding], f)

    test(test_nodes, test_targets, generator, "cora_gcn_model" + save_str + ".h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph node classification using GCN")

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=50,
        help="The number of epochs to train the model",
    )
    parser.add_argument(
        "-d",
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate for the GCN model, between 0.0 and 1.0",
    )
    parser.add_argument(
        "-r",
        "--learningrate",
        type=float,
        default=0.005,
        help="Initial learning rate for model training",
    )
    parser.add_argument(
        "-s",
        "--layer_sizes",
        type=int,
        nargs="*",
        default=[32, 7],
        help="The number of hidden features at each GCN layer",
    )
    parser.add_argument(
        "-l",
        "--location",
        type=str,
        default=None,
        help="Location of the CORA dataset (directory)",
    )
    args, cmdline_args = parser.parse_known_args()

    # Load the dataset - this assumes it is the CORA dataset
    # Load graph edgelist
    if args.location is not None:
        graph_loc = os.path.expanduser(args.location)
    else:
        raise ValueError(
            "Please specify the directory containing the dataset using the '-l' flag"
        )

    activations = ["relu", "softmax"]
    main(
        graph_loc,
        args.layer_sizes,
        activations,
        args.dropout,
        args.learningrate,
        args.epochs,
    )
