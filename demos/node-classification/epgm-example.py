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
import networkx as nx
from stellargraph.data.converter import (
    NodeAttributeSpecification,
    OneHotCategoricalConverter,
    BinaryConverter,
)
from stellargraph.layer.graphsage import GraphSAGE, MeanAggregator
from stellargraph.mapper.node_mappers import GraphSAGENodeMapper
from stellargraph.data.stellargraph import StellarGraph
from stellargraph import globals
from sklearn import preprocessing, feature_extraction, model_selection

def load_data(data_dir):
    """
    Load the cora dataset into a graph
    Args:
        graph_loc: path to the dataset

    Returns:

    """
    # Load the edgelist:
    edgelist = pd.read_table(os.path.join(data_dir, "cora.cites"), header=None, names=["source", "target"])

    # Create a graph:
    g = nx.from_pandas_edgelist(edgelist)

    # Load the features and subject for the nodes
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    column_names = feature_names + ["subject"]
    node_data = pd.read_table(os.path.join(data_dir, "cora.content"), header=None, names=column_names)

    return g, node_data


class NodeClassification(object):
    """
    Node classification class
    """

    def __init__(self, g, node_data):
        """

        Args:
            g: networkx graph
            node_data: pandas dataframe with node data
        """
        self.g = g
        self.node_data = node_data


    def train(
        self,
        layer_size,
        num_samples,
        batch_size=100,
        num_epochs=10,
        learning_rate=0.005,
        dropout=0.0,
    ):
        """
        Train a GraphSAGE model on the graph self.g with given parameters.

        Args:
            layer_size: A list of number of hidden nodes in each layer
            num_samples: Number of neighbours to sample at each layer
            batch_size: Size of batch for inference
            num_epochs: Number of epochs to train the model
            learning_rate: Initial Learning rate
            dropout: The dropout (0->1)
        """

        # Split nodes into train, validation, and test sets:
        train_data, remaining_data = model_selection.train_test_split(self.node_data, train_size=140, test_size=None,
                                                                 stratify=self.node_data['subject'])
        val_data, test_data = model_selection.train_test_split(remaining_data, train_size=None, test_size=0.7)

        # Convert node target and features to numeric values:
        # Categorical target:
        target_encoding = feature_extraction.DictVectorizer(sparse=False)
        train_targets = target_encoding.fit_transform(train_data[[args.target]].to_dict('records'))
        val_targets = target_encoding.transform(val_data[[args.target]].to_dict('records'))
        test_targets = target_encoding.transform(test_data[[args.target]].to_dict('records'))

        # One-hot features:
        feature_names = [n for n in self.node_data.columns if n != args.target]
        node_features = self.node_data[feature_names].values

        # Put features and targets into the graph as node attributes:
        # Features and node types:
        for nid, f in zip(self.node_data.index, node_features):
            self.g.node[nid][globals.FEATURE_ATTR_NAME] = f
            self.g.node[nid][globals.TYPE_ATTR_NAME] = "paper"

        # Targets:
        all_targets = target_encoding.transform(self.node_data[[args.target]].to_dict('records'))
        for nid, t in zip(self.node_data.index, all_targets):
            self.g.node[nid]["target"] = t

        # Convert self.g into StellarGraph:
        self.g = StellarGraph(self.g)
        # Prepare it for ML
        self.g.fit_attribute_spec()
        print(self.g.info())

        # Create mappers for GraphSAGE that input data from the graph to the model
        train_nodes = train_data.index
        val_nodes = val_data.index
        test_nodes = test_data.index

        train_mapper = GraphSAGENodeMapper(
            self.g, train_nodes, batch_size, num_samples, targets=train_targets
        )
        val_mapper = GraphSAGENodeMapper(
            self.g, val_nodes, batch_size, num_samples, targets=val_targets
        )

        # GraphSAGE model
        model = GraphSAGE(
            layer_sizes=layer_size, mapper=train_mapper, bias=True, dropout=dropout
        )
        x_inp, x_out = model.default_model(flatten_output=True)

        # Final estimator layer
        prediction = layers.Dense(units=self.g.get_target_size("paper"), activation="softmax")(
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
        test_mapper = GraphSAGENodeMapper(
            self.g, test_nodes, batch_size, num_samples, targets=test_targets
        )

        test_metrics = model.evaluate_generator(test_mapper)
        print("\nTest Set Metrics:")
        for name, val in zip(model.metrics_names, test_metrics):
            print("\t{}: {:0.4f}".format(name, val))

        # Get predictions for all nodes
        all_nodes = list(self.g)
        all_mapper = GraphSAGENodeMapper(self.g, all_nodes, batch_size, num_samples)
        all_predictions = model.predict_generator(all_mapper)

        # Turn predictions back into the original categories
        node_predictions = target_encoding.inverse_transform(all_predictions)
        accuracy = np.mean(
            [
                self.g.node[n]["subject"] == p["subject"]
                for n, p in zip(all_nodes, node_predictions)
            ]
        )
        print("All-node accuracy: {:3f}".format(accuracy))

        # TODO: extract the GraphSAGE embeddings from x_out, and save/plot them

        # Save model
        save_str = "_n{}_l{}_d{}_r{}".format(
            "_".join([str(x) for x in num_samples]),
            "_".join([str(x) for x in layer_size]),
            dropout,
            learning_rate,
        )
        model.save("epgm_example_model" + save_str + ".h5")

        # We must also save the attribute spec, as it is fitted to the data
        with open("epgm_example_specs" + save_str + ".pkl", "wb") as f:
            pickle.dump([nfs, nts], f)


    def test(self, model_file, batch_size):
        """
        Load the serialized model and evaluate on all nodes in the graph.

        Args:
            target_converter: Class to give numeric representations of node targets
            feature_converter: CLass to give numeric representations of the node features
            model_file: Location of Keras model to load
            batch_size: Size of batch for inference
        """
        # TODO: This needs to be written
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
        default=0.005,
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
        "-p", "--data_path", type=str, default="../data/cora", help="Dataset path (directory)"
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
    data_path = os.path.expanduser(args.data_path)
    Gnx, node_data = load_data(data_path)

    classifier = NodeClassification(Gnx, node_data)

    if args.checkpoint is None:
        classifier.train(
            args.layer_size,
            args.neighbour_samples,
            args.batch_size,
            args.epochs,
            args.learningrate,
            args.dropout,
        )
    else:
        classifier.test(args.checkpoint, args.batch_size)
