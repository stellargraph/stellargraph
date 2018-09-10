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
Example of heterogeneous graph node classification using HinSAGE.

Requires the preprocessed Yelp dataset .. see the script `yelp_preprocessing.py`
for more details. We assume that the preprocessing script has been run.

Example usage:
python yelp-example.py -l <location_of_preprocessed_data>

Additional command line arguments are available to tune the learned model, to see a
description of these arguments use the `--help` argument:
python yelp-example.py --help

"""
import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import keras
from keras import optimizers, losses, layers, metrics
import keras.backend as K

from stellargraph.data.stellargraph import StellarGraph
from stellargraph.layer.hinsage import HinSAGE, MeanHinAggregator
from stellargraph.mapper.node_mappers import HinSAGENodeMapper

from sklearn import model_selection
from sklearn import metrics as sk_metrics


def weighted_binary_crossentropy(weights):
    """
    Weighted binary cross-entropy loss
    Args:
        weights: A list or numpy array of weights per class

    Returns:
        A Keras loss function
    """
    weights = np.asanyarray(weights, dtype="float32")

    def loss_fn(y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred) * weights, axis=-1)

    return loss_fn


def train(
    Gnx,
    user_targets,
    layer_size,
    num_samples,
    batch_size,
    num_epochs,
    learning_rate,
    dropout,
):
    """
    Train a HinSAGE model on the specified graph G with given parameters.

    Args:
        Gnx: A NetworkX with the Yelp dataset
        layer_size: A list of number of hidden nodes in each layer
        num_samples: Number of neighbours to sample at each layer
        batch_size: Size of batch for inference
        num_epochs: Number of epochs to train the model
        learning_rate: Initial Learning rate
        dropout: The dropout (0->1)
    """
    # Create stellar Graph object
    G = StellarGraph(Gnx, node_type_name="ntype", edge_type_name="etype")

    print(G.info())

    # Fit the graph to the attribute converters:
    G.fit_attribute_spec()

    # Split "user" nodes into train/test
    # Split nodes into train/test using stratification.
    train_targets, test_targets = model_selection.train_test_split(
        user_targets, train_size=0.25, test_size=None
    )

    # The mapper feeds data from sampled subgraph to GraphSAGE model
    # Train mapper
    train_mapper = HinSAGENodeMapper(
        G, train_targets.index, batch_size, num_samples, targets=train_targets.values
    )
    # Test mapper
    test_mapper = HinSAGENodeMapper(
        G, test_targets.index, batch_size, num_samples, targets=test_targets.values
    )

    # GraphSAGE model
    model = HinSAGE(layer_size, train_mapper, dropout=dropout)
    x_inp, x_out = model.default_model(flatten_output=True)

    # Final estimator layer
    prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

    # Calculate weights based on empirical count
    class_count = train_targets.values.sum(axis=0)
    weights = class_count.sum() / class_count

    print("Weighting loss by: {}".format(weights))

    # Create Keras model for training
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=learning_rate),
        loss=weighted_binary_crossentropy(weights),
        metrics=[metrics.binary_accuracy],
    )

    # Train model
    history = model.fit_generator(
        train_mapper, epochs=num_epochs, verbose=2, shuffle=True
    )

    # Evaluate on test set and print metrics
    predictions = model.predict_generator(test_mapper)
    binary_predictions = predictions[:, 1] > 0.5
    print("\nTest Set Metrics of trained model (on {} nodes)".format(len(predictions)))

    # Calculate metrics using Scikit-Learn
    cm = sk_metrics.confusion_matrix(test_targets.iloc[:, 1], binary_predictions)
    print("Confusion matrix:")
    print(cm)

    accuracy = sk_metrics.accuracy_score(test_targets.iloc[:, 1], binary_predictions)
    precision = sk_metrics.precision_score(test_targets.iloc[:, 1], binary_predictions)
    recall = sk_metrics.recall_score(test_targets.iloc[:, 1], binary_predictions)
    f1 = sk_metrics.f1_score(test_targets.iloc[:, 1], binary_predictions)
    roc_auc = sk_metrics.roc_auc_score(test_targets.iloc[:, 1], binary_predictions)

    print(
        "accuracy = {:0.3}, precision = {:0.3}, recall = {:0.3}, f1 = {:0.3}".format(
            accuracy, precision, recall, f1
        )
    )
    print("ROC AUC = {:0.3}".format(roc_auc))

    # Save model
    save_str = "_n{}_l{}_d{}_r{}".format(
        "_".join([str(x) for x in num_samples]),
        "_".join([str(x) for x in layer_size]),
        dropout,
        learning_rate,
    )
    model.save("yelp_model" + save_str + ".h5")


def test(G, target_converter, feature_converter, model_file, batch_size, target_attr):
    """
    Load the serialized model and evaluate on all nodes in the graph.

    Args:
        G: NetworkX graph file
        target_converter: Class to give numeric representations of node targets
        feature_converter: CLass to give numeric representations of the node features
        model_file: Location of Keras model to load
        batch_size: Size of batch for inference
    """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graph node classification using GraphSAGE"
    )
    parser.add_argument(
        "-l",
        "--location",
        type=str,
        default=None,
        help="The location of the pre-processes Yelp dataset.",
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
        "-b", "--batch_size", type=int, default=200, help="Batch size for training"
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
        default=0.001,
        help="Learning rate for training model",
    )
    parser.add_argument(
        "-n",
        "--neighbour_samples",
        type=int,
        nargs="*",
        default=[10, 5],
        help="The number of nodes sampled at each layer",
    )
    parser.add_argument(
        "-s",
        "--layer_size",
        type=int,
        nargs="*",
        default=[80, 80],
        help="The number of hidden features at each layer",
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
    args = parser.parse_args()

    # Load graph and data
    data_loc = os.path.expanduser(args.location)

    # Read the data
    user_features = pd.read_pickle(os.path.join(data_loc, "user_features_filtered.pkl"))
    business_features = pd.read_pickle(
        os.path.join(data_loc, "business_features_filtered.pkl")
    )
    user_targets = pd.read_pickle(os.path.join(data_loc, "user_targets_filtered.pkl"))

    # Load graph
    Gnx = nx.read_graphml(os.path.join(data_loc, "yelp_graph_filtered.graphml"))

    # Adding node features to graph
    for user_id, row in user_features.iterrows():
        if user_id not in Gnx:
            print("User not found")
        Gnx.node[user_id]["feature"] = row.values

    for business_id, row in business_features.iterrows():
        if business_id not in Gnx:
            print("Business not found")
        Gnx.node[business_id]["feature"] = row.values

    if args.checkpoint is None:
        train(
            Gnx,
            user_targets,
            args.layer_size,
            args.neighbour_samples,
            args.batch_size,
            args.epochs,
            args.learningrate,
            args.dropout,
        )
    else:
        test(Gnx, args.checkpoint, args.batch_size)
