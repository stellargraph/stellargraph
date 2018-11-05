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
This currently is only tested on the CORA dataset, which can be downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

The following is the description of the dataset:
> The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
> The citation network consists of 5429 links. Each publication in the dataset is described by a
> 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
> The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download and unzip the cora.tgz file to a location on your computer and pass this location
(which should contain cora.cites and cora.content) as a command line argument to this script.

Run this script as follows:
    python graphsage-cora-example.py -l <path_to_cora_dataset>

Other optional arguments can be seen by running
    python graphsage-cora-example.py --help

"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import keras
from keras import optimizers, losses, layers, metrics
from sklearn import preprocessing, feature_extraction, model_selection
import stellargraph as sg
from stellargraph.layer import GraphSAGE, MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator
from stellargraph.mapper import GraphSAGENodeGenerator


def train(
    edgelist,
    node_data,
    layer_size,
    num_samples,
    batch_size=100,
    num_epochs=10,
    learning_rate=0.005,
    dropout=0.0,
    target_name="subject",
):
    """
    Train a GraphSAGE model on the specified graph G with given parameters, evaluate it, and save the model.

    Args:
        edgelist: Graph edgelist
        node_data: Feature and target data for nodes
        layer_size: A list of number of hidden nodes in each layer
        num_samples: Number of neighbours to sample at each layer
        batch_size: Size of batch for inference
        num_epochs: Number of epochs to train the model
        learning_rate: Initial Learning rate
        dropout: The dropout (0->1)
    """
    # Extract target and encode as a one-hot vector
    target_encoding = feature_extraction.DictVectorizer(sparse=False)
    node_targets = target_encoding.fit_transform(
        node_data[[target_name]].to_dict("records")
    )
    node_ids = node_data.index

    # Extract the feature data. These are the feature vectors that the Keras model will use as input.
    # The CORA dataset contains attributes 'w_x' that correspond to words found in that publication.
    node_features = node_data[feature_names]

    # Create graph from edgelist and set node features and node type
    Gnx = nx.from_pandas_edgelist(edgelist)

    # Convert to StellarGraph and prepare for ML
    G = sg.StellarGraph(Gnx, node_type_name="label", node_features=node_features)

    # Split nodes into train/test using stratification.
    train_nodes, test_nodes, train_targets, test_targets = model_selection.train_test_split(
        node_ids, node_targets, train_size=140, test_size=None, stratify=node_targets, random_state=55232
    )

    # Split test set into test and validation
    val_nodes, test_nodes, val_targets, test_targets = model_selection.train_test_split(
        test_nodes, test_targets, train_size=500, test_size=None, random_state=523214
    )

    # Create mappers for GraphSAGE that input data from the graph to the model
    generator = GraphSAGENodeGenerator(
        G, batch_size, num_samples, seed=42
    )
    train_gen = generator.flow(train_nodes, train_targets)
    val_gen = generator.flow(val_nodes, val_targets)

    # GraphSAGE model
    model = GraphSAGE(
        layer_sizes=layer_size, generator=train_gen, bias=True, dropout=dropout, aggregator=MeanAggregator
    )
    # Expose the input and output sockets of the model:
    x_inp, x_out = model.default_model(flatten_output=True)

    # Snap the final estimator layer to x_out
    prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

    # Create Keras model for training
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=learning_rate, decay=0.001),
        loss=losses.categorical_crossentropy,
        metrics=[metrics.categorical_accuracy],
    )
    print(model.summary())

    # Train model
    history = model.fit_generator(
        train_gen,
        epochs=num_epochs,
        validation_data=val_gen,
        verbose=2,
        shuffle=True,
    )

    # Evaluate on test set and print metrics
    test_metrics = model.evaluate_generator(generator.flow(test_nodes, test_targets))
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Get predictions for all nodes
    all_predictions = model.predict_generator(generator.flow(node_ids))

    # Turn predictions back into the original categories
    node_predictions = pd.DataFrame(
        target_encoding.inverse_transform(all_predictions), index=node_ids
    )
    accuracy = np.mean(
        [
            "subject=" + gt_subject == p
            for gt_subject, p in zip(
                node_data["subject"], node_predictions.idxmax(axis=1)
            )
        ]
    )
    print("All-node accuracy: {:3f}".format(accuracy))

    # TODO: extract the GraphSAGE embeddings from x_out, and save/plot them

    # Save the trained model
    save_str = "_n{}_l{}_d{}_r{}".format(
        "_".join([str(x) for x in num_samples]),
        "_".join([str(x) for x in layer_size]),
        dropout,
        learning_rate,
    )
    model.save("cora_example_model" + save_str + ".h5")

    # We must also save the target encoding to convert model predictions
    with open("cora_example_encoding" + save_str + ".pkl", "wb") as f:
        pickle.dump([target_encoding], f)


def test(edgelist, node_data, model_file, batch_size, target_name="subject"):
    """
    Load the serialized model and evaluate on all nodes in the graph.

    Args:
        G: NetworkX graph file
        target_converter: Class to give numeric representations of node targets
        feature_converter: CLass to give numeric representations of the node features
        model_file: Location of Keras model to load
        batch_size: Size of batch for inference
    """
    # Extract the feature data. These are the feature vectors that the Keras model will use as input.
    # The CORA dataset contains attributes 'w_x' that correspond to words found in that publication.
    node_features = node_data[feature_names]

    # Create graph from edgelist and set node features and node type
    Gnx = nx.from_pandas_edgelist(edgelist)

    # We must also save the target encoding to convert model predictions
    encoder_file = model_file.replace(
        "cora_example_model", "cora_example_encoding"
    ).replace(".h5", ".pkl")
    with open(encoder_file, "rb") as f:
        target_encoding = pickle.load(f)[0]

    # Endode targets with pre-trained encoder
    node_targets = target_encoding.transform(
        node_data[[target_name]].to_dict("records")
    )
    node_ids = node_data.index

    # Convert to StellarGraph and prepare for ML
    G = sg.StellarGraph(Gnx, node_features=node_features)

    # Load Keras model
    model = keras.models.load_model(
        model_file, custom_objects={"MeanAggregator": MeanAggregator}
    )
    print("Loaded model:")
    model.summary()

    # Get required samples from model
    # TODO: Can we move this to the library?
    num_samples = [
        int(model.input_shape[ii + 1][1] / model.input_shape[ii][1])
        for ii in range(len(model.input_shape) - 1)
    ]

    # Create mappers for GraphSAGE that input data from the graph to the model
    generator = GraphSAGENodeGenerator(
        G, batch_size, num_samples, seed=42
    )
    all_gen = generator.flow(node_ids, node_targets)

    # Evaluate and print metrics
    all_metrics = model.evaluate_generator(all_gen)

    print("\nAll-node Evaluation:")
    for name, val in zip(model.metrics_names, all_metrics):
        print("\t{}: {:0.4f}".format(name, val))


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
        help="Load a saved checkpoint .h5 file",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=20, help="Batch size for training"
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
        default=0.3,
        help="Dropout rate for the GraphSAGE model, between 0.0 and 1.0",
    )
    parser.add_argument(
        "-r",
        "--learningrate",
        type=float,
        default=0.005,
        help="Initial learning rate for model training",
    )
    parser.add_argument(
        "-n",
        "--neighbour_samples",
        type=int,
        nargs="*",
        default=[20, 10],
        help="The number of neighbour nodes sampled at each GraphSAGE layer",
    )
    parser.add_argument(
        "-s",
        "--layer_size",
        type=int,
        nargs="*",
        default=[20, 20],
        help="The number of hidden features at each GraphSAGE layer",
    )
    parser.add_argument(
        "-l",
        "--location",
        type=str,
        default=None,
        help="Location of the CORA dataset (directory)",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="subject",
        help="The target node attribute (categorical)",
    )
    args, cmdline_args = parser.parse_known_args()

    # Load the dataset - this assumes it is the CORA dataset
    # Load graph edgelist
    graph_loc = os.path.expanduser(args.location)
    edgelist = pd.read_table(
        os.path.join(graph_loc, "cora.cites"), header=None, names=["source", "target"]
    )

    # Load node features
    # The CORA dataset contains binary attributes 'w_x' that correspond to whether the corresponding keyword
    # (out of 1433 keywords) is found in the corresponding publication.
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    # Also, there is a "subject" column
    column_names = feature_names + ["subject"]
    node_data = pd.read_table(
        os.path.join(graph_loc, "cora.content"), header=None, names=column_names
    )

    if args.checkpoint is None:
        train(
            edgelist,
            node_data,
            args.layer_size,
            args.neighbour_samples,
            args.batch_size,
            args.epochs,
            args.learningrate,
            args.dropout,
        )
    else:
        test(edgelist, node_data, args.checkpoint, args.batch_size)
