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

"""
Graph node classification using Graph Attention Network (GAT) model.
This example uses the CORA dataset, which can be downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

The following is the description of the dataset:
> The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
> The citation network consists of 5429 links. Each publication in the dataset is described by a
> 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
> The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download and unzip the cora.tgz file to a location on your computer and pass this location
(which should contain cora.cites and cora.content) as a command line argument to this script.

Run this script as follows:
    python gat-cora-example.py -l <path_to_cora_dataset>

Other optional arguments can be seen by running
    python gat-cora-example.py --help

"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import keras
from keras import optimizers, losses, layers, metrics
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn import preprocessing, feature_extraction, model_selection
import stellargraph as sg
from stellargraph.layer import GAT
from stellargraph.mapper import FullBatchNodeGenerator


def train(
    edgelist,
    node_data,
    attn_heads,
    layer_sizes,
    num_epochs=10,
    learning_rate=0.005,
    es_patience=100,
    dropout=0.0,
    target_name="subject",
):
    """
    Train a GAT model on the specified graph G with given parameters, evaluate it, and save the model.

    Args:
        edgelist: Graph edgelist
        node_data: Feature and target data for nodes
        attn_heads: Number of attention heads in GAT layers
        layer_sizes: A list of number of hidden nodes in each layer
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
        node_ids,
        node_targets,
        train_size=140,
        test_size=None,
        stratify=node_targets,
        random_state=55232,
    )

    # Further split test set into validation and test
    val_nodes, test_nodes, val_targets, test_targets = model_selection.train_test_split(
        test_nodes, test_targets, train_size=500, test_size=1000, random_state=523214
    )

    # Create mappers for GraphSAGE that input data from the graph to the model
    generator = FullBatchNodeGenerator(G, method="gat")
    train_gen = generator.flow(train_nodes, train_targets)
    val_gen = generator.flow(val_nodes, val_targets)

    # GAT model
    gat = GAT(
        layer_sizes=layer_sizes,
        attn_heads=attn_heads,
        generator=generator,
        bias=True,
        in_dropout=dropout,
        attn_dropout=dropout,
        activations=["elu", "elu"],
        normalize=None,
    )
    # Expose the input and output tensors of the GAT model for nodes:
    x_inp, x_out = gat.node_model()

    # Snap the final estimator layer to x_out
    x_out = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

    # Create Keras model for training
    model = keras.Model(inputs=x_inp, outputs=x_out)
    model.compile(
        optimizer=optimizers.Adam(lr=learning_rate, decay=0.001),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )
    print(model.summary())

    # Train model
    # Callbacks
    if not os.path.isdir("logs"):
        os.makedirs("logs")
    N = len(node_ids)
    es_callback = EarlyStopping(monitor="val_acc", patience=es_patience)
    tb_callback = TensorBoard(batch_size=N)
    mc_callback = ModelCheckpoint(
        "logs/best_model.h5",
        monitor="val_acc",
        save_best_only=True,
        save_weights_only=True,
    )

    if args.interface == "fit":
        print("\nUsing model.fit() to train the model\n")
        # Get the training data
        inputs_train, y_train = train_gen[0]

        # Get the validation data
        inputs_val, y_val = val_gen[0]

        history = model.fit(
            x=inputs_train,
            y=y_train,
            batch_size=N,
            shuffle=False,  # must be False, since shuffling data means shuffling the whole graph
            epochs=num_epochs,
            verbose=2,
            validation_data=(inputs_val, y_val),
            callbacks=[es_callback, tb_callback, mc_callback],
        )
    else:
        print("\nUsing model.fit_generator() to train the model\n")
        history = model.fit_generator(
            train_gen,
            epochs=num_epochs,
            validation_data=val_gen,
            verbose=2,
            shuffle=False,
            callbacks=[es_callback, tb_callback, mc_callback],
        )

    # Load best model
    model.load_weights("logs/best_model.h5")

    # Evaluate on validation set and print metrics
    if args.interface == "fit":
        val_metrics = model.evaluate(x=inputs_val, y=y_val)
    else:
        val_metrics = model.evaluate_generator(val_gen)

    print("\nBest model's Validation Set Metrics:")
    for name, val in zip(model.metrics_names, val_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Evaluate on test set and print metrics
    if args.interface == "fit":
        inputs_test, y_test = generator.flow(test_nodes, test_targets)[0]
        test_metrics = model.evaluate(x=inputs_test, y=y_test)
    else:
        test_metrics = model.evaluate_generator(
            generator.flow(test_nodes, test_targets)
        )

    print("\nBest model's Test Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Get predictions for all nodes
    all_predictions = model.predict_generator(generator.flow(node_ids))

    # Remove singleton batch dimension
    all_predictions = np.squeeze(all_predictions)

    # Turn predictions back into the original categories
    node_predictions = pd.DataFrame(
        target_encoding.inverse_transform(all_predictions), index=list(G.nodes())
    )
    accuracy = np.mean(
        [
            "subject=" + gt_subject == p
            for gt_subject, p in zip(
                node_data["subject"], node_predictions.idxmax(axis=1)
            )
        ]
    )
    print("\nAll-node accuracy: {:0.4f}".format(accuracy))

    # Save the trained model
    save_str = "_h{}_l{}_d{}_r{}".format(
        attn_heads, "_".join([str(x) for x in layer_sizes]), dropout, learning_rate
    )
    model.save("cora_gat_model" + save_str + ".h5")

    # We must also save the target encoding to convert model predictions
    with open("cora_gat_encoding" + save_str + ".pkl", "wb") as f:
        pickle.dump([target_encoding], f)


def test():
    raise NotImplemented


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graph node classification using Graph Attention Network (GAT)"
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
        default=0.5,
        help="Dropout rate for the GAT model, between 0.0 and 1.0",
    )
    parser.add_argument(
        "-r",
        "--learningrate",
        type=float,
        default=0.005,
        help="Initial learning rate for model training",
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping (number of epochs with no improvement after which training should be stopped)",
    )
    parser.add_argument(
        "-a",
        "--attn_heads",
        type=int,
        nargs="*",
        default=[8, 1],
        help="Number of attention heads",
    )
    parser.add_argument(
        "-s",
        "--layer_sizes",
        type=int,
        nargs="*",
        default=[8, 8],
        help="The number of hidden features at each GAT layer",
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
    parser.add_argument(
        "-i",
        "--interface",
        type=str,
        default="fit_generator",
        help="Defines which method is used for model training (.fit() or .fit_generator())",
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

    edgelist = pd.read_csv(
        os.path.join(graph_loc, "cora.cites"),
        sep="\t",
        header=None,
        names=["source", "target"],
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

    if args.checkpoint is None:
        train(
            edgelist,
            node_data,
            args.attn_heads,
            args.layer_sizes,
            args.epochs,
            args.learningrate,
            args.patience,
            args.dropout,
        )
    else:
        test()
