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
Graph node classification using HinSAGE.
Requires the Yelp dataset in EPGM format as input.

Example usage:
python epgm-example.py -g ../../tests/resources/data/yelp/yelp.epgm -e 20

"""
import os
import argparse
import pickle

import keras
from keras import optimizers, losses, layers, metrics

from stellar.data.stellargraph import StellarGraph
from stellar.data.converter import *
from stellar.data.node_splitter import train_val_test_split
from stellar.data.loader import from_epgm
from stellar.layer.hinsage import HinSAGE, MeanHinAggregator
from stellar.mapper.node_mappers import HinSAGENodeMapper


def train(G, layer_size, num_samples, batch_size, num_epochs, learning_rate, dropout):
    """
    Train a HinSAGE model on the specified graph G with given parameters.

    Args:
        G: A NetworkX or StellarGraph with the Yelp dataset
        layer_size: A list of number of hidden nodes in each layer
        num_samples: Number of neighbours to sample at each layer
        batch_size: Size of batch for inference
        num_epochs: Number of epochs to train the model
        learning_rate: Initial Learning rate
        dropout: The dropout (0->1)
    """
    # Specify node attributes to use to create features
    user_attrs = [
        "cool",
        "useful",
        "funny",
        "fans",
        "average_stars",
        "compliment_cool",
        "compliment_hot",
        "compliment_more",
        "compliment_profile",
        "compliment_cute",
        "compliment_list",
        "compliment_note",
        "compliment_plain",
        "compliment_funny",
        "compliment_writer",
        "compliment_photos",
    ]

    review_attrs = ["stars", "useful"]

    business_attrs = [
        "BikeParking",
        "Alcohol",
        "RestaurantsPriceRange2",
        "BusinessAcceptsCreditCards",
        "WiFi",
        "DogsAllowed",
        "GoodForKids",
        "NoiseLevel",
    ]

    # Convert graph node attributes to feature vectors
    nfs = NodeAttributeSpecification()
    nfs.add_attribute_list("user", user_attrs, NumericConverter)
    nfs.add_attribute_list("review", review_attrs, NumericConverter)
    nfs.add_attribute("business", "stars", NumericConverter)
    nfs.add_attribute_list("business", business_attrs, OneHotCategoricalConverter)

    # Convert graph node attributes for target type to target vector
    target_node_type = "user"
    nts = NodeAttributeSpecification()
    nts.add_attribute_list(
        target_node_type, ["elite"], OneHotCategoricalConverter
    )

    # Reduce graph to only contain a subset of node types
    node_types_to_keep = ["review", "user", "business"]
    filtered_nodes = [
        n for n, ndata in G.nodes(data=True) if ndata["label"] in node_types_to_keep
    ]
    G = StellarGraph(G.subgraph(filtered_nodes))

    # Fit the graph to the attribute converters:
    G.fit_attribute_spec(feature_spec=nfs, target_spec=nts)

    # Split "user" nodes into train/test
    train_nodes, val_nodes, test_nodes, _ = train_val_test_split(
        G, node_type=target_node_type, train_size=0.25, test_size=0.5
    )

    # Get targets for the mapper
    train_targets = G.get_target_for_nodes(train_nodes)
    val_targets = G.get_target_for_nodes(val_nodes)

    # The mapper feeds data from sampled subgraph to GraphSAGE model
    train_mapper = HinSAGENodeMapper(
        G, train_nodes, batch_size, num_samples, targets=train_targets
    )
    val_mapper = HinSAGENodeMapper(
        G, val_nodes, batch_size, num_samples, targets=val_targets
    )

    # GraphSAGE model
    model = HinSAGE(layer_size, train_mapper, dropout=dropout)
    x_inp, x_out = model.default_model(flatten_output=True)

    # Final estimator layer
    prediction = layers.Dense(
        units=G.get_target_size(target_node_type), activation="softmax"
    )(x_out)

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
        shuffle=True
    )

    # Evaluate on test set and print metrics
    test_targets = G.get_target_for_nodes(test_nodes)
    test_mapper = HinSAGENodeMapper(
        G, test_nodes, batch_size, num_samples, targets=test_targets
    )
    test_metrics = model.evaluate_generator(test_mapper)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Evaluate over all nodes and print metrics
    user_nodes = G.get_nodes_of_type(target_node_type)
    all_targets = G.get_target_for_nodes(user_nodes)
    all_mapper = HinSAGENodeMapper(
        G, user_nodes, batch_size, num_samples, targets=all_targets
    )
    all_nodes_metrics = model.evaluate_generator(all_mapper)
    print("\nAll-node Evaluation:")
    for name, val in zip(model.metrics_names, all_nodes_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Save model
    save_str = "_n{}_l{}_d{}_r{}".format(
        "_".join([str(x) for x in num_samples]),
        "_".join([str(x) for x in layer_size]),
        dropout,
        learning_rate,
    )
    model.save("epgm_yelp_model" + save_str + ".h5")

    # We must also save the attribute spec, as it is fitted to the data
    with open("epgm_yelp_specs" + save_str + ".pkl", "wb") as f:
        pickle.dump([nfs, nts], f)


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
        default=[10, 10],
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

    print(G.info())

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
