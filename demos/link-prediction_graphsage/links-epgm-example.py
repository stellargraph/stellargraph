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
Graph link prediction using GraphSAGE.
Requires a EPGM graph as input.
This currently is only tested on the CORA dataset.

Example usage:
python links-epgm-example.py -g ../../tests/resources/data/cora/cora.epgm -e 10 -l 64 32 -b 10 -s 10 20 -r 0.001 -d 0.5 --ignore_node_attr subject --edge_sampling_method local --edge_feature_method concat

usage: epgm-example.py [-h] [-c [CHECKPOINT]] [-e EPOCHS] [-b BATCH_SIZE]
                       [-s [NEIGHBOUR_SAMPLES [NEIGHBOUR_SAMPLES ...]]]
                       [-l [LAYER_SIZE [LAYER_SIZE ...]]] [-g GRAPH]
                       [-r LEARNING_RATE] [-d DROPOUT]
                       [-i [IGNORE_NODE_ATTR]]
                       [--edge_sampling_method] [--edge_feature_method]

optional arguments:
  -h, --help            show this help message and exit
  -c [CHECKPOINT], --checkpoint [CHECKPOINT]
                        Load a saved checkpoint file
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training/validation/testing
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs to train the model
  -d DROPOUT, --dropout DROPOUT
                        Dropout for the GraphSAGE model, between 0.0 and 1.0
  -r LEARNING_RATE, --learningrate LEARNING_RATE
                        Learning rate for training model
  -s [NEIGHBOUR_SAMPLES [NEIGHBOUR_SAMPLES ...]], --neighbour_samples [NEIGHBOUR_SAMPLES [NEIGHBOUR_SAMPLES ...]]
                        The number of nodes sampled at each layer
  -l [LAYER_SIZE [LAYER_SIZE ...]], --layer_size [LAYER_SIZE [LAYER_SIZE ...]]
                        The number of hidden features at each layer
  -g GRAPH, --graph GRAPH
                        The graph stored in EPGM format.
  -i IGNORE_NODE_ATTR, --ignore_node_attr FEATURES
                        List of node attributes to ignore (e.g., names, ids, etc.)
  --edge_sampling_method
        method for sampling negative links, either 'local' or 'global'
        'local': negative links are sampled to have destination nodes that are in the local neighbourhood of a source node,
                i.e., 2-k hops away, where k is the maximum number of hops specified by --edge_sampling_probs argument
        'global': negative links are sampled randomly from all negative links in the graph
  --edge_sampling_probs
        probabilities for sampling negative links.
        Must start with 0 (no negative links to 1-hop neighbours, as these are positive by definition)
  --edge_feature_method
        Method for combining (src, dst) node embeddings into edge embeddings.
        Can be one of 'ip' (inner product), 'mul' (element-wise multiplication), and 'concat' (concatenation)
"""
import os
import argparse
import networkx as nx
from typing import AnyStr, List

import keras
from keras import optimizers, losses, metrics

from stellar.data.loader import from_epgm
from stellar.data.edge_splitter import EdgeSplitter
from stellar.data.converter import (
    NodeAttributeSpecification,
    BinaryConverter,
)

from stellar.layer.graphsage import GraphSAGE, MeanAggregator
from stellar.mapper.link_mappers import GraphSAGELinkMapper
from stellar.layer.link_inference import link_classification
from stellar.data.stellargraph import *


def train(
    G,
    layer_size: List[int],
    num_samples: List[int],
    batch_size: int = 100,
    num_epochs: int = 10,
    learning_rate: float = 0.005,
    dropout: float = 0.0,
):
    """
    Train the GraphSAGE model on the specified graph G
    with given parameters.

    Args:
        G: NetworkX graph file
        layer_size: A list of number of hidden units in each layer of the GraphSAGE model
        num_samples: Number of neighbours to sample at each layer of the GraphSAGE model
        batch_size: Size of batch for inference
        num_epochs: Number of epochs to train the model
        learning_rate: Initial Learning rate
        dropout: The dropout (0->1)
    """

    # Split links into train/test
    print(
        "Using '{}' method to sample negative links".format(args.edge_sampling_method)
    )

    # From the original graph, extract E_test and the reduced graph G_test:
    edge_splitter_test = EdgeSplitter(G)
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method=args.edge_sampling_method, probs=args.edge_sampling_probs
    )

    # From G_test, extract E_train and the reduced graph G_train:
    edge_splitter_train = EdgeSplitter(G_test, G)
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.1, method=args.edge_sampling_method, probs=args.edge_sampling_probs
    )

    # Convert G_train and G_test to StellarGraph objects for ML:
    if G_train.is_directed():
        G_train = StellarDiGraph(G_train)
    else:
        G_train = StellarGraph(G_train)

    if G_test.is_directed():
        G_test = StellarDiGraph(G_test)
    else:
        G_test = StellarGraph(G_test)

    # Convert node attributes to feature values
    nfs = NodeAttributeSpecification()
    nfs.add_all_attributes(
        G_train, "paper", BinaryConverter, ignored_attributes=["subject"]
    )

    # Learn feature and target conversion for ML for G_train
    G_train.fit_attribute_spec(feature_spec=nfs)
    # Apply feature and target conversion to G_test
    G_test.set_attribute_spec(feature_spec=nfs)

    # Mapper feeds link data from sampled subgraphs to GraphSAGE model
    train_mapper = GraphSAGELinkMapper(
        G_train,
        edge_ids_train,
        edge_labels_train,
        batch_size,
        num_samples,
        name="train",
    )
    test_mapper = GraphSAGELinkMapper(
        G_test, edge_ids_test, edge_labels_test, batch_size, num_samples, name="test"
    )

    # GraphSAGE model
    # Old way to initialise GraphSAGE:
    # graphsage = GraphSAGE(
    #     layer_sizes=layer_size,
    #     n_samples=num_samples,
    #     input_dim=G_train.get_feature_size(),
    #     bias=True,
    #     dropout=dropout,
    # )
    # New way to initialise GraphSAGE:
    graphsage = GraphSAGE(
        layer_sizes=layer_size,
        mapper=train_mapper,
        bias=True,
        dropout=dropout,
    )

    # Expose input and output sockets of the model, for source and destination nodes:
    x_inp_src, x_out_src = graphsage.default_model(flatten_output=False)
    x_inp_dst, x_out_dst = graphsage.default_model(flatten_output=False)
    # re-pack into a list where (source, target) inputs alternate, for link inputs:
    x_inp = [x for ab in zip(x_inp_src, x_inp_dst) for x in ab]
    # same for outputs:
    x_out = [x_out_src, x_out_dst]

    # Final estimator layer
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_feature_method=args.edge_feature_method
    )(x_out)

    # Stack the GraphSAGE and prediction layers into a Keras model, and specify the loss
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=learning_rate),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy],
    )

    # Evaluate the initial (untrained) model on the train and test set:
    init_train_metrics = model.evaluate_generator(train_mapper)
    init_test_metrics = model.evaluate_generator(test_mapper)

    print("\nTrain Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Train model
    print("\nTraining the model for {} epochs...".format(num_epochs))
    history = model.fit_generator(
        train_mapper,
        epochs=num_epochs,
        validation_data=test_mapper,
        verbose=2,
        shuffle=True,
    )

    # Evaluate and print metrics
    train_metrics = model.evaluate_generator(train_mapper)
    test_metrics = model.evaluate_generator(test_mapper)

    print("\nTrain Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Save model
    save_str = "_n{}_l{}_d{}_r{}".format(
        "_".join([str(x) for x in num_samples]),
        "_".join([str(x) for x in layer_size]),
        dropout,
        learning_rate,
    )
    model.save("graphsage_link_pred" + save_str + ".h5")


def test(G, model_file: AnyStr, batch_size: int):
    """
    Load the serialized model and evaluate on all links in the graph.

    Args:
        G: NetworkX graph file
        model_file: Location of Keras model to load
        batch_size: Size of batch for inference
    """
    model = keras.models.load_model(
        model_file, custom_objects={"MeanAggregator": MeanAggregator}
    )

    # Get required input shapes from model
    num_samples = [
        int(model.input_shape[ii + 1][1] / model.input_shape[ii][1])
        for ii in range(1, len(model.input_shape) - 1, 2)
    ]

    G = get_largest_cc(G)

    edge_splitter_test = EdgeSplitter(G)
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method=args.edge_sampling_method, probs=args.edge_sampling_probs
    )

    # Convert G_test to StellarGraph objects for ML:
    if G_test.is_directed():
        G_test = StellarDiGraph(
            G_test,
            node_type_name=GLOBALS.TYPE_ATTR_NAME,
            edge_type_name=GLOBALS.TYPE_ATTR_NAME,
        )
    else:
        G_test = StellarGraph(
            G_test,
            node_type_name=GLOBALS.TYPE_ATTR_NAME,
            edge_type_name=GLOBALS.TYPE_ATTR_NAME,
        )

    # Convert node attributes to feature values
    nfs = NodeAttributeSpecification()
    nfs.add_all_attributes(
        G_test, "paper", BinaryConverter, ignored_attributes=["subject"]
    )

    # Learn feature conversion for G_test
    G_test.fit_attribute_spec(feature_spec=nfs)

    # Mapper feeds data from (source, target) sampled subgraphs to GraphSAGE model
    test_mapper = GraphSAGELinkMapper(
        G_test, edge_ids_test, edge_labels_test, batch_size, num_samples, name="test"
    )

    # Evaluate and print metrics
    # TODO: add all-links evaluation
    test_metrics = model.evaluate_generator(test_mapper)

    print("\nTest Set Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graph link prediction using GraphSAGE"
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
        "--edge_sampling_method",
        nargs="?",
        default="global",
        help="Negative edge sampling method: local or global",
    )
    parser.add_argument(
        "--edge_sampling_probs",
        nargs="?",
        default=[0.0, 0.25, 0.50, 0.25],
        help="Negative edge sample probabilities (for local sampling method) with respect to distance from starting node",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=500, help="Batch size for training"
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
        default=0.0005,
        help="Learning rate for training model",
    )
    parser.add_argument(
        "-s",
        "--neighbour_samples",
        type=int,
        nargs="*",
        default=[30, 10],
        help="The number of nodes sampled at each layer",
    )
    parser.add_argument(
        "-l",
        "--layer_size",
        type=int,
        nargs="*",
        default=[50, 50],
        help="The number of hidden features at each layer",
    )
    parser.add_argument(
        "-g", "--graph", type=str, default=None, help="The graph stored in EPGM format."
    )
    parser.add_argument(
        "-i",
        "--ignore_node_attr",
        nargs="+",
        default=[],
        help="List of node attributes to ignore (e.g., name, id, etc.)",
    )
    parser.add_argument(
        "-m",
        "--edge_feature_method",
        type=str,
        default="ip",
        help="The method for combining node embeddings into edge embeddings: 'concat', 'mul', or 'ip",
    )
    args, cmdline_args = parser.parse_known_args()

    graph_loc = os.path.expanduser(args.graph)
    # G = read_epgm_graph(
    #     graph_loc,
    #     ignored_attributes=args.ignore_node_attr,
    #     remove_converted_attrs=False,
    # )

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
