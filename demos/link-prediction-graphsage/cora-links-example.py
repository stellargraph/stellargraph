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
This example requires the CORA dataset - see the README for how to obtain the dataset.

Example usage, assuming the CORA dataset has been downloaded and extracted into ~/data/cora:
python cora-links-example.py -g ~/data/cora -e 10 -d 0.1 --ignore_node_attr subject --edge_sampling_method global --edge_feature_method ip

usage: cora-links-example.py [-h] [-c [CHECKPOINT]] [-e EPOCHS] [-b BATCH_SIZE]
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
        One of:
                'ip' or 'dot' (inner product, ip(u,v) = sum_{i=1..d}{u_i*v_i}),
                'mul' or 'hadamard' (element-wise multiplication, h(u,v)_i = u_i*v_i),
                'concat' (concatenation),
                'l1' (l1(u,v)_i = |u_i-v_i|),
                'l2' (l2(u,v)_i = (u_i-v_i)^2),
                'avg' (avg(u,v) = (u+v)/2)
"""
import os
import argparse
import networkx as nx
import pandas as pd
from typing import AnyStr, List

import keras
from keras import optimizers, losses, metrics

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.layer import GraphSAGE, MeanAggregator, link_classification
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph import globals

from sklearn import feature_extraction


def load_data(graph_loc, ignore_attr):
    """
    Load Cora dataset, and create a NetworkX graph
    Args:
        graph_loc: dataset path

    Returns: NetworkX graph with numeric node features

    """

    # Load the edge list
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

    g = nx.from_pandas_edgelist(edgelist)

    # Extract the feature data. These are the node feature vectors that the Keras model will use as input.
    # The CORA dataset contains attributes 'w_x' that correspond to words found in that publication.
    predictor_names = sorted(set(column_names) - set(ignore_attr))

    if "subject" in predictor_names:
        # Convert node features to numeric vectors
        feature_encoding = feature_extraction.DictVectorizer(sparse=False)
        node_features = feature_encoding.fit_transform(
            node_data[predictor_names].to_dict("records")
        )
    else:  # node features are already numeric, no further conversion is needed
        node_features = node_data[predictor_names].values

    node_ids = node_data.index

    for nid, f in zip(node_ids, node_features):
        g.node[nid][globals.TYPE_ATTR_NAME] = "paper"
        g.node[nid][globals.FEATURE_ATTR_NAME] = f

    return g


def train(
    G,
    layer_size: List[int],
    num_samples: List[int],
    batch_size: int = 100,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
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
    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
    # reduced graph G_test with the sampled links removed:
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method=args.edge_sampling_method, probs=args.edge_sampling_probs
    )

    # From G_test, extract E_train and the reduced graph G_train:
    edge_splitter_train = EdgeSplitter(G_test, G)
    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
    # further reduced graph G_train with the sampled links removed:
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.1, method=args.edge_sampling_method, probs=args.edge_sampling_probs
    )

    # G_train, edge_ds_train, edge_labels_train will be used for model training
    # G_test, edge_ds_test, edge_labels_test will be used for model testing

    # Convert G_train and G_test to StellarGraph objects (undirected, as required by GraphSAGE) for ML:
    G_train = sg.StellarGraph(G_train, node_features="feature")
    G_test = sg.StellarGraph(G_test, node_features="feature")

    # Mapper feeds link data from sampled subgraphs to GraphSAGE model
    # We need to create two mappers: for training and testing of the model
    train_gen = GraphSAGELinkGenerator(
        G_train,
        batch_size,
        num_samples,
        name="train",
    ).flow(edge_ids_train, edge_labels_train)

    test_gen = GraphSAGELinkGenerator(
        G_test,
        batch_size,
        num_samples,
        name="train",
    ).flow(edge_ids_test, edge_labels_test)

    # GraphSAGE model
    graphsage = GraphSAGE(
        layer_sizes=layer_size, generator=train_gen, bias=True, dropout=dropout
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
    init_train_metrics = model.evaluate_generator(train_gen)
    init_test_metrics = model.evaluate_generator(test_gen)

    print("\nTrain Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Train model
    print("\nTraining the model for {} epochs...".format(num_epochs))
    history = model.fit_generator(
        train_gen,
        epochs=num_epochs,
        validation_data=test_gen,
        verbose=2,
        shuffle=True,
    )

    # Evaluate and print metrics
    train_metrics = model.evaluate_generator(train_gen)
    test_metrics = model.evaluate_generator(test_gen)

    print("\nTrain Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Save the trained model
    save_str = "_n{}_l{}_d{}_r{}".format(
        "_".join([str(x) for x in num_samples]),
        "_".join([str(x) for x in layer_size]),
        dropout,
        learning_rate,
    )
    model.save("graphsage_link_pred" + save_str + ".h5")


def test(G, model_file: AnyStr, batch_size: int = 100):
    """
    Load the serialized model and evaluate on a random balanced subset of all links in the graph.
    Note that the set of links the model is evaluated on may contain links from the model's training set.
    To avoid this, set the seed of the edge splitter to the same seed as used for link splitting in train()

    Args:
        G: NetworkX graph file
        model_file: Location of Keras model to load
        batch_size: Size of batch for inference
    """
    print("Loading model from ", model_file)
    model = keras.models.load_model(
        model_file, custom_objects={"MeanAggregator": MeanAggregator}
    )

    # Get required input shapes from model
    num_samples = [
        int(model.input_shape[ii + 1][1] / model.input_shape[ii][1])
        for ii in range(1, len(model.input_shape) - 1, 2)
    ]

    edge_splitter_test = EdgeSplitter(G)
    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
    # reduced graph G_test with the sampled links removed:
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method=args.edge_sampling_method, probs=args.edge_sampling_probs
    )

    # Convert G_test to StellarGraph object (undirected, as required by GraphSAGE):
    G_test = sg.StellarGraph(G_test, node_features="feature")

    # Mapper feeds data from (source, target) sampled subgraphs to GraphSAGE model
    test_mapper = GraphSAGELinkMapper(
        G_test, edge_ids_test, edge_labels_test, batch_size, num_samples, name="test"
    )

    # Evaluate and print metrics
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
        help="Load a saved model checkpoint file",
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
        help="Negative edge sample probabilities (for local sampling method only) with respect to distance from starting node. "
        "This should always start with 0.0",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=50, help="Batch size for training"
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
        help="Initial learning rate for model training",
    )
    parser.add_argument(
        "-s",
        "--neighbour_samples",
        type=int,
        nargs="*",
        default=[20, 10],
        help="The number of neighbour nodes sampled at each GraphSAGE layer",
    )
    parser.add_argument(
        "-l",
        "--layer_size",
        type=int,
        nargs="*",
        default=[50, 50],
        help="The number of hidden features at each GraphSAGE layer",
    )
    parser.add_argument(
        "-g",
        "--graph",
        type=str,
        default=None,
        help="Location of the dataset (directory)",
    )
    parser.add_argument(
        "-i",
        "--ignore_node_attr",
        nargs="+",
        default=[],
        help="List of node attributes to ignore (e.g., subject, name, id, etc.)",
    )
    parser.add_argument(
        "-m",
        "--edge_feature_method",
        type=str,
        default="ip",
        help="The method for combining node embeddings into edge embeddings: 'concat', 'mul', 'ip', 'l1', 'l2', or 'avg'",
    )

    args, cmdline_args = parser.parse_known_args()

    graph_loc = os.path.expanduser(args.graph)
    # Load the dataset - this assumes it is the CORA dataset
    G = load_data(graph_loc, args.ignore_node_attr)

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
