"""
Graph link prediction using GraphSAGE.
Requires a EPGM graph as input.
This currently is only tested on the CORA dataset.

Example usage:
python epgm-example.py -g ../../tests/resources/data/cora/cora.epgm -l 50 50 -s 20 10 -e 20 -d 0.5 -r 0.01

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
import networkx as nx
from typing import AnyStr, List, Optional

import keras
from keras import optimizers, losses, layers, metrics
from keras.utils.np_utils import to_categorical
from keras.layers import Concatenate, Dense, Lambda, Multiply
from keras import backend as K

from stellar.data.epgm import EPGM
from stellar.data.edge_splitter import EdgeSplitter

from stellar.layer.graphsage import GraphSAGE, MeanAggregator
from stellar.mapper.link_mappers import GraphSAGELinkMapper


def read_epgm_graph(
    graph_file,
    dataset_name=None,
    node_type=None,
    target_attribute=None,
    ignored_attributes=[],
    target_type=None,
    remove_converted_attrs=False,
):
    G_epgm = EPGM(graph_file)
    graphs = G_epgm.G["graphs"]

    # if dataset_name is not given, use the name of the 1st graph head
    if not dataset_name:
        dataset_name = graphs[0]["meta"]["label"]
        print(
            "WARNING: dataset name not specified, using dataset '{}' in the 1st graph head".format(
                dataset_name
            )
        )

    graph_id = None
    for g in graphs:
        if g["meta"]["label"] == dataset_name:
            graph_id = g["id"]

    if node_type is None:
        node_type = G_epgm.node_types(graph_id)[0]

    g_nx = G_epgm.to_nx(graph_id)

    # Check if graph is connected; if not, then select the largest subgraph to continue
    if nx.is_connected(g_nx):
        print("Graph is connected")
    else:
        print("Graph is not connected")
        # take the largest connected component as the data
        g_nx = max(nx.connected_component_subgraphs(g_nx, copy=True), key=len)
        print(
            "Largest subgraph statistics: {} nodes, {} edges".format(
                g_nx.number_of_nodes(), g_nx.number_of_edges()
            )
        )

    # This is the correct way to set the edge weight in a MultiGraph.
    edge_weights = {e: 1 for e in g_nx.edges(keys=True)}
    nx.set_edge_attributes(g_nx, "weight", edge_weights)

    # Find target and predicted attributes from attribute set
    node_attributes = set(G_epgm.node_attributes(graph_id, node_type))
    pred_attr = node_attributes.difference(
        set(ignored_attributes).union([target_attribute])
    )
    converted_attr = pred_attr.union([target_attribute])

    # Enumerate attributes to give numerical index
    g_nx.pred_map = {a: ii for ii, a in enumerate(pred_attr)}

    # Store feature size in graph [??]
    g_nx.feature_size = len(g_nx.pred_map)

    # We map the targets differently depending on target type/
    # * If target_type is None, use the target values directly
    #   (e.g. for regression or binary classification problems)
    # * If the target is 'categorical' encode the target categories
    #   as integers between 0 and number_of_categories - 1
    # * If the target is '1hot' encode the target categories
    #   as a binary vector of length number_of_categories.
    #   see the Keras function keras.utils.np_utils.to_categorical
    g_nx.target_category_values = None
    if target_type is None:
        target_value_function = lambda x: x

    elif target_type == "categorical":
        g_nx.target_category_values = list(
            set([g_nx.node[n][target_attribute] for n in g_nx.nodes()])
        )
        target_value_function = lambda x: g_nx.target_category_values.index(x)

    elif target_type == "1hot":
        g_nx.target_category_values = list(
            set([g_nx.node[n][target_attribute] for n in g_nx.nodes()])
        )
        target_value_function = lambda x: to_categorical(
            g_nx.target_category_values.index(x), len(g_nx.target_category_values)
        )

    else:
        raise ValueError("Target type '{}' is not supported.".format(target_type))

    # Set the "feature" and encoded "target" attributes for all nodes in the graph.
    for v, vdata in g_nx.nodes(data=True):
        # Decode attributes to a feature array
        attr_array = np.zeros(g_nx.feature_size)
        for attr_name, attr_value in vdata.items():
            col = g_nx.pred_map.get(attr_name)
            if col:
                attr_array[col] = attr_value

        # Replace with feature array
        vdata["feature"] = attr_array

        # Decode target attribute to target array
        vdata["target"] = target_value_function(vdata.get(target_attribute))

        # Remove attributes
        if remove_converted_attrs:
            for attr_name in converted_attr:
                if attr_name in vdata:
                    del vdata[attr_name]

    print(
        "Graph statistics: {} nodes, {} edges".format(
            g_nx.number_of_nodes(), g_nx.number_of_edges()
        )
    )
    return g_nx


def classification_predictor(
    hidden_1: Optional[int] = None,
    hidden_2: Optional[int] = None,
    output_act: AnyStr = "softmax",
    method: AnyStr = "ip",
):
    """Returns a function that predicts an edge classification output from node features.

        hidden_1 ([type], optional): Hidden size for the transform of node 1 features.
        hidden_2 ([type], optional): Hidden size for the transform of node 1 features.
        edge_function (str, optional): One of 'ip', 'mul', and 'concat'

    Returns:
        Function taking HinSAGE edge tensors and returning a logit function.
    """

    def edge_function(x):
        x0 = x[0]
        x1 = x[1]

        if hidden_1:
            x0 = Dense(hidden_1, activation="relu")(x0)

        if hidden_2:
            x1 = Dense(hidden_2, activation="relu")(x1)

        if method == "ip":
            out = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=False))(
                [x0, x1]
            )

        elif method == "mul":
            le = Multiply()([x0, x1])
            out = Dense(2, activation=output_act)(le)

        elif method == "concat":
            le = Concatenate()([x0, x1])
            out = Dense(2, activation=output_act)(le)

        return out

    return edge_function


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
    # From the original graph, extract E_test and the reduced graph G_test:
    edge_splitter_test = EdgeSplitter(G)
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method=args.sampling_method, probs=args.sampling_probs
    )

    # From G_test, extract E_train and the reduced graph G_train:
    edge_splitter_train = EdgeSplitter(G_test, G)
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.1, method=args.sampling_method, probs=args.sampling_probs
    )

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
    model = GraphSAGE(
        output_dims=layer_size,
        n_samples=num_samples,
        input_dim=G.feature_size,
        bias=True,
        dropout=dropout,
    )
    x_inp_src, x_out_src = model.default_model(flatten_output=not True)
    x_inp_dst, x_out_dst = model.default_model(flatten_output=not True)
    # re-pack into a list where (source, target) inputs alternate, for link inputs:
    x_inp = [x for ab in zip(x_inp_src, x_inp_dst) for x in ab]
    x_out = [x_out_src, x_out_dst]

    # Final estimator layer - a binary link classifier
    prediction = classification_predictor(hidden_1=32, hidden_2=32, method="ip")(x_out)

    # Create Keras model for training
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=learning_rate),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy],
    )

    # Evaluate the initial model on the train and test set:
    init_train_metrics = model.evaluate_generator(train_mapper)
    init_test_metrics = model.evaluate_generator(test_mapper)

    print("\nTrain Set Metrics of the initial model:")
    for name, val in zip(model.metrics_names, init_train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the initial model:")
    for name, val in zip(model.metrics_names, init_test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Train model
    print("\nTraining the model for {} epochs...".format(num_epochs))
    history = model.fit_generator(
        train_mapper,
        epochs=num_epochs,
        # validation_data=val_mapper,
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
    str_numsamp = "_".join([str(x) for x in num_samples])
    str_layer = "_".join([str(x) for x in layer_size])
    model.save(
        "graphsage_n{}_l{}_d{}_i{}.h5".format(
            str_numsamp, str_layer, dropout, G.feature_size
        )
    )


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
        for ii in range(len(model.input_shape) - 1)
    ]

    edge_splitter_test = EdgeSplitter(G)
    G_test, edge_ids_all, edge_labels_all = edge_splitter_test.train_test_split(
        p=1, method=args.sampling_method, probs=args.sampling_probs
    )

    # Mapper feeds data from (source, target) sampled subgraphs to GraphSAGE model
    test_mapper = GraphSAGELinkMapper(
        G, edge_ids_all, edge_labels_all, batch_size, num_samples, name="test_all"
    )

    # Evaluate and print metrics
    test_metrics = model.evaluate_generator(test_mapper)

    print("\nAll-link Evaluation:")
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
        "--sampling_method",
        nargs="?",
        default="global",
        help="Negative edge sampling method: local or global",
    )
    parser.add_argument(
        "--sampling_probs",
        nargs="?",
        default="0.0, 0.25, 0.50, 0.25",
        help="Negative edge sample probabilities (for local sampling method) with respect to distance from starting node",
    )
    parser.add_argument(
        "-n", "--batch_size", type=int, default=500, help="Batch size for training"
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

    graph_loc = os.path.expanduser(args.graph)
    G = read_epgm_graph(
        graph_loc,
        target_attribute=args.target,
        target_type="1hot",
        remove_converted_attrs=False,
    )

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
