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
import numpy as np
import networkx as nx
from typing import AnyStr, List, Optional

import keras
from keras import optimizers, losses, metrics
from keras.layers import Concatenate, Dense, Lambda, Multiply, Reshape
from keras import backend as K

from stellar.data.epgm import EPGM
from stellar.data.edge_splitter import EdgeSplitter

from stellar.layer.graphsage import GraphSAGE, MeanAggregator
from stellar.mapper.link_mappers import GraphSAGELinkMapper


def read_epgm_graph(
    graph_file,
    dataset_name=None,
    node_type=None,
    ignored_attributes=[],
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
    pred_attr = node_attributes.difference(set(ignored_attributes))
    converted_attr = pred_attr

    # sets are unordered, so sort them to ensure reproducible order of node features:
    pred_attr = sorted(pred_attr)
    converted_attr = sorted(converted_attr)

    # Enumerate attributes to give numerical index
    g_nx.pred_map = {a: ii for ii, a in enumerate(pred_attr)}

    # Store feature size in graph [??]
    g_nx.feature_size = len(g_nx.pred_map)

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


def link_classifier(
    hidden_src: Optional[int] = None,
    hidden_dst: Optional[int] = None,
    output_dim: int = 1,
    output_act: AnyStr = "sigmoid",
    edge_feature_method: AnyStr = "ip",
):
    """Returns a function that predicts a binary edge classification output from node features.

        hidden_src ([type], optional): Hidden size for the transform of source node features.
        hidden_dst ([type], optional): Hidden size for the transform of destination node features.
        output_dim: Number of classifier's output units (desired dimensionality of the output)
        output_act: (str, optional): output function, one of "softmax", "sigmoid", etc. - this can be user-defined, but must be a Keras function
        edge_feature_method (str, optional): Name of the method of combining (src,dst) node features into edge features.
            One of 'ip' (inner product), 'mul' (element-wise multiplication), and 'concat' (concatenation)

    Returns:
        Function taking GraphSAGE edge tensors (i.e., pairs of (node_src, node_dst) tensors) and
        returning logits of output_dim length.
    """

    def edge_function(x):
        x0 = x[0]
        x1 = x[1]

        if hidden_src:
            x0 = Dense(hidden_src, activation="relu")(x0)

        if hidden_dst:
            x1 = Dense(hidden_dst, activation="relu")(x1)

        if edge_feature_method == "ip":
            out = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=False))(
                [x0, x1]
            )

        elif edge_feature_method == "mul":
            le = Multiply()([x0, x1])
            out = Dense(output_dim, activation=output_act)(le)
            out = Reshape((output_dim,))(out)

        elif edge_feature_method == "concat":
            le = Concatenate()([x0, x1])
            out = Dense(output_dim, activation=output_act)(le)
            out = Reshape((output_dim,))(out)

        else:
            raise NotImplementedError(
                "classification_predictor: the requested method '{}' is not known/not implemented".format(
                    edge_feature_method
                )
            )

        return out

    print(
        "Using '{}' method to combine node embeddings into edge embeddings".format(
            edge_feature_method
        )
    )
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
    graphsage = GraphSAGE(
        output_dims=layer_size,
        n_samples=num_samples,
        input_dim=G.feature_size,
        bias=True,
        dropout=dropout,
    )

    # Expose input and output sockets of the model, for source and destination nodes:
    x_inp_src, x_out_src = graphsage.default_model(flatten_output=not True)
    x_inp_dst, x_out_dst = graphsage.default_model(flatten_output=not True)
    # re-pack into a list where (source, target) inputs alternate, for link inputs and outputs:
    x_inp = [x for ab in zip(x_inp_src, x_inp_dst) for x in ab]
    x_out = [x_out_src, x_out_dst]

    # Final estimator layer
    prediction = link_classifier(
        hidden_src=None,
        hidden_dst=None,
        output_dim=1,
        output_act="sigmoid",
        method=args.edge_feature_method,
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
    G = read_epgm_graph(
        graph_loc,
        ignored_attributes=args.ignore_node_attr,
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
