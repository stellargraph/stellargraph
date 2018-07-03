"""
Graph node classification using GraphSAGE.
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
from typing import AnyStr, List

import keras
from keras import optimizers, losses, layers, metrics
from keras.utils.np_utils import to_categorical

from stellar.data.epgm import EPGM
from stellar.data.node_splitter import NodeSplitter
from stellar.data.explorer import SampledBreadthFirstWalk

from stellar.layer.graphsage import GraphSAGE, MeanAggregator
from stellar.mapper.node_mappers import GraphSAGENodeMapper


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
        layer_size: A list of number of hidden nodes in each layer
        num_samples: Number of neighbours to sample at each layer
        batch_size: Size of batch for inference
        num_epochs: Number of epochs to train the model
        learning_rate: Initial Learning rate
        dropout: The dropout (0->1)
    """
    # Split head nodes into train/test
    splitter = NodeSplitter()
    graph_nodes = np.array(
        [(v, vdata.get("subject")) for v, vdata in G.nodes(data=True)]
    )
    train_nodes, val_nodes, test_nodes, _ = splitter.train_test_split(
        y=graph_nodes, p=100, test_size=500
    )
    train_ids = [v[0] for v in train_nodes]
    test_ids = list(G.nodes())
    val_ids = [v[0] for v in val_nodes]

    # Sampler chooses random sampled subgraph for each head node
    sampler = SampledBreadthFirstWalk(G)

    # Mapper feeds data from sampled subgraph to GraphSAGE model
    train_mapper = GraphSAGENodeMapper(
        G, train_ids, sampler, batch_size, num_samples, target_id="target", name="train"
    )
    test_mapper = GraphSAGENodeMapper(
        G, test_ids, sampler, batch_size, num_samples, target_id="target", name="test"
    )
    val_mapper = GraphSAGENodeMapper(
        G,
        val_ids,
        sampler,
        batch_size,
        num_samples,
        target_id="target",
        name="validate",
    )

    # GraphSAGE model
    model = GraphSAGE(
        output_dims=layer_size,
        n_samples=num_samples,
        input_dim=G.feature_size,
        bias=True,
        dropout=dropout,
    )
    x_inp, x_out = model.default_model(flatten_output=True)

    # Final estimator layer
    prediction = layers.Dense(
        units=len(G.target_category_values), activation="softmax"
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
        train_mapper, epochs=num_epochs, verbose=2, shuffle=True
    )

    # Evaluate and print metrics
    val_metrics = model.evaluate_generator(val_mapper)
    test_metrics = model.evaluate_generator(test_mapper)

    print("\nValidation Set Metrics:")
    for name, val in zip(model.metrics_names, val_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics:")
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
    Load the serialized model and evaluate on all nodes in the graph.

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

    # Sampler chooses random sampled subgraph for each head node
    sampler = SampledBreadthFirstWalk(G)
    all_ids = list(G.nodes())

    # Mapper feeds data from sampled subgraph to GraphSAGE model
    test_mapper = GraphSAGENodeMapper(
        G, all_ids, sampler, batch_size, num_samples, target_id="target", name="test"
    )

    # Evaluate and print metrics
    test_metrics = model.evaluate_generator(test_mapper)

    print("\nAll-node Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
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
        help="Load a saved checkpoint file",
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
