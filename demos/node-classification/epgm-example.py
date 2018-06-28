"""
Graph node classification using GraphSAGE. Requires a EPGM graph as input.
This currently is only tested on the CORA dataset.


usage: epgm-example.py [-h] [-c [CHECKPOINT]] [-n BATCH_SIZE] [-e EPOCHS]
                       [-s [NEIGHBOUR_SAMPLES [NEIGHBOUR_SAMPLES ...]]]
                       [-l [LAYER_SIZE [LAYER_SIZE ...]]] [-g GRAPH]
                       [-f FEATURES] [-t TARGET]

optional arguments:
  -h, --help            show this help message and exit
  -c [CHECKPOINT], --checkpoint [CHECKPOINT]
                        Load a saved checkpoint file
  -n BATCH_SIZE, --batch_size BATCH_SIZE
                        Load a save checkpoint file
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train for
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
                        The target node attribute
"""
import time
import os
import argparse
import pickle
import random

import tensorflow as tf
import numpy as np
import pandas as pd
import networkx as nx
import itertools as it
import functools as ft

from collections import defaultdict
from typing import AnyStr, Any, List, Tuple, Dict, Optional

from keras import backend as K
from keras import Input, Model, optimizers, losses, activations, metrics
from keras.layers import Layer, Dense, Concatenate, Multiply, Activation, Lambda
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical

from stellar.data.epgm import EPGM
from stellar.data.node_splitter import NodeSplitter
from stellar.layer.graphsage import GraphSAGE
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

    # Index nodes in graph
    for ii, v in enumerate(g_nx.nodes):
        g_nx.nodes[v]["id"] = ii

    # Enumerate attributes to give numerical index
    g_nx.pred_map = {a: ii for ii, a in enumerate(pred_attr)}

    # Store feature size in graph [??]
    g_nx.feature_size = len(g_nx.pred_map)

    # How do we map target attributes to numerical values?
    g_nx.target_category_values = None
    if target_type is None:
        target_value_function = lambda x: x

    elif target_type == "categorical":
        g_nx.target_category_values = list(
            set([g_nx.nodes[n][target_attribute] for n in g_nx.nodes])
        )
        target_value_function = lambda x: g_nx.target_category_values.index(x)

    elif target_type == "1hot":
        g_nx.target_category_values = list(
            set([g_nx.nodes[n][target_attribute] for n in g_nx.nodes])
        )
        target_value_function = lambda x: to_categorical(
            g_nx.target_category_values.index(x), len(g_nx.target_category_values)
        )

    else:
        raise ValueError("Target type '{}' is not supported.".format(target_type))

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


class GraphsageSampler:
    def __init__(self, G, samples_per_hop):
        self.G = G
        self.samples_per_hop = samples_per_hop

    def _sample_neighbourhood(self, head_nodes, samples_per_hop):
        if len(samples_per_hop) < 1:
            return [head_nodes]

        num_for_hop, next_samples = samples_per_hop[0], samples_per_hop[1:]
        hop_samples = []
        for hn in head_nodes:
            neighs = list(self.G.neighbors(hn))
            if len(neighs) < num_for_hop:
                neigh_sampled = random.choices(neighs, k=num_for_hop)
            else:
                neigh_sampled = random.sample(neighs, num_for_hop)
            hop_samples.extend(neigh_sampled)

        return [head_nodes] + self._sample_neighbourhood(hop_samples, next_samples)

    def __call__(self, head_nodes):
        return self._sample_neighbourhood(head_nodes, self.samples_per_hop)


def train(
    G,
    layer_size: List[int],
    node_samples: List[int],
    batch_size: int = 100,
    num_epochs: int = 10,
):
    # Sampler object
    sampler = GraphsageSampler(G, node_samples)

    # Split head nodes into train/test
    splitter = NodeSplitter()
    graph_nodes = np.array(
        [(v, vdata.get("subject")) for v, vdata in G.nodes(data=True)]
    )
    train_nodes, val_nodes, test_nodes, _ = splitter.train_test_split(
        y=graph_nodes, p=300, test_size=500
    )
    train_ids = [v[0] for v in train_nodes]
    test_ids = [v[0] for v in test_nodes]
    val_ids = [v[0] for v in val_nodes]

    # Mapper object
    train_mapper = GraphSAGENodeMapper(
        G, train_ids, sampler, batch_size, target_id="target", name="train"
    )
    test_mapper = GraphSAGENodeMapper(
        G, test_ids, sampler, batch_size, target_id="target", name="test"
    )

    # GraphSAGE model
    model = GraphSAGE(
        output_dims=layer_size,
        n_samples=node_samples,
        input_dim=G.feature_size,
        bias=True,
        dropout=0.5,
    )
    x_inp, x_out = model.default_model(flatten_output=True)

    # Final estimator layer
    prediction = Dense(units=len(G.target_category_values), activation="softmax")(x_out)

    # Create Keras model for training
    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=0.0005),
        loss=losses.categorical_crossentropy,
        metrics=[metrics.categorical_accuracy],
    )

    # Train model
    history = model.fit_generator(
        train_mapper, epochs=num_epochs, verbose=2, shuffle=True
    )

    # Evaluate and print metrics
    test_metrics = model.evaluate_generator(test_mapper)

    print("Test Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))


def test(G, model_file: AnyStr):
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
        "-n", "--batch_size", type=int, default=500, help="Load a save checkpoint file"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=10, help="Number of epochs to train for"
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

    # Training: batch size & epochs
    batch_size = args.batch_size
    num_epochs = args.epochs

    # number of samples per additional layer,
    node_samples = args.neighbour_samples

    # number of features per additional layer
    layer_size = args.layer_size

    if args.checkpoint is None:
        train(G, layer_size, node_samples, batch_size, num_epochs)
    else:
        test(G, args.checkpoint)
