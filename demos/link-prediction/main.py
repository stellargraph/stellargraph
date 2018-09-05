# -*- coding: utf-8 -*-
#
# Copyright 2017-2018 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
from math import isclose
import os
import networkx as nx
import numpy as np
from stellargraph.data.edge_splitter import EdgeSplitter
from stellargraph.data.stellargraph import StellarGraph
from utils.cl_arguments_parser import parse_args
from utils.read_graph import read_graph
from utils.predictors import *
from collections import Counter
import multiprocessing


# Default parameters for Node2Vec
parameters = {
    "p": 1.,  # Parameter p
    "q": 1.,  # Parameter q
    "dimensions": 128,  # dimensionality of node2vec embeddings
    "num_walks": 10,  # Number of walks from each node
    "walk_length": 80,  # Walk length
    "window_size": 10,  # Context size for word2vec
    "iter": 1,  # number of SGD iterations (epochs)
    "workers": multiprocessing.cpu_count(),  # number of workers for word2vec
    "weighted": False,  # is graph weighted?
    "directed": False,  # are edges directed?
}


def print_distance_probabilities(node_distances):
    counts = Counter(node_distances)
    d_total = sum(counts.values())
    counts_normalized = {k: v / d_total for k, v in counts.items()}
    counts_normalized = sorted(counts_normalized.items(), key=lambda x: x[0])
    counts = [v for k, v in counts_normalized]
    print(
        "Normalized distances between source and target nodes in negative samples: {}".format(
            counts
        )
    )


def get_metapaths_from_str(metapaths_str):
    """
    Metapaths are given as a string with the following format "p, v, p; a, p, v, p, a" where two metapaths
    are given separated by ; and each metapath consists of node labels (treated as string) separated by commas.
    Args:
        metapaths_str: Metapaths in string format

    Returns: A list of list of node labels where each list specifies a metapath.

    """
    if len(metapaths_str) == 0:
        metapaths = [
            ["group", "user", "group"],
            ["user", "group", "user"],
            ["user", "group", "user", "user"],
            ["user", "user"],
        ]
    else:
        metapaths = []
        m_tokens = metapaths_str.split(";")

        for metapath in m_tokens:
            metapaths.append(list(metapath.split(",")))

    return metapaths


if __name__ == "__main__":
    args = parse_args()

    p = float(args.p)
    if p <= 0 or p >= 1:
        print("** Invalid value: p should be in the interval (0, 1) **")
        exit(0)

    subgraph_size = float(args.subgraph_size)
    if subgraph_size <= 0 or subgraph_size > 1:
        print("** Invalid value: subgraph_size should be in the interval (0, 1] **")
        exit(0)

    print("Negative edges sampling method is set to {}.".format(args.sampling_method))
    sampling_probs = np.array([float(n) for n in args.sampling_probs.split(",")])
    if not isclose(sum(sampling_probs), 1.0):
        print(
            "WARNING: Negative edge distance sampling probabilities do not sum to 1.0. They will be normalized to continue."
        )
        sampling_probs = sampling_probs / np.sum(
            sampling_probs
        )  # make sure that the probabilities sum to 1.0
        print("  Normalized Sampling Probabilities: {}".format(sampling_probs))

    graph_filename = os.path.expanduser(args.input_graph)
    dataset_name = args.dataset_name
    # Load the graph from disk
    g_nx = read_graph(
        graph_file=graph_filename,
        dataset_name=dataset_name,
        is_directed=parameters["directed"],
        is_weighted=parameters["weighted"],
    )

    if args.subsample_graph:
        # subsample g_nx
        nodes = g_nx.nodes(data=False)
        np.random.shuffle(nodes)
        subgraph_num_nodes = int(len(nodes) * subgraph_size)
        g_nx = g_nx.subgraph(nodes[0:subgraph_num_nodes])

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

    # From the original graph, extract E_test and G_test
    edge_splitter_test = EdgeSplitter(g_nx)
    if args.hin:
        g_test, edge_data_ids_test, edge_data_labels_test = edge_splitter_test.train_test_split(
            p=p,
            edge_label=args.edge_type,
            edge_attribute_label=args.edge_attribute_label,
            edge_attribute_threshold=args.edge_attribute_threshold,
            attribute_is_datetime=args.attribute_is_datetime,
            method=args.sampling_method,
            probs=sampling_probs,
        )
    else:
        g_test, edge_data_ids_test, edge_data_labels_test = edge_splitter_test.train_test_split(
            p=p, method=args.sampling_method, probs=sampling_probs
        )
    if args.show_histograms:
        if args.sampling_method == "local":
            bins = np.arange(1, len(sampling_probs) + 2)
        else:
            bins = np.arange(
                1, np.max(edge_splitter_test.negative_edge_node_distances) + 2
            )

        plt.hist(
            edge_splitter_test.negative_edge_node_distances,
            bins=bins,
            rwidth=0.5,
            align="left",
        )
        plt.show()

    if args.sampling_method == "local":
        print_distance_probabilities(edge_splitter_test.negative_edge_node_distances)

    edge_splitter_train = EdgeSplitter(g_test, g_nx)
    if args.hin:
        g_train, edge_data_ids_train, edge_data_labels_train = edge_splitter_train.train_test_split(
            p=p,
            edge_label=args.edge_type,
            edge_attribute_label=args.edge_attribute_label,
            edge_attribute_threshold=args.edge_attribute_threshold,
            attribute_is_datetime=args.attribute_is_datetime,
            method=args.sampling_method,
            probs=sampling_probs,
        )
    else:
        g_train, edge_data_ids_train, edge_data_labels_train = edge_splitter_train.train_test_split(
            p=p, method=args.sampling_method, probs=sampling_probs
        )
    if args.show_histograms:
        if args.sampling_method == "local":
            bins = np.arange(1, len(sampling_probs) + 2)
        else:
            bins = np.arange(
                1, np.max(edge_splitter_train.negative_edge_node_distances) + 2
            )
        plt.hist(
            edge_splitter_train.negative_edge_node_distances,
            bins=bins,
            rwidth=0.5,
            align="left",
        )
        plt.show()

    if args.sampling_method == "local":
        print_distance_probabilities(edge_splitter_train.negative_edge_node_distances)

    # this is so that Node2Vec works because it expects Graph not MultiGraph type
    g_test = nx.Graph(g_test)
    g_train = nx.Graph(g_train)

    if args.hin:
        # prepare the metapaths if given in the command line
        metapaths = get_metapaths_from_str(args.metapaths)

        train_heterogeneous_graph(
            g_train=StellarGraph(g_train),
            g_test=StellarGraph(g_test),
            output_node_features=args.output_node_features,
            edge_data_ids_train=edge_data_ids_train,
            edge_data_labels_train=edge_data_labels_train,
            edge_data_ids_test=edge_data_ids_test,
            edge_data_labels_test=edge_data_labels_test,
            metapaths=metapaths,
            parameters=parameters,
        )
    else:
        train_homogeneous_graph(
            g_train=g_train,
            g_test=g_test,
            output_node_features=args.output_node_features,
            edge_data_ids_train=edge_data_ids_train,
            edge_data_labels_train=edge_data_labels_train,
            edge_data_ids_test=edge_data_ids_test,
            edge_data_labels_test=edge_data_labels_test,
            parameters=parameters,
        )
