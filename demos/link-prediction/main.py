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
from utils.node2vec_feature_learning import Node2VecFeatureLearning
from sklearn.pipeline import Pipeline
from collections import Counter
import multiprocessing
import argparse

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from stellargraph.data.epgm import EPGM


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


def parse_args():
    """
    Parses the command line arguments.

    :return:
    """
    parser = argparse.ArgumentParser(
        description="Run link prediction on homogeneous and heterogeneous graphs."
    )

    parser.add_argument(
        "--dataset_name",
        nargs="?",
        default="cora",
        help="The dataset name as stored in graphs.json",
    )

    parser.add_argument(
        "--p",
        nargs="?",
        default=0.1,
        help="Percent of edges to sample for positive and negative examples (valid values 0 < p < 1)",
    )

    parser.add_argument(
        "--subgraph_size",
        nargs="?",
        default=0.1,
        help="Percent of nodes for a subgraph of the input data when --subsample is specified (valid values 0 < subgraph_size < 1)",
    )

    parser.add_argument(
        "--edge_type", nargs="?", default="friend", help="The edge type to predict"
    )

    parser.add_argument(
        "--edge_attribute_label",
        nargs="?",
        default="date",
        help="The attribute label by which to split edges",
    )

    parser.add_argument(
        "--edge_attribute_threshold",
        nargs="?",
        default=None,
        help="Any edge with attribute value less that the threshold cannot be removed from graph",
    )

    parser.add_argument(
        "--attribute_is_datetime",
        dest="attribute_is_datetime",
        action="store_true",
        help="If specified, the edge attribute to split on is considered datetime in format dd/mm/yyyy",
    )

    parser.add_argument(
        "--hin",
        dest="hin",
        action="store_true",
        help="If specified, it indicates that the input graph in a heterogenous network; otherwise, the input graph is assumed homogeneous",
    )

    parser.add_argument(
        "--input_graph",
        nargs="?",
        default="~/Projects/data/cora/cora.epgm/",
        help="Input graph filename",
    )

    parser.add_argument(
        "--output_node_features",
        nargs="?",
        default="~/Projects/data/cora/cora.features/cora.emb",
        help="Input graph filename",
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
        "--show_hist",
        dest="show_histograms",
        action="store_true",
        help="If specified, a histogram of the distances between source and target nodes for \
                         negative edge samples will be plotted.",
    )

    parser.add_argument(
        "--subsample",
        dest="subsample_graph",
        action="store_true",
        help="If specified, then the original graph is randomly subsampled to 10% of the original size, \
                        with respect to the number of nodes",
    )

    return parser.parse_args()


def read_graph(graph_file, dataset_name):
    """
    Reads the input network in networkx.

    :param graph_file: The directory where graph in EPGM format is stored
    :param dataset_name: The name of the graph selected out of all the graph heads in EPGM file
    :return: The graph in networkx format
    """
    try:  # assume args.input points to an EPGM graph
        G_epgm = EPGM(graph_file)
        graphs = G_epgm.G["graphs"]
        if (
            dataset_name is None
        ):  # if dataset_name is not given, use the name of the 1st graph head
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

        g = G_epgm.to_nx(graph_id, parameters["directed"])
        if parameters["weighted"]:
            raise NotImplementedError
        else:
            # This is the correct way to set the edge weight in a MultiGraph.
            edge_weights = {e: 1 for e in g.edges(keys=True)}
            nx.set_edge_attributes(g, name="weight", values=edge_weights)
    except:  # otherwise, assume arg.input points to an edgelist file
        if parameters["weighted"]:
            g = nx.read_edgelist(
                graph_file,
                nodetype=int,
                data=(("weight", float),),
                create_using=nx.DiGraph(),
            )
        else:
            g = nx.read_edgelist(graph_file, nodetype=int, create_using=nx.DiGraph())
            for edge in g.edges():
                g[edge[0]][edge[1]]["weight"] = 1  # {'weight': 1}

        if not parameters["directed"]:
            g = g.to_undirected()

    print(
        "Graph statistics: {} nodes, {} edges".format(
            g.number_of_nodes(), g.number_of_edges()
        )
    )
    return g


def link_prediction_clf(feature_learner, edge_data, binary_operators=None):
    """
    Performs link prediction given that node features have already been computed. It uses the node features to
    derive edge features using the operators given. Then it trains a Logistic Regression classifier to predict
    links between nodes.
    :param feature_learner: Representation learning object.
    :param edge_data: (2-tuple) Positive and negative edge data for training the classifier
    :param binary_operators: Binary operators applied on node features to produce the corresponding edge feature.
    :return: Returns the ROCAUC score achieved by the classifier for each of the specified binary operators
    """
    scores = []  # the auc values for each binary operator (based on test set performance)
    clf_best = None
    score_best = 0
    op_best = ""

    if binary_operators is None:
        print("WARNING: Using default binary operator 'l1'")
        binary_operators = ["l1"]

    # for each type of binary operator
    for binary_operator in binary_operators:
        X, y = feature_learner.transform(edge_data, binary_operator)
        #
        # Split the data and keep X_test, y_test for scoring the model; setting the random_state to
        # the same constant for every iteration gives the same split of data so the comparison is fair.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, test_size=0.25
        )
        # LogisticRegressionCV automatically tunes the parameter C using cross validation and the ROC AUC metric
        clf = Pipeline(
            steps=[
                ("sc", StandardScaler()),
                (
                    "clf",
                    LogisticRegressionCV(
                        Cs=10, cv=10, scoring="roc_auc", verbose=False
                    ),
                ),
            ]
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)  # predict on the test set
        if clf.classes_[0] == 1:  # only needs probabilities of positive class
            score_auc = roc_auc_score(y_test, y_pred[:, 0])
        else:
            score_auc = roc_auc_score(y_test, y_pred[:, 1])

        if score_auc >= score_best:
            score_best = score_auc
            clf_best = clf
            op_best = binary_operator

        print(
            "Operator: {} Score (on test set of edge_data): {}".format(
                binary_operator, score_auc
            )
        )
        scores.append({"op": binary_operator, "score": score_auc})

    return scores, clf_best, op_best


def predict_links(feature_learner, edge_data, clf, binary_operators=None):
    """
    Given a node feature learner and a trained classifier, it computes edge features, uses the classifier to predict
    the given edge data and calculate prediction accuracy.
    :param feature_learner:
    :param edge_data:
    :param clf:
    :param binary_operators:
    :return:
    """
    if binary_operators is None:
        print("WARNING: Using default binary operator 'l1'")
        binary_operators = ["l1"]

    scores = []  # the auc values for each binary operator (based on test set performance)

    # for each type of binary operator
    for binary_operator in binary_operators:
        # Derive edge features from node features using the given binary operator
        X, y = feature_learner.transform(edge_data, binary_operator)
        #
        y_pred = clf.predict_proba(X)  # predict
        if clf.classes_[0] == 1:  # only needs probabilities of positive class
            score_auc = roc_auc_score(y, y_pred[:, 0])
        else:
            score_auc = roc_auc_score(y, y_pred[:, 1])

        print("Prediction score:", score_auc)
        scores.append({"op": binary_operator, "score": score_auc})

    return scores


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
    g_nx = read_graph(graph_file=graph_filename, dataset_name=dataset_name)

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

    print_distance_probabilities(edge_splitter_train.negative_edge_node_distances)

    # this is so that Node2Vec works because it expects Graph not MultiGraph type
    g_test = nx.Graph(g_test)
    g_train = nx.Graph(g_train)

    # Using g_train and edge_data_train train a classifier for edge prediction
    feature_learner_train = Node2VecFeatureLearning(
        g_train, embeddings_filename=os.path.expanduser(args.output_node_features)
    )
    feature_learner_train.fit(
        p=parameters["p"],
        q=parameters["q"],
        d=parameters["dimensions"],
        r=parameters["num_walks"],
        l=parameters["walk_length"],
        k=parameters["window_size"],
    )
    # Train the classifier
    binary_operators = ["avg", "l1", "l2", "h"]
    scores_train, clf_edge, binary_operator = link_prediction_clf(
        feature_learner=feature_learner_train,
        edge_data=(edge_data_ids_train, edge_data_labels_train),
        binary_operators=binary_operators,
    )

    # Do representation learning on g_test and use the previously trained classifier on g_train to predict
    # edge_data_test
    feature_learner_test = Node2VecFeatureLearning(
        g_test, embeddings_filename=os.path.expanduser(args.output_node_features)
    )
    feature_learner_test.fit(
        p=parameters["p"],
        q=parameters["q"],
        d=parameters["dimensions"],
        r=parameters["num_walks"],
        l=parameters["walk_length"],
        k=parameters["window_size"],
    )

    scores = predict_links(
        feature_learner=feature_learner_test,
        edge_data=(edge_data_ids_test, edge_data_labels_test),
        clf=clf_edge,
        binary_operators=[binary_operator],
    )

    print("\n  **** Scores on test set ****\n")
    for score in scores:
        print("     Operator: {}  Score: {}".format(score["op"], score["score"]))
    print("\n  ****************************")
