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
import os

from utils.node2vec_feature_learning import Node2VecFeatureLearning
from utils.metapath2vec_feature_learning import Metapath2VecFeatureLearning

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def link_prediction_clf(feature_learner, edge_data, binary_operators=None):
    """
    Performs link prediction given that node features have already been computed. It uses the node features to
    derive edge features using the operators given. Then it trains a Logistic Regression classifier to predict
    links between nodes.

    Args:
        feature_learner: Representation learning object.
        edge_data: (2-tuple) Positive and negative edge data for training the classifier
        binary_operators: Binary operators applied on node features to produce the corresponding edge feature.

    Returns:
        Returns the ROCAUC score achieved by the classifier for each of the specified binary operators.
    """
    scores = []  # the auc values for each binary operator (based on test set performance)
    clf_best = None
    score_best = 0
    op_best = ""

    if binary_operators is None:
        print("WARNING: Using default binary operator 'h'")
        binary_operators = ["h"]

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
            "Operator: {} Score (ROC AUC on test set of edge_data): {:.3f}".format(
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
        print("WARNING: Using default binary operator 'h'")
        binary_operators = ["h"]

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

        # print('Correct labels y {}'.format(y))
        # print('Predictions y_pred {}'.format(y_pred[:, 1]))
        print("Prediction score (ROC AUC):", score_auc)
        scores.append({"op": binary_operator, "score": score_auc})

    return scores


def predict(feature_learner, edge_data, clf, binary_operators=None):
    """
    Given a node feature learner and a trained classifier, it computes edge features, uses the classifier to predict
    the given edge data and calculate prediction accuracy.
    :param feature_learner:
    :param edge_data:
    :param clf:
    :param binary_operators:
    :return: a prediction probability for each binary operator in a dictionary where operator is key and prediction is
    the value.
    """

    if binary_operators is None:
        print("WARNING: Using default binary operator 'h'")
        binary_operators = ["h"]

    predictions = {}
    # for each type of binary operator
    for binary_operator in binary_operators:
        # Derive edge features from node features using the given binary operator
        X, y = feature_learner.transform(edge_data, binary_operator)
        #
        y_pred = clf.predict_proba(X)  # predict

        predictions[binary_operator] = y_pred

    return predictions


def train_homogeneous_graph(
    g_train,
    g_test,
    output_node_features,  # filename for writing node embeddings
    edge_data_ids_train,
    edge_data_labels_train,  # train edge data
    edge_data_ids_test,
    edge_data_labels_test,  # test edge data
    parameters,
):
    # Using g_train and edge_data_train train a classifier for edge prediction
    feature_learner_train = Node2VecFeatureLearning(
        g_train, embeddings_filename=os.path.expanduser(output_node_features)
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
    binary_operators = ["h", "avg", "l1", "l2"]
    scores_train, clf_edge, binary_operator = link_prediction_clf(
        feature_learner=feature_learner_train,
        edge_data=(edge_data_ids_train, edge_data_labels_train),
        binary_operators=binary_operators,
    )

    # Do representation learning on g_test and use the previously trained classifier on g_train to predict
    # edge_data_test
    feature_learner_test = Node2VecFeatureLearning(
        g_test, embeddings_filename=os.path.expanduser(output_node_features)
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
        print(
            "     Operator: {}  Score (ROC AUC): {:.2f}".format(
                score["op"], score["score"]
            )
        )
    print("\n  ****************************")

    return feature_learner_train, feature_learner_test, clf_edge


def train_heterogeneous_graph(
    g_train,
    g_test,
    output_node_features,  # filename for writing node embeddings
    edge_data_ids_train,
    edge_data_labels_train,  # train edge data
    edge_data_ids_test,
    edge_data_labels_test,  # test edge data
    metapaths,
    parameters,
):
    # metapaths = [["Person", "Group", "Person"], ["Person", "Group", "Person", "Person"]]

    # Using g_train and edge_data_train train a classifier for edge prediction
    feature_learner_train = Metapath2VecFeatureLearning(
        g_train, embeddings_filename=os.path.expanduser(output_node_features)
    )
    feature_learner_train.fit(
        metapaths=metapaths,
        d=parameters["dimensions"],
        r=parameters["num_walks"],
        l=parameters["walk_length"],
        k=parameters["window_size"],
    )
    # Train the classifier
    binary_operators = ["h", "avg", "l1", "l2"]
    scores_train, clf_edge, binary_operator = link_prediction_clf(
        feature_learner=feature_learner_train,
        edge_data=(edge_data_ids_train, edge_data_labels_train),
        binary_operators=binary_operators,
    )

    # Do representation learning on g_test and use the previously trained classifier on g_train to predict
    # edge_data_test
    feature_learner_test = Metapath2VecFeatureLearning(
        g_test, embeddings_filename=os.path.expanduser(output_node_features)
    )
    feature_learner_test.fit(
        metapaths=metapaths,
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

    print("\n  **** Scores on test set (HIN) ****\n")
    for score in scores:
        print(
            "     Operator: {}  Score (ROC AUC): {:.2f}".format(
                score["op"], score["score"]
            )
        )
    print("\n  ****************************")

    return feature_learner_train, feature_learner_test, clf_edge
