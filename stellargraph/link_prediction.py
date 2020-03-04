# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings


def _operator_hadamard(u, v):
    return u * v


def _operator_avg(u, v):
    return (u + v) / 2.0


def _operator_l1(u, v):
    return np.abs(u - v)


def _operator_l2(u, v):
    return (u - v) ** 2


def _create_link_feature(pair, binary_operator):
    u, v = pair
    if binary_operator == "h":
        return _operator_hadamard(u, v)
    elif binary_operator == "avg":
        return _operator_avg(u, v)
    elif binary_operator == "l1":
        return _operator_l1(u, v)
    elif binary_operator == "l2":
        return _operator_l2(u, v)
    else:
        raise ValueError(f"Unexpected binary operator: {binary_operator}")


class LinkPredictionClassifier:
    """
    Link Prediction Classifier designed to be used with an existing node representation learner.

    Args:
        transform_node (callable): A function to transform node IDs into node features.
        binary_operators (list of str, optional): Binary operators applied on node features to
            produce the corresponding edge feature. Each binary operator will be used to train and
            predict a separate classifier. By default, all binary operators are used.

    """

    _BINARY_OPERATORS = ["h", "avg", "l1", "l2"]

    def __init__(self, transform_node, binary_operators=None):
        self.transform_node = transform_node
        if binary_operators is not None:
            self._binary_operators = binary_operators
        else:
            self._binary_operators = self._BINARY_OPERATORS
        self._classifiers = dict()

    def _transform_node_pairs(self, link_examples):
        return [
            (self.transform_node(src), self.transform_node(dst))
            for src, dst in link_examples
        ]

    def fit(self, link_examples, link_labels, max_iter=500):
        """
        Train a link prediction classifier on the provided positive/negative link examples for each
        binary operator.

        Args:
            link_examples (list of tuple): Tuples of positive and negative links (source, target)
            link_labels (list of int): List of labels corresponding to link examples - 1 for positive,
                0 for negative.
            max_iter (int): Maximum number of training iterations for the classifier.

        """
        node_feature_pairs = self._transform_node_pairs(link_examples)
        for binary_operator in self._binary_operators:
            link_features = [
                _create_link_feature(pair, binary_operator)
                for pair in node_feature_pairs
            ]
            clf = Pipeline(
                steps=[
                    ("sc", StandardScaler()),
                    (
                        "clf",
                        LogisticRegressionCV(
                            Cs=10,
                            cv=10,
                            scoring="roc_auc",
                            verbose=False,
                            max_iter=max_iter,
                        ),
                    ),
                ]
            )
            clf.fit(link_features, link_labels)
            self._classifiers[binary_operator] = clf

    def predict(self, link_examples, binary_operators=None):
        """
        Predict links using trained classifiers.

        Args:
            link_examples (list of tuple): Tuples of links to predict (source, target)
            binary_operators (list of str, optional): If provided, only predict using these binary
                operators regardless of which classifiers have been trained. By default, all trained
                classifiers are used.

        Returns:
            Dict of predictions grouped by binary operator
        """
        node_feature_pairs = self._transform_node_pairs(link_examples)
        predictions = dict()

        if binary_operators is None:
            binary_operators = self._classifiers.keys()

        for binary_operator in binary_operators:
            if binary_operator not in self._classifiers:
                warnings.warn(
                    f"Skipping binary operator '{binary_operator}': "
                    f"no trained classifier available for this operator."
                )
            else:
                classifier = self._classifiers[binary_operator]
                link_features = [
                    _create_link_feature(pair, binary_operator)
                    for pair in node_feature_pairs
                ]
                predictions[binary_operator] = classifier.predict_proba(link_features)

        return predictions

    def evaluate(self, link_examples, link_labels):
        """
        Evaluate trained classifiers using the provided test set.

        Args:
            link_examples (list of tuple): Tuples of test set links to predict (source, target)
            link_labels (list of int): List of labels corresponding to link examples - 1 for positive,
                0 for negative.

        Returns:
            Dict of Scores grouped by binary operator
        """
        predictions = self.predict(link_examples)
        scores = dict()
        for binary_operator, predicted in predictions.items():
            if self._classifiers[binary_operator].classes_[0] == 1:
                scores[binary_operator] = roc_auc_score(link_labels, predicted[:, 0])
            else:
                scores[binary_operator] = roc_auc_score(link_labels, predicted[:, 1])

        return scores
