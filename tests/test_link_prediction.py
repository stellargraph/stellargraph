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

from stellargraph.link_prediction import LinkPredictionClassifier, _get_operator
import pytest
import numpy as np


def transform_node(node):
    return np.array([node])


class MockClassifier:
    """
    Helper class to mock a scikit-learn classifier. This assumes that `clf.predict_proba` will be
    called to make predictions.

    """

    def __init__(self, predict_proba):
        self.predict_proba = predict_proba
        self.classes_ = [0, 1]


@pytest.mark.parametrize("binary_operators", [None, ["h"], ["avg", "l1", "l2"]])
def test_link_prediction_classifier_fit(binary_operators):
    clf = LinkPredictionClassifier(transform_node, binary_operators)
    n = 20
    link_examples = [(i, i + 1) for i in range(n)]
    link_labels = [i % 2 for i in range(n)]
    clf.fit(link_examples, link_labels)

    if binary_operators is None:
        binary_operators = ["h", "avg", "l1", "l2"]

    assert set(clf._classifiers.keys()) == set(binary_operators)


@pytest.mark.parametrize("binary_operators", [None, ["h"], ["avg", "l1", "l2"]])
def test_link_prediction_classifier_predict(binary_operators):
    clf = LinkPredictionClassifier(transform_node, binary_operators)

    if binary_operators is None:
        binary_operators = ["h", "avg", "l1", "l2"]

    link_examples = [(1, 2), (2, 3)]
    clf._classifiers = {op: MockClassifier(lambda x: x) for op in binary_operators}
    expected = {
        op: [_get_operator(op)(src, dst) for src, dst in link_examples]
        for op in binary_operators
    }

    predictions = clf.predict(link_examples)
    assert predictions == expected


def test_link_prediction_classifier_predict_transform_node():
    clf = LinkPredictionClassifier(transform_node)
    clf._classifiers = {"avg": MockClassifier(lambda x: x)}

    link_examples = [(1, 2), (2, 3)]

    # add 2 to each node when transforming
    predictions = clf.predict(link_examples, transform_node=lambda x: x + 2)

    assert predictions == {
        "avg": [np.mean(np.array(link) + 2) for link in link_examples]
    }


def test_link_prediction_classifier_predict_binary_operators():
    clf = LinkPredictionClassifier(transform_node, ["h"])
    clf._classifiers = {
        "h": MockClassifier(lambda x: x),
        "avg": MockClassifier(lambda x: [i + 2 for i in x]),
    }

    link_examples = [(1, 2), (2, 3)]

    # only use avg
    predictions = clf.predict(link_examples, binary_operators=["avg"])

    assert predictions == {
        "avg": [np.mean(np.array(link)) + 2 for link in link_examples]
    }


@pytest.mark.parametrize("binary_operators", [None, ["h"], ["avg", "l1", "l2"]])
def test_link_prediction_classifier_evaluate_roc_auc(binary_operators):
    clf = LinkPredictionClassifier(transform_node, binary_operators)

    if binary_operators is None:
        binary_operators = ["h", "avg", "l1", "l2"]

    link_examples = [(1, 2), (2, 3)]
    link_labels = [1, 0]
    clf._classifiers = {
        # mock classifier always predicts zero (first class)
        op: MockClassifier(lambda x: np.array([[1, 0] for _ in x]))
        for op in binary_operators
    }

    # roc auc should be 0.5 given one example for each class
    expected = {op: 0.5 for op in binary_operators}
    scores = clf.evaluate_roc_auc(link_examples, link_labels)

    assert scores == expected
