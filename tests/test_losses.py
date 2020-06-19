# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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

from stellargraph.losses import *
import numpy as np
import pytest
import tensorflow as tf
from scipy.special import softmax, expit  # sigmoid


def test_graph_log_likelihood():

    batch_rows = 7

    batch_adj = (np.random.random((batch_rows, 1, 100)) > 0.7).astype(np.float32)

    expected_walks = np.random.random((batch_rows, 1, 100)).astype(np.float32)
    scores = np.random.random((batch_rows, 1, 100)).astype(np.float32)

    wys_output = np.concatenate((expected_walks, scores), axis=1)

    actual_loss = graph_log_likelihood(batch_adj, wys_output).numpy()[0]

    sigmoid_scores = 1 / (1 + np.exp(-scores))
    expected_loss = np.abs(
        -expected_walks * np.log(sigmoid_scores)
        - (batch_adj == 0) * np.log(1 - sigmoid_scores)
    )

    expected_loss = expected_loss.sum()

    np.testing.assert_allclose(actual_loss, expected_loss, rtol=0.01)


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 2.0])
def test_self_adversarial_negative_sampling(temperature):
    labels = np.array([1, 0, -2, 0, 1], dtype=np.int32)
    logit_scores = np.array([1.2, -2.3, 0.0, 4.5, -0.67], dtype=np.float32)
    scores = expit(logit_scores)

    loss_func = SelfAdversarialNegativeSampling(temperature)
    actual_loss = loss_func(tf.constant(labels), tf.constant(logit_scores))

    def loss_part(score, label):
        # equations (5) and (6) in http://arxiv.org/abs/1902.10197
        if label == 1:
            # positive edge
            return -np.log(score)

        # Negative sample. The
        relevant = scores[np.where(labels == label)]
        numer = np.exp(temperature * score)
        denom = np.sum(np.exp(temperature * relevant))

        # sigmoid(-x) == 1 - sigmoid(x)
        return -np.log(1 - score) * numer / denom

    expected_loss = np.mean(
        [loss_part(score, label) for score, label in zip(scores, labels)]
    )
    assert actual_loss.numpy() == pytest.approx(expected_loss, rel=1e-6)


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 2.0])
def test_self_adversarial_negative_sampling_extreme(temperature):
    # validate numerical stability
    val = 1e10

    labels = np.array([1, 1, 0, 0], dtype=np.int32)
    logit_scores = np.array([-val, val, -val, val], dtype=np.float32)
    scores = expit(logit_scores)

    loss_func = SelfAdversarialNegativeSampling(temperature)
    actual_loss = loss_func(tf.constant(labels), tf.constant(logit_scores))

    def log_sigmoid(x):
        return -np.logaddexp(0, -x)

    neg_weights = np.exp(temperature * scores[2:4])

    expected_loss = np.array(
        [
            -log_sigmoid(logit_scores[0]),
            -log_sigmoid(logit_scores[1]),
            neg_weights[0] / neg_weights.sum() * -log_sigmoid(-logit_scores[2]),
            neg_weights[1] / neg_weights.sum() * -log_sigmoid(-logit_scores[3]),
        ],
        dtype=np.float32,
    )

    assert actual_loss == np.mean(expected_loss)
