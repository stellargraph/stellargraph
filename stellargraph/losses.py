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

__all__ = [
    "graph_log_likelihood",
    "SelfAdversarialNegativeSampling",
]

import tensorflow as tf

from .core.experimental import experimental


@experimental(reason="lack of unit tests", issues=[804])
def graph_log_likelihood(batch_adj, wys_output):
    """
    Computes the graph log likelihood loss function as in https://arxiv.org/abs/1710.09599.

    This is different to most Keras loss functions in that it doesn't directly compare predicted values to expected
    values. It uses `wys_output` which contains the dot products of embeddings and expected random walks,
    and part of the adjacency matrix `batch_adj` to calculate how well the node embeddings capture the graph
    structure in some sense.

    .. seealso: The :class:`.WatchYourStep` model, for which this loss function is designed.

    Args:
        batch_adj: tensor with shape ``batch_rows x 1 x num_nodes`` containing rows of the adjacency matrix
        wys_output: tensor with shape ``batch_rows x 2 x num_nodes`` containing the embedding outer product
            scores with shape ``batch_rows x 1 x num_nodes`` and attentive expected random walk
            with shape ``batch_rows x 1, num_nodes`` concatenated.
    Returns:
        the graph log likelihood loss for the batch
    """

    expected_walks = tf.gather(wys_output, [0], axis=1)
    scores = tf.gather(wys_output, [1], axis=1)

    adj_mask = tf.cast((batch_adj == 0), "float32")

    log_sigmoid = tf.math.log_sigmoid(scores)
    log1m_sigmoid = log_sigmoid - scores  # log(1 - σ(scores)), simplified
    matrix = -expected_walks * log_sigmoid - adj_mask * log1m_sigmoid
    loss = tf.math.reduce_sum(tf.abs(matrix))

    return tf.expand_dims(loss, 0)


class SelfAdversarialNegativeSampling(tf.keras.losses.Loss):
    """
    Computes the self-adversarial binary cross entropy for negative sampling, from [1].

    [1] Z. Sun, Z.-H. Deng, J.-Y. Nie, and J. Tang, “RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space,” `arXiv:1902.10197 <http://arxiv.org/abs/1902.10197>`_

    Args:
        temperature (float, optional): a scaling factor for the weighting of negative samples
    """

    def __init__(
        self, temperature=1.0, name="self_adversarial_negative_sampling",
    ):
        self._temperature = temperature
        super().__init__(name=name)

    def call(self, labels, logit_scores):
        """
        Args:
            labels: tensor of integer labels for each row, either 1 for a true sample, or any value <= 0 for negative samples. Negative samples with identical labels are combined for the softmax normalisation.
            logit_scores: tensor of scores for each row in logits
        """

        scores = tf.math.sigmoid(logit_scores)

        if labels.dtype != tf.int32:
            labels = tf.cast(labels, tf.int64)

        flipped_labels = -labels

        exp_scores = tf.math.exp(self._temperature * scores)
        sums = tf.math.unsorted_segment_sum(
            exp_scores, flipped_labels, tf.reduce_max(flipped_labels) + 1
        )

        denoms = tf.gather(sums, tf.maximum(flipped_labels, 0))

        # adversarial sampling shouldn't influence the gradient/update
        negative_weights = tf.stop_gradient(exp_scores / denoms)

        loss_elems = tf.where(
            labels > 0,
            -tf.math.log_sigmoid(logit_scores),
            -tf.math.log_sigmoid(-logit_scores) * negative_weights,
        )

        return tf.reduce_mean(loss_elems, axis=-1)
