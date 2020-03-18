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

import tensorflow as tf

from .core.experimental import experimental


def graph_log_likelihood(batch_adj, wys_output):
    """
    Computes the graph log likelihood loss function as in https://arxiv.org/abs/1710.09599

    Args:
        batch_adj: tensor with shape (batch_rows, 1, num_nodes) containing rows of the adjacency matrix
        wys_output: tensor with shape (batch_rows, 2, num_nodes) containing the embedding outer product
            scores (with shape (batch_rows, 1, num_nodes) and attentive expected random walk
            (with shape (batch_rows, 1, num_nodes) concatenated.
    Returns:
        the graph log likelihood loss for the batch
    """

    batch_adj = tf.squeeze(batch_adj, axis=1)

    expected_walks = tf.gather(wys_output, [0], axis=1)
    scores = tf.gather(wys_output, [1], axis=1)

    adj_mask = tf.cast((batch_adj == 0), "float32")

    loss = tf.math.reduce_sum(
        tf.abs(
            expected_walks * tf.math.log_sigmoid(scores)
            - adj_mask * (tf.math.log_sigmoid(scores) - scores)
        )
    )

    return tf.expand_dims(loss, 0)
