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


@experimental(reason="lack of unit tests", issues=[804])
def graph_log_likelihood(batch_adj, wys_output):
    """
    Computes the graph log likelihood loss function as in https://arxiv.org/abs/1710.09599.

    This is different to most keras loss functions in that it doesn't directly compare predicted values to expected
    values. It uses `wys_output` which contains the dot products of embeddings and expected random walks,
    and part of the adjacency matrix `batch_adj` to calculate how well the node embeddings capture the graph
    structure in some sense.

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
