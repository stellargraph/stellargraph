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
def graph_log_likelihood(y_true, y_pred):
    """
    """
    batch_adj = tf.gather(y_true, [0], axis=1)

    expected_walks = tf.gather(y_pred, [0], axis=1)
    sigmoids = tf.gather(y_pred, [1], axis=1)

    adj_mask = tf.cast((batch_adj == 0), "float32")

    loss = tf.math.reduce_sum(
        tf.abs(
            -(expected_walks * tf.math.log(sigmoids + 1e-9))
            - adj_mask * tf.math.log(1 - sigmoids + 1e-9)
        )
    )

    return tf.expand_dims(loss, 0)
