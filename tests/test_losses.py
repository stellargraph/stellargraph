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

from stellargraph.losses import graph_log_likelihood
import numpy as np


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
