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

import numpy as np
import pandas as pd
import tensorflow as tf

from stellargraph import StellarGraph, IndexedArray
from stellargraph.mapper import SlidingFeaturesNodeGenerator
from stellargraph.layer import GraphConvolutionLSTM


def test_gcn_lstm_generator():
    nodes = IndexedArray(np.arange(3 * 7).reshape(3, 7) / 21, index=["a", "b", "c"])
    edges = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"]})
    graph = StellarGraph(nodes, edges)

    gen = SlidingFeaturesNodeGenerator(graph, 2, batch_size=3)
    gcn_lstm = GraphConvolutionLSTM(None, None, 2, [4], ["relu", "relu"], generator=gen)

    model = tf.keras.Model(*gcn_lstm.in_out_tensors())

    model.compile("adam", loss="mse")

    history = model.fit(gen.flow(slice(0, 5), target_distance=1))

    predictions = model.predict(gen.flow(slice(5, 7)))

    model2 = tf.keras.Model(*gcn_lstm.in_out_tensors())
    predictions2 = model2.predict(gen.flow(slice(5, 7)))
    np.testing.assert_array_equal(predictions, predictions2)
