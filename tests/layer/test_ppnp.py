# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
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

from stellargraph.layer import PPNP
from stellargraph.mapper import FullBatchNodeGenerator, FullBatchLinkGenerator
from stellargraph import StellarGraph
from stellargraph.core.utils import PPNP_Aadj_feats_op

import networkx as nx
import pandas as pd
import numpy as np
from tensorflow import keras
import pytest
from ..test_utils.graphs import create_graph_features


def test_PPNP_edge_cases():
    G, features = create_graph_features()
    adj = G.to_adjacency_matrix()
    features, adj = PPNP_Aadj_feats_op(features, adj)

    ppnp_sparse_failed = False
    try:
        generator = FullBatchNodeGenerator(G, sparse=True, method="ppnp")
    except ValueError as e:
        ppnp_sparse_failed = True
    assert ppnp_sparse_failed

    generator = FullBatchNodeGenerator(G, sparse=False, method="ppnp")

    try:
        ppnpModel = PPNP([2, 2], generator=generator, activations=["relu"], dropout=0.5)
    except ValueError as e:
        error = e
    assert str(error) == "The number of layers should equal the number of activations"

    try:
        ppnpModel = PPNP([2], generator=[0, 1], activations=["relu"], dropout=0.5)
    except TypeError as e:
        error = e
    assert str(error) == "Generator should be a instance of FullBatchNodeGenerator"


def test_PPNP_apply_dense():
    G, features = create_graph_features()
    adj = G.to_adjacency_matrix()
    features, adj = PPNP_Aadj_feats_op(features, adj)
    adj = adj[None, :, :]

    generator = FullBatchNodeGenerator(G, sparse=False, method="ppnp")
    ppnpModel = PPNP([2], generator=generator, activations=["relu"], dropout=0.5)

    x_in, x_out = ppnpModel.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, adj])
    assert preds_1.shape == (1, 2, 2)

    # Check fit method
    preds_2 = model.predict(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)
