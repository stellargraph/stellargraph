# -*- coding: utf-8 -*-
#
# Copyright 2018-2019 Data61, CSIRO
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

from stellargraph.utils.saliency_maps import GraphPreProcessingLayer
from stellargraph.core.utils import GCN_Aadj_feats_op
from stellargraph import StellarGraph
import pytest
import networkx as nx
import keras.backend as K
import numpy as np
from scipy.sparse import coo_matrix

def test_preprocessing_layer():
    #check whether the layer implementation is equivalent to the numpy implementation.
    feature = np.array([[1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]])
    adj = np.array([[0, 0, 0, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0]])
    graph_norm_layer = GraphPreProcessingLayer(
        output_dim=(4, 4)
    )
    adj_t = K.variable(adj, dtype="float32")
    layer_normalized_adj = K.eval(graph_norm_layer(adj_t))
    _, np_normalized_adj = GCN_Aadj_feats_op(feature, coo_matrix(adj), method='gcn')

    print(layer_normalized_adj)

    print(np_normalized_adj.todense())
    assert pytest.approx(layer_normalized_adj, np_normalized_adj.todense())

