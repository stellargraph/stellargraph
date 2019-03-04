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

import pytest
import networkx as nx

from stellargraph import StellarGraph
from stellargraph.layer import GraphSAGE, GCN, GAT, HinSAGE
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.data.converter import *
from stellargraph.utils import Ensemble

from keras import layers, Model


def example_graph_1(feature_size=None):
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    G.add_nodes_from([1, 2, 3, 4], label="default")
    G.add_edges_from(elist, label="default")

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = np.ones(feature_size)
        return StellarGraph(G, node_features="feature")

    else:
        return StellarGraph(G)


def create_graphSAGE_model(graph):
    generator = GraphSAGENodeGenerator(graph, batch_size=2, num_samples=[2, 2])
    train_gen = generator.flow([1, 2])

    base_model = GraphSAGE(
        layer_sizes=[8, 8], generator=train_gen, bias=True, dropout=0.5
    )

    x_inp, x_out = base_model.default_model(flatten_output=True)
    prediction = layers.Dense(units=1, activation="sigmoid")(x_out)

    keras_model = Model(inputs=x_inp, outputs=prediction)

    return base_model, keras_model, generator, train_gen


#
# Test for class Ensemble instance creation with invalid parameters given.
#
def test_ensemble_init_parameters():

    graph = example_graph_1(feature_size=10)

    base_model, keras_model, generator, train_gen = create_graphSAGE_model(graph)

    # Test mixed types
    with pytest.raises(ValueError):
        Ensemble(base_model, n_estimators=3, n_predictions=3)

    with pytest.raises(ValueError):
        Ensemble(keras_model, n_estimators=1, n_predictions=0)

    with pytest.raises(ValueError):
        Ensemble(keras_model, n_estimators=1, n_predictions=-3)

    with pytest.raises(ValueError):
        Ensemble(keras_model, n_estimators=1, n_predictions=1.7)

    with pytest.raises(ValueError):
        Ensemble(keras_model, n_estimators=0, n_predictions=11)

    with pytest.raises(ValueError):
        Ensemble(keras_model, n_estimators=-8, n_predictions=11)

    with pytest.raises(ValueError):
        Ensemble(keras_model, n_estimators=2.5, n_predictions=11)
