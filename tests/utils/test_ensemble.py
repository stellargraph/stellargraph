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
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy


def example_graph_1(feature_size=None):
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2), (5, 6), (1, 5)]
    G.add_nodes_from([1, 2, 3, 4, 5, 6], label="default")
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
    prediction = layers.Dense(units=2, activation="softmax")(x_out)

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

    ens = Ensemble(keras_model, n_estimators=14, n_predictions=10)

    assert len(ens.models) == 14
    assert ens.n_estimators == 14
    assert ens.n_predictions == 10


def test_compile():

    graph = example_graph_1(feature_size=10)

    base_model, keras_model, generator, train_gen = create_graphSAGE_model(graph)

    ens = Ensemble(keras_model, n_estimators=5, n_predictions=12)

    # These are actually raised by keras but I added a check just to make sure
    with pytest.raises(ValueError):
        ens.compile(optimizer=Adam(), loss=None, weighted_metrics=["acc"])

    with pytest.raises(ValueError):  # must specify the optimizer to use
        ens.compile(
            optimizer=None, loss=categorical_crossentropy, weighted_metrics=["acc"]
        )

    with pytest.raises(
        ValueError
    ):  # The metric is made up so it should raise ValueError
        ens.compile(
            optimizer=Adam(),
            loss=categorical_crossentropy,
            weighted_metrics=["f1_accuracy"],
        )


def test_fit_generator():

    train_data = np.array([1, 2])
    train_targets = np.array([[1, 0], [0, 1]])

    graph = example_graph_1(feature_size=10)

    base_model, keras_model, generator, train_gen = create_graphSAGE_model(graph)

    ens = Ensemble(keras_model, n_estimators=2, n_predictions=1)

    ens.compile(
        optimizer=Adam(), loss=categorical_crossentropy, weighted_metrics=["acc"]
    )

    # Specifying train_data and train_targets, implies the use of bagging so train_gen would
    # be of the wrong type for this call to fit_generator.
    with pytest.raises(ValueError):
        ens.fit_generator(
            train_gen,
            train_data=train_data,
            train_targets=train_targets,
            epochs=20,
            validation_generator=train_gen,
            verbose=0,
            shuffle=False,
        )

    with pytest.raises(ValueError):
        ens.fit_generator(
            generator=generator,
            train_data=train_data,
            train_targets=None,  # Should not be None
            epochs=20,
            validation_generator=train_gen,
            verbose=0,
            shuffle=False,
        )

    with pytest.raises(ValueError):
        ens.fit_generator(
            generator=generator,
            train_data=None,
            train_targets=None,
            epochs=20,
            validation_generator=None,
            verbose=0,
            shuffle=False,
        )

    with pytest.raises(ValueError):
        ens.fit_generator(
            generator=generator,
            train_data=train_data,
            train_targets=train_targets,
            epochs=20,
            validation_generator=None,
            verbose=0,
            shuffle=False,
            bag_size=-1,  # should be positive integer smaller than or equal to len(train_data) or None
        )

    with pytest.raises(ValueError):
        ens.fit_generator(
            generator=generator,
            train_data=train_data,
            train_targets=train_targets,
            epochs=20,
            validation_generator=None,
            verbose=0,
            shuffle=False,
            bag_size=10,  # larger than the number of training points
        )


def test_evaluate_generator():

    train_data = np.array([1, 2])
    train_targets = np.array([[1, 0], [0, 1]])

    test_data = np.array([3, 4, 5])
    test_targets = np.array([[1, 0], [0, 1], [0, 1]])

    graph = example_graph_1(feature_size=10)

    base_model, keras_model, generator, train_gen = create_graphSAGE_model(graph)

    ens = Ensemble(keras_model, n_estimators=2, n_predictions=1)

    ens.compile(
        optimizer=Adam(), loss=categorical_crossentropy, weighted_metrics=["acc"]
    )

    # Check that passing invalid parameters is handled correctly. We will not check error handling for those parameters
    # that Keras will be responsible for.
    with pytest.raises(ValueError):
        ens.evaluate_generator(
            generator=generator, test_data=test_data, test_targets=test_targets
        )

    with pytest.raises(ValueError):
        ens.evaluate_generator(
            generator=generator, test_data=test_data, test_targets=None,  # must give test_targets
        )

    with pytest.raises(ValueError):
        ens.evaluate_generator(
            generator=generator.flow(test_data, test_targets),
            test_data=test_data,
            test_targets=test_targets,
        )

    # We won't train the model instead use the initial random weights to test
    # the evaluate_generator method.
    test_metrics_mean, test_metrics_std = ens.evaluate_generator(
        generator.flow(test_data, test_targets)
    )

    assert len(test_metrics_mean) == len(test_metrics_std)
    assert len(test_metrics_mean.shape) == 1
    assert len(test_metrics_std.shape) == 1
