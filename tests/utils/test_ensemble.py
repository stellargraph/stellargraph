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
from stellargraph.layer import (
    GraphSAGE,
    GCN,
    GAT,
    HinSAGE,
    link_classification,
    link_regression,
)
from stellargraph.mapper import (
    GraphSAGENodeGenerator,
    FullBatchNodeGenerator,
    HinSAGENodeGenerator,
    GraphSAGELinkGenerator,
    HinSAGELinkGenerator,
)
from stellargraph.data.converter import *
from stellargraph.data import UnsupervisedSampler
from stellargraph.utils import Ensemble, BaggingEnsemble

from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow import keras

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

def create_graphSAGE_model(graph, link_prediction=False):

    if link_prediction:
        # We are going to train on the original graph
        generator = GraphSAGELinkGenerator(graph, batch_size=2, num_samples=[2, 2])
        edge_ids_train = np.array([[1, 2], [2, 3], [1, 3]])
        train_gen = generator.flow(edge_ids_train, np.array([1, 1, 0]))
    else:
        generator = GraphSAGENodeGenerator(graph, batch_size=2, num_samples=[2, 2])
        train_gen = generator.flow([1, 2], np.array([[1, 0], [0, 1]]))

    # if link_prediction:
    #     edge_ids_train = np.array([[1, 2], [2, 3], [1, 3]])
    #     train_gen = generator.flow(edge_ids_train, np.array([1, 1, 0]))
    # else:
    #     train_gen = generator.flow([1, 2], np.array([[1, 0], [0, 1]]))

    base_model = GraphSAGE(
        layer_sizes=[8, 8], generator=train_gen, bias=True, dropout=0.5
    )

    if link_prediction:
        # Expose input and output sockets of graphsage, for source and destination nodes:
        x_inp_src, x_out_src = base_model.node_model()
        x_inp_dst, x_out_dst = base_model.node_model()
        # re-pack into a list where (source, destination) inputs alternate, for link inputs:
        x_inp = [x for ab in zip(x_inp_src, x_inp_dst) for x in ab]
        # same for outputs:
        x_out = [x_out_src, x_out_dst]

        prediction = link_classification(
            output_dim=1, output_act="relu", edge_embedding_method="ip"
        )(x_out)

        keras_model = Model(inputs=x_inp, outputs=prediction)
    else:
        x_inp, x_out = base_model.node_model()
        prediction = layers.Dense(units=2, activation="softmax")(x_out)

        keras_model = Model(inputs=x_inp, outputs=prediction)

    return base_model, keras_model, generator, train_gen

def create_Unsupervised_graphSAGE_model(graph):
    generator = GraphSAGELinkGenerator(graph, batch_size=2, num_samples=[2, 2])
    unsupervisedSamples = UnsupervisedSampler(
        graph, nodes=graph.nodes(), length=3, number_of_walks=2
    )
    train_gen = generator.flow(unsupervisedSamples)

    base_model = GraphSAGE(
        layer_sizes=[8, 8], generator=train_gen, bias=True, dropout=0.5
    )
    x_inp, x_out = base_model.build()

    prediction = link_classification(
        output_dim=1, output_act="relu", edge_embedding_method="ip"
    )(x_out)

    keras_model = Model(inputs=x_inp, outputs=prediction)
    return base_model, keras_model, generator, train_gen

def create_HinSAGE_model(graph, link_prediction=False):

    if link_prediction:
        generator = HinSAGELinkGenerator(graph, batch_size=2, num_samples=[2, 1])
        edge_ids_train = np.array([[1, 2], [2, 3], [1, 3]])
        train_gen = generator.flow(edge_ids_train, np.array([1, 1, 0]))
    else:
        generator = HinSAGENodeGenerator(graph, batch_size=2, num_samples=[2, 2])
        train_gen = generator.flow([1, 2], np.array([[1, 0], [0, 1]]))

    base_model = HinSAGE(
        layer_sizes=[8, 8], generator=train_gen, bias=True, dropout=0.5
    )

    if link_prediction:
        # Define input and output sockets of hinsage:
        x_inp, x_out = base_model.build()

        # Final estimator layer
        prediction = link_regression(edge_embedding_method="ip")(x_out)
    else:
        x_inp, x_out = base_model.build()
        prediction = layers.Dense(units=2, activation="softmax")(x_out)

    keras_model = Model(inputs=x_inp, outputs=prediction)

    return base_model, keras_model, generator, train_gen

def create_GCN_model(graph):

    generator = FullBatchNodeGenerator(graph)
    train_gen = generator.flow([1, 2], np.array([[1, 0], [0, 1]]))

    base_model = GCN(
        layer_sizes=[8, 2],
        generator=generator,
        bias=True,
        dropout=0.5,
        activations=["elu", "softmax"],
    )

    x_inp, x_out = base_model.node_model()

    keras_model = Model(inputs=x_inp, outputs=x_out)

    return base_model, keras_model, generator, train_gen

def create_GAT_model(graph):

    generator = FullBatchNodeGenerator(graph, sparse=False)
    train_gen = generator.flow([1, 2], np.array([[1, 0], [0, 1]]))

    base_model = GAT(
        layer_sizes=[8, 8, 2],
        generator=generator,
        bias=True,
        in_dropout=0.5,
        attn_dropout=0.5,
        activations=["elu", "elu", "softmax"],
        normalize=None,
    )

    x_inp, x_out = base_model.node_model()

    keras_model = Model(inputs=x_inp, outputs=x_out)

    return base_model, keras_model, generator, train_gen

#
# Test for class Ensemble instance creation with invalid parameters given.
#
def test_ensemble_init_parameters():

    graph = example_graph_1(feature_size=10)

    base_model, keras_model, generator, train_gen = create_graphSAGE_model(graph)

    # base_model, keras_model, generator, train_gen
    gnn_models = [
        create_graphSAGE_model(graph),
        create_Unsupervised_graphSAGE_model(graph),
        create_HinSAGE_model(graph),
        create_graphSAGE_model(graph, link_prediction=True),
        create_HinSAGE_model(graph, link_prediction=True),
        create_GCN_model(graph),
        create_GAT_model(graph),
    ]

    for gnn_model in gnn_models:
        base_model = gnn_model[0]
        keras_model = gnn_model[1]

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

        ens = Ensemble(keras_model, n_estimators=7, n_predictions=10)

        assert len(ens.models) == 7
        assert ens.n_estimators == 7
        assert ens.n_predictions == 10

        #
        # Repeat for BaggingEnsemble
        # Test mixed types
        with pytest.raises(ValueError):
            BaggingEnsemble(base_model, n_estimators=3, n_predictions=3)

        with pytest.raises(ValueError):
            BaggingEnsemble(keras_model, n_estimators=1, n_predictions=0)

        with pytest.raises(ValueError):
            BaggingEnsemble(keras_model, n_estimators=1, n_predictions=-3)

        with pytest.raises(ValueError):
            BaggingEnsemble(keras_model, n_estimators=1, n_predictions=1.7)

        with pytest.raises(ValueError):
            BaggingEnsemble(keras_model, n_estimators=0, n_predictions=11)

        with pytest.raises(ValueError):
            BaggingEnsemble(keras_model, n_estimators=-8, n_predictions=11)

        with pytest.raises(ValueError):
            BaggingEnsemble(keras_model, n_estimators=2.5, n_predictions=11)

        ens = BaggingEnsemble(keras_model, n_estimators=7, n_predictions=10)

        assert len(ens.models) == 7
        assert ens.n_estimators == 7
        assert ens.n_predictions == 10

def test_compile():

    graph = example_graph_1(feature_size=10)

    # base_model, keras_model, generator, train_gen
    gnn_models = [
        create_graphSAGE_model(graph),
        create_Unsupervised_graphSAGE_model(graph),
        create_HinSAGE_model(graph),
        create_graphSAGE_model(graph, link_prediction=True),
        create_HinSAGE_model(graph, link_prediction=True),
        create_GCN_model(graph),
        create_GAT_model(graph),
    ]

    for gnn_model in gnn_models:
        keras_model = gnn_model[1]
        ens = Ensemble(keras_model, n_estimators=2, n_predictions=5)

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

        #
        # Repeat for BaggingEnsemble
        ens = BaggingEnsemble(keras_model, n_estimators=2, n_predictions=5)

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

def test_Ensemble_fit_generator():

    graph = example_graph_1(feature_size=10)

    # base_model, keras_model, generator, train_gen
    gnn_models = [
        create_graphSAGE_model(graph),
        create_HinSAGE_model(graph),
        create_GCN_model(graph),
        create_GAT_model(graph),
    ]

    for gnn_model in gnn_models:
        keras_model = gnn_model[1]
        generator = gnn_model[2]
        train_gen = gnn_model[3]

        ens = Ensemble(keras_model, n_estimators=2, n_predictions=1)

        ens.compile(
            optimizer=Adam(), loss=categorical_crossentropy, weighted_metrics=["acc"]
        )

        ens.fit_generator(train_gen, epochs=10, verbose=0, shuffle=False)

        with pytest.raises(ValueError):
            ens.fit_generator(
                generator=generator,  # wrong type
                epochs=10,
                validation_data=train_gen,
                verbose=0,
                shuffle=False,
            )

def test_unsupervised_Ensemble_fit_generator():

    graph = example_graph_1(feature_size=10)

    # base_model, keras_model, generator, train_gen
    gnn_models = [create_graphSAGE_model(graph)]

    for gnn_model in gnn_models:
        keras_model = gnn_model[1]
        generator = gnn_model[2]
        train_gen = gnn_model[3]

        ens = Ensemble(keras_model, n_estimators=2, n_predictions=1)
        ens.compile(
            optimizer=Adam(), loss=binary_crossentropy, weighted_metrics=["acc"]
        )
        ens.fit_generator(train_gen, epochs=1, verbose=0, shuffle=False)

        with pytest.raises(ValueError):
            ens.fit_generator(
                generator=generator,  # wrong type
                epochs=10,
                validation_data=train_gen,
                verbose=0,
                shuffle=False,
            )

def test_BaggingEnsemble_fit_generator():

    train_data = np.array([1, 2])
    train_targets = np.array([[1, 0], [0, 1]])

    graph = example_graph_1(feature_size=10)

    # base_model, keras_model, generator, train_gen
    gnn_models = [
        create_graphSAGE_model(graph),
        create_HinSAGE_model(graph),
        create_GCN_model(graph),
        create_GAT_model(graph),
    ]

    for gnn_model in gnn_models:
        keras_model = gnn_model[1]
        generator = gnn_model[2]
        train_gen = gnn_model[3]

        ens = BaggingEnsemble(keras_model, n_estimators=2, n_predictions=1)

        ens.compile(
            optimizer=Adam(), loss=categorical_crossentropy, weighted_metrics=["acc"]
        )

        ens.fit_generator(
            generator=generator,
            train_data=train_data,
            train_targets=train_targets,
            epochs=10,
            validation_data=train_gen,
            verbose=0,
            shuffle=False,
        )

        # This is a BaggingEnsemble so the generator in the below call is of the wrong type.
        with pytest.raises(ValueError):
            ens.fit_generator(
                train_gen,
                train_data=train_data,
                train_targets=train_targets,
                epochs=10,
                verbose=0,
                shuffle=False,
            )

        with pytest.raises(ValueError):
            ens.fit_generator(
                generator=generator,
                train_data=train_data,
                train_targets=None,  # Should not be None
                epochs=10,
                validation_data=train_gen,
                verbose=0,
                shuffle=False,
            )

        with pytest.raises(ValueError):
            ens.fit_generator(
                generator=generator,
                train_data=None,
                train_targets=None,
                epochs=10,
                validation_data=None,
                verbose=0,
                shuffle=False,
            )

        with pytest.raises(ValueError):
            ens.fit_generator(
                generator=generator,
                train_data=train_data,
                train_targets=train_targets,
                epochs=10,
                validation_data=None,
                verbose=0,
                shuffle=False,
                bag_size=-1,  # should be positive integer smaller than or equal to len(train_data) or None
            )

        with pytest.raises(ValueError):
            ens.fit_generator(
                generator=generator,
                train_data=train_data,
                train_targets=train_targets,
                epochs=10,
                validation_data=None,
                verbose=0,
                shuffle=False,
                bag_size=10,  # larger than the number of training points
            )

def test_unsupervised_BaggingEnsemble_fit_generator():
    train_data = np.array([1, 2])
    train_targets = np.array([[1, 0], [0, 1]])
    graph = example_graph_1(feature_size=10)

    # base_model, keras_model, generator, train_gen
    gnn_models = [create_graphSAGE_model(graph)]

    for gnn_model in gnn_models:
        keras_model = gnn_model[1]
        generator = gnn_model[2]
        train_gen = gnn_model[3]

        ens = BaggingEnsemble(keras_model, n_estimators=2, n_predictions=1)
        ens.compile(
            optimizer=Adam(), loss=binary_crossentropy, weighted_metrics=["acc"]
        )

        ens.fit_generator(
            generator=generator,
            train_data=train_data,
            train_targets=train_targets,
            epochs=1,
            validation_data=train_gen,
            verbose=0,
            shuffle=False,
        )

        # This is a BaggingEnsemble so the generator in the below call is of the wrong type.
        with pytest.raises(ValueError):
            ens.fit_generator(
                train_gen,
                train_data=train_data,
                train_targets=train_targets,
                epochs=10,
                verbose=0,
                shuffle=False,
            )

        with pytest.raises(ValueError):
            ens.fit_generator(
                generator=generator,
                train_data=train_data,
                train_targets=None,  # Should not be None
                epochs=10,
                validation_data=train_gen,
                verbose=0,
                shuffle=False,
            )

        with pytest.raises(ValueError):
            ens.fit_generator(
                generator=generator,
                train_data=None,
                train_targets=None,
                epochs=10,
                validation_data=None,
                verbose=0,
                shuffle=False,
            )

        with pytest.raises(ValueError):
            ens.fit_generator(
                generator=generator,
                train_data=train_data,
                train_targets=train_targets,
                epochs=10,
                validation_data=None,
                verbose=0,
                shuffle=False,
                bag_size=-1,  # should be positive integer smaller than or equal to len(train_data) or None
            )

        with pytest.raises(ValueError):
            ens.fit_generator(
                generator=generator,
                train_data=train_data,
                train_targets=train_targets,
                epochs=10,
                validation_data=None,
                verbose=0,
                shuffle=False,
                bag_size=10,  # larger than the number of training points
            )

def test_evaluate_generator():

    test_data = np.array([3, 4, 5])
    test_targets = np.array([[1, 0], [0, 1], [0, 1]])

    graph = example_graph_1(feature_size=5)

    # base_model, keras_model, generator, train_gen
    gnn_models = [
        create_graphSAGE_model(graph),
        create_HinSAGE_model(graph),
        create_GCN_model(graph),
        create_GAT_model(graph),
    ]

    for gnn_model in gnn_models:
        keras_model = gnn_model[1]
        generator = gnn_model[2]

        ens = Ensemble(keras_model, n_estimators=2, n_predictions=1)

        ens.compile(
            optimizer=Adam(), loss=categorical_crossentropy, weighted_metrics=["acc"]
        )

        # Check that passing invalid parameters is handled correctly. We will not check error handling for those
        # parameters that Keras will be responsible for.
        with pytest.raises(ValueError):
            ens.evaluate_generator(
                generator=generator, test_data=test_data, test_targets=test_targets
            )

        with pytest.raises(ValueError):
            ens.evaluate_generator(
                generator=generator,
                test_data=test_data,
                test_targets=None,  # must give test_targets
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

        #
        # Repeat for BaggingEnsemble

        ens = BaggingEnsemble(keras_model, n_estimators=2, n_predictions=1)

        ens.compile(
            optimizer=Adam(), loss=categorical_crossentropy, weighted_metrics=["acc"]
        )

        # Check that passing invalid parameters is handled correctly. We will not check error handling for those
        # parameters that Keras will be responsible for.
        with pytest.raises(ValueError):
            ens.evaluate_generator(
                generator=generator, test_data=test_data, test_targets=test_targets
            )

        with pytest.raises(ValueError):
            ens.evaluate_generator(
                generator=generator,
                test_data=test_data,
                test_targets=None,  # must give test_targets
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

def test_predict_generator():

    # test_data = np.array([[0, 0], [1, 1], [0.8, 0.8]])
    test_data = np.array([4, 5, 6])
    test_targets = np.array([[1, 0], [0, 1], [0, 1]])

    graph = example_graph_1(feature_size=2)

    # base_model, keras_model, generator, train_gen
    gnn_models = [
        create_graphSAGE_model(graph),
        create_HinSAGE_model(graph),
        create_GCN_model(graph),
        create_GAT_model(graph),
    ]

    for i, gnn_model in enumerate(gnn_models):
        keras_model = gnn_model[1]
        generator = gnn_model[2]

        ens = Ensemble(keras_model, n_estimators=2, n_predictions=2)

        ens.compile(
            optimizer=Adam(), loss=categorical_crossentropy, weighted_metrics=["acc"]
        )

        test_gen = generator.flow(test_data)
        # Check that passing invalid parameters is handled correctly. We will not check error handling for those
        # parameters that Keras will be responsible for.
        with pytest.raises(ValueError):
            ens.predict_generator(generator=test_gen, predict_data=test_data)

        # We won't train the model instead use the initial random weights to test
        # the evaluate_generator method.
        test_predictions = ens.predict_generator(test_gen, summarise=True)

        print("test_predictions shape {}".format(test_predictions.shape))
        if i > 1:
            # GAT and GCN are full batch so the batch dimension is 1
            assert len(test_predictions) == 1
            assert test_predictions.shape[1] == test_targets.shape[0]
        else:
            assert len(test_predictions) == len(test_data)
        assert test_predictions.shape[-1] == test_targets.shape[-1]

        test_predictions = ens.predict_generator(test_gen, summarise=False)

        assert test_predictions.shape[0] == ens.n_estimators
        assert test_predictions.shape[1] == ens.n_predictions
        if i > 1:
            assert test_predictions.shape[2] == 1
        else:
            assert test_predictions.shape[2] == len(test_data)
        assert test_predictions.shape[-1] == test_targets.shape[-1]

        #
        # Repeat for BaggingEnsemble
        ens = BaggingEnsemble(keras_model, n_estimators=2, n_predictions=2)

        ens.compile(
            optimizer=Adam(), loss=categorical_crossentropy, weighted_metrics=["acc"]
        )

        test_gen = generator.flow(test_data)
        # Check that passing invalid parameters is handled correctly. We will not check error handling for those
        # parameters that Keras will be responsible for.
        with pytest.raises(ValueError):
            ens.predict_generator(generator=test_gen, predict_data=test_data)

        # We won't train the model instead use the initial random weights to test
        # the evaluate_generator method.
        test_predictions = ens.predict_generator(test_gen, summarise=True)

        print("test_predictions shape {}".format(test_predictions.shape))
        if i > 1:
            # GAT and GCN are full batch so the batch dimension is 1
            assert len(test_predictions) == 1
            assert test_predictions.shape[1] == test_targets.shape[0]
        else:
            assert len(test_predictions) == len(test_data)
        assert test_predictions.shape[-1] == test_targets.shape[-1]

        test_predictions = ens.predict_generator(test_gen, summarise=False)

        assert test_predictions.shape[0] == ens.n_estimators
        assert test_predictions.shape[1] == ens.n_predictions
        if i > 1:
            assert test_predictions.shape[2] == 1
        else:
            assert test_predictions.shape[2] == len(test_data)
        assert test_predictions.shape[-1] == test_targets.shape[-1]

#
# Tests for link prediction that can't be combined easily with the node attribute inference workflow above.
#
def test_evaluate_generator_link_prediction():

    edge_ids_test = np.array([[1, 2], [2, 3], [1, 3]])
    edge_labels_test = np.array([1, 1, 0])

    graph = example_graph_1(feature_size=4)

    # base_model, keras_model, generator, train_gen
    gnn_models = [
        create_graphSAGE_model(graph, link_prediction=True),
        create_HinSAGE_model(graph, link_prediction=True),
    ]

    for gnn_model in gnn_models:
        keras_model = gnn_model[1]
        generator = gnn_model[2]

        ens = Ensemble(keras_model, n_estimators=2, n_predictions=3)

        ens.compile(
            optimizer=Adam(), loss=binary_crossentropy, weighted_metrics=["acc"]
        )

        # Check that passing invalid parameters is handled correctly. We will not check error handling for those
        # parameters that Keras will be responsible for.
        with pytest.raises(ValueError):
            ens.evaluate_generator(
                generator=generator,
                test_data=edge_ids_test,
                test_targets=edge_labels_test,
            )

        with pytest.raises(ValueError):
            ens.evaluate_generator(
                generator=generator,
                test_data=edge_labels_test,
                test_targets=None,  # must give test_targets
            )

        with pytest.raises(ValueError):
            ens.evaluate_generator(
                generator=generator.flow(edge_ids_test, edge_labels_test),
                test_data=edge_ids_test,
                test_targets=edge_labels_test,
            )

        # We won't train the model instead use the initial random weights to test
        # the evaluate_generator method.
        test_metrics_mean, test_metrics_std = ens.evaluate_generator(
            generator.flow(edge_ids_test, edge_labels_test)
        )

        assert len(test_metrics_mean) == len(test_metrics_std)
        assert len(test_metrics_mean.shape) == 1
        assert len(test_metrics_std.shape) == 1

        #
        # Repeat for BaggingEnsemble

        ens = BaggingEnsemble(keras_model, n_estimators=2, n_predictions=3)

        ens.compile(
            optimizer=Adam(), loss=binary_crossentropy, weighted_metrics=["acc"]
        )

        # Check that passing invalid parameters is handled correctly. We will not check error handling for those
        # parameters that Keras will be responsible for.
        with pytest.raises(ValueError):
            ens.evaluate_generator(
                generator=generator,
                test_data=edge_ids_test,
                test_targets=edge_labels_test,
            )

        with pytest.raises(ValueError):
            ens.evaluate_generator(
                generator=generator,
                test_data=edge_labels_test,
                test_targets=None,  # must give test_targets
            )

        with pytest.raises(ValueError):
            ens.evaluate_generator(
                generator=generator.flow(edge_ids_test, edge_labels_test),
                test_data=edge_ids_test,
                test_targets=edge_labels_test,
            )

        # We won't train the model instead use the initial random weights to test
        # the evaluate_generator method.
        test_metrics_mean, test_metrics_std = ens.evaluate_generator(
            generator.flow(edge_ids_test, edge_labels_test)
        )

        assert len(test_metrics_mean) == len(test_metrics_std)
        assert len(test_metrics_mean.shape) == 1
        assert len(test_metrics_std.shape) == 1

def test_predict_generator_link_prediction():

    edge_ids_test = np.array([[1, 2], [2, 3], [1, 3]])

    graph = example_graph_1(feature_size=2)

    # base_model, keras_model, generator, train_gen
    gnn_models = [
        create_graphSAGE_model(graph, link_prediction=True),
        create_HinSAGE_model(graph, link_prediction=True),
    ]

    for gnn_model in gnn_models:
        keras_model = gnn_model[1]
        generator = gnn_model[2]

        ens = Ensemble(keras_model, n_estimators=2, n_predictions=3)

        ens.compile(
            optimizer=Adam(), loss=binary_crossentropy, weighted_metrics=["acc"]
        )

        test_gen = generator.flow(edge_ids_test)
        # Check that passing invalid parameters is handled correctly. We will not check error handling for those
        # parameters that Keras will be responsible for.
        with pytest.raises(ValueError):
            ens.predict_generator(generator=test_gen, predict_data=edge_ids_test)

        # We won't train the model instead use the initial random weights to test
        # the evaluate_generator method.
        test_predictions = ens.predict_generator(test_gen, summarise=True)

        print("test_predictions shape {}".format(test_predictions.shape))
        assert len(test_predictions) == len(edge_ids_test)

        assert test_predictions.shape[1] == 1

        test_predictions = ens.predict_generator(test_gen, summarise=False)

        assert test_predictions.shape[0] == ens.n_estimators
        assert test_predictions.shape[1] == ens.n_predictions
        assert test_predictions.shape[2] == len(edge_ids_test)
        assert test_predictions.shape[3] == 1

        #
        # Repeat for BaggingEnsemble
        ens = BaggingEnsemble(keras_model, n_estimators=2, n_predictions=3)

        ens.compile(
            optimizer=Adam(), loss=binary_crossentropy, weighted_metrics=["acc"]
        )

        test_gen = generator.flow(edge_ids_test)
        # Check that passing invalid parameters is handled correctly. We will not check error handling for those
        # parameters that Keras will be responsible for.
        with pytest.raises(ValueError):
            ens.predict_generator(generator=test_gen, predict_data=edge_ids_test)

        # We won't train the model instead use the initial random weights to test
        # the evaluate_generator method.
        test_predictions = ens.predict_generator(test_gen, summarise=True)

        print("test_predictions shape {}".format(test_predictions.shape))
        assert len(test_predictions) == len(edge_ids_test)

        assert test_predictions.shape[1] == 1

        test_predictions = ens.predict_generator(test_gen, summarise=False)

        assert test_predictions.shape[0] == ens.n_estimators
        assert test_predictions.shape[1] == ens.n_predictions
        assert test_predictions.shape[2] == len(edge_ids_test)
        assert test_predictions.shape[3] == 1

def test_unsupervised_embeddings_prediction():
    #
    # Tests for embedding generation using inductive GCNs
    #
    edge_ids_test = np.array([[1, 2], [2, 3], [1, 3]])

    graph = example_graph_1(feature_size=2)

    # base_model, keras_model, generator, train_gen
    gnn_models = [create_Unsupervised_graphSAGE_model(graph)]

    for gnn_model in gnn_models:
        keras_model = gnn_model[1]
        generator = gnn_model[2]
        train_gen = gnn_model[3]

        ens = Ensemble(keras_model, n_estimators=2, n_predictions=1)

        ens.compile(
            optimizer=Adam(), loss=binary_crossentropy, weighted_metrics=["acc"]
        )

        # Check that passing invalid parameters is handled correctly. We will not check error handling for those
        # parameters that Keras will be responsible for.
        with pytest.raises(ValueError):
            ens.predict_generator(generator=generator, predict_data=edge_ids_test)

        # We won't train the model instead use the initial random weights to test
        # the evaluate_generator method.
        ens.models = [
            keras.Model(inputs=model.input, outputs=model.output[-1])
            for model in ens.models
        ]

        test_predictions = ens.predict_generator(train_gen, summarise=False)

        print("test_predictions embeddings shape {}".format(test_predictions.shape))
        assert test_predictions.shape[0] == ens.n_estimators
        assert test_predictions.shape[1] == ens.n_predictions
        assert (
            test_predictions.shape[2] > 1
        )  # Embeddings dim is > than binary prediction

        #
        # Repeat for BaggingEnsemble
        ens = BaggingEnsemble(keras_model, n_estimators=2, n_predictions=1)

        ens.compile(
            optimizer=Adam(), loss=binary_crossentropy, weighted_metrics=["acc"]
        )

        # Check that passing invalid parameters is handled correctly. We will not check error handling for those
        # parameters that Keras will be responsible for.
        with pytest.raises(ValueError):
            ens.predict_generator(generator=train_gen, predict_data=edge_ids_test)

        # We won't train the model instead use the initial random weights to test
        # the evaluate_generator method.
        test_predictions = ens.predict_generator(train_gen, summarise=False)

        print("test_predictions shape {}".format(test_predictions.shape))

        assert test_predictions.shape[1] == 1

        test_predictions = ens.predict_generator(train_gen, summarise=False)

        assert test_predictions.shape[0] == ens.n_estimators
        assert test_predictions.shape[1] == ens.n_predictions
        assert test_predictions.shape[2] > 1
