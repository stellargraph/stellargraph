# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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


"""
GCN tests

"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.sparse as sps
import pytest
from stellargraph.layer.misc import *
from ..test_utils.graphs import create_graph_features, example_graph
from stellargraph.mapper import *
from stellargraph.layer import *


def sparse_matrix_example(N=10, density=0.1):
    A = sps.rand(N, N, density=density, format="coo")
    A_indices = np.hstack((A.row[:, None], A.col[:, None]))
    A_values = A.data
    return A_indices, A_values, A


def test_squeezedsparseconversion():
    N = 10
    x_t = keras.Input(batch_shape=(1, N, 1), dtype="float32")
    A_ind = keras.Input(batch_shape=(1, None, 2), dtype="int64")
    A_val = keras.Input(batch_shape=(1, None), dtype="float32")

    A_mat = SqueezedSparseConversion(shape=(N, N), dtype=A_val.dtype)([A_ind, A_val])

    x_out = keras.layers.Lambda(
        lambda xin: K.expand_dims(K.dot(xin[0], K.squeeze(xin[1], 0)), 0)
    )([A_mat, x_t])

    model = keras.Model(inputs=[x_t, A_ind, A_val], outputs=x_out)

    x = np.random.randn(1, N, 1)
    A_indices, A_values, A = sparse_matrix_example(N)

    z = model.predict([x, np.expand_dims(A_indices, 0), np.expand_dims(A_values, 0)])

    assert np.allclose(z.squeeze(), A.dot(x.squeeze()), atol=1e-7)


def test_squeezedsparseconversion_dtype():
    N = 10
    x_t = keras.Input(batch_shape=(1, N, 1), dtype="float64")
    A_ind = keras.Input(batch_shape=(1, None, 2), dtype="int64")
    A_val = keras.Input(batch_shape=(1, None), dtype="float32")

    A_mat = SqueezedSparseConversion(shape=(N, N), dtype="float64")([A_ind, A_val])

    x_out = keras.layers.Lambda(
        lambda xin: K.expand_dims(K.dot(xin[0], K.squeeze(xin[1], 0)), 0)
    )([A_mat, x_t])

    model = keras.Model(inputs=[x_t, A_ind, A_val], outputs=x_out)

    x = np.random.randn(1, N, 1)
    A_indices, A_values, A = sparse_matrix_example(N)

    z = model.predict([x, np.expand_dims(A_indices, 0), np.expand_dims(A_values, 0)])

    assert A_mat.dtype == tf.dtypes.float64
    assert np.allclose(z.squeeze(), A.dot(x.squeeze()), atol=1e-7)


def test_squeezedsparseconversion_axis():
    N = 10
    A_indices, A_values, A = sparse_matrix_example(N)
    nnz = len(A_indices)

    A_ind = keras.Input(batch_shape=(nnz, 2), dtype="int64")
    A_val = keras.Input(batch_shape=(nnz, 1), dtype="float32")

    # Keras reshapes everything to have ndim at least 2, we need to flatten values
    A_val_1 = keras.layers.Lambda(lambda A: K.reshape(A, (-1,)))(A_val)

    A_mat = SqueezedSparseConversion(shape=(N, N), axis=None, dtype=A_val_1.dtype)(
        [A_ind, A_val_1]
    )

    ones = tf.ones((N, 1))

    x_out = keras.layers.Lambda(lambda xin: K.dot(xin, ones))(A_mat)

    model = keras.Model(inputs=[A_ind, A_val], outputs=x_out)
    z = model.predict([A_indices, A_values])

    assert np.allclose(z, A.sum(axis=1), atol=1e-7)


def test_gather_indices():
    batch_dim = 3
    data_in = keras.Input(batch_shape=(batch_dim, 5, 7))
    indices_in = keras.Input(batch_shape=(batch_dim, 11), dtype="int32")

    data = np.arange(np.product(data_in.shape)).reshape(data_in.shape)
    indices = np.random.choice(range(min(data_in.shape)), indices_in.shape)

    # check that the layer acts the same as tf.gather
    def run(**kwargs):
        layer = GatherIndices(**kwargs)
        expected = tf.gather(data, indices, **kwargs)

        out = GatherIndices(**kwargs)([data_in, indices_in])
        model = keras.Model(inputs=[data_in, indices_in], outputs=out)
        pred = model.predict([data, indices])
        np.testing.assert_array_equal(pred, expected)

    # default settings
    run()
    # with a batch dimension
    run(batch_dims=1)
    # various other forms...
    run(axis=1)
    run(batch_dims=1, axis=2)


def _deprecated_test(sg_model):

    with pytest.warns(DeprecationWarning):
        x_in, x_out = sg_model.build()

    try:
        if type(sg_model) is not RGCN:
            x_in, x_out = sg_model._node_model()
            with pytest.warns(DeprecationWarning):
                x_in, x_out = sg_model.node_model()
    except AttributeError:
        pass

    try:
        x_in, x_out = sg_model._link_model()
        with pytest.warns(DeprecationWarning):
            x_in, x_out = sg_model.link_model()
    except AttributeError:
        pass


def test_deprecated_model_functions():
    G, _ = create_graph_features()

    # full batch models
    generator = FullBatchNodeGenerator(G)
    for model_type in [GCN, GAT, PPNP, APPNP]:
        sg_model = model_type(
            generator=generator, layer_sizes=[4], activations=["relu"]
        )
        _deprecated_test(sg_model)

    # test DeepGraphInfomax here because it needs a fullbatch model
    sg_model = DeepGraphInfomax(sg_model)
    _deprecated_test(sg_model)

    # models with layer_sizes and activations args
    generators = [
        ClusterNodeGenerator(G),
        HinSAGENodeGenerator(
            G, batch_size=1, num_samples=[2], head_node_type="default"
        ),
        GraphSAGENodeGenerator(G, batch_size=1, num_samples=[2]),
        RelationalFullBatchNodeGenerator(G),
        PaddedGraphGenerator([G]),
    ]

    model_types = [
        ClusterGCN,
        HinSAGE,
        GraphSAGE,
        RGCN,
        GCNSupervisedGraphClassification,
    ]

    for generator, model_type in zip(generators, model_types):
        sg_model = model_type(
            layer_sizes=[2], activations=["relu"], generator=generator
        )
        _deprecated_test(sg_model)

    # models with embedding_dimension arg
    model_types = [WatchYourStep, DistMult, ComplEx, Attri2Vec]
    generators = [
        AdjacencyPowerGenerator(G),
        KGTripleGenerator(G, batch_size=1),
        KGTripleGenerator(G, batch_size=1),
    ]

    for generator, model_type in zip(generators, model_types):
        sg_model = model_type(generator=generator, embedding_dimension=2)
        _deprecated_test(sg_model)

    # outlier models that need to be treated separately

    generator = Attri2VecLinkGenerator(G, batch_size=1)
    sg_model = Attri2Vec(generator=generator, layer_sizes=[4], activation="sigmoid")
    _deprecated_test(sg_model)

    G = example_graph(feature_size=1, is_directed=True)
    generator = DirectedGraphSAGENodeGenerator(
        G, batch_size=1, in_samples=[2], out_samples=[2]
    )
    sg_model = DirectedGraphSAGE(
        generator=generator, layer_sizes=[4], activations=["relu"]
    )
    _deprecated_test(sg_model)
