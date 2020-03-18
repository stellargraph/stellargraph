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

import itertools

import pytest

import pandas as pd
import numpy as np

from tensorflow.keras import Model, initializers, losses

from stellargraph import StellarGraph
from stellargraph.mapper.knowledge_graph import KGTripleGenerator
from stellargraph.layer.knowledge_graph import ComplEx, DistMult

from .. import test_utils
from ..test_utils.graphs import knowledge_graph


pytestmark = [
    test_utils.ignore_stellargraph_experimental_mark,
    pytest.mark.filterwarnings(
        r"ignore:ComplEx:stellargraph.core.experimental.ExperimentalWarning"
    ),
]


def triple_df(*values):
    return pd.DataFrame(values, columns=["source", "label", "target"])


def test_complex(knowledge_graph):
    # this test creates a random untrained model and predicts every possible edge in the graph, and
    # compares that to a direct implementation of the scoring method in the paper
    gen = KGTripleGenerator(knowledge_graph, 3)

    # use a random initializer with a large positive range, so that any differences are obvious
    init = initializers.RandomUniform(-1, 1)
    complex_model = ComplEx(gen, 5, embeddings_initializer=init)
    x_inp, x_out = complex_model.build()

    model = Model(x_inp, x_out)
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True))

    every_edge = itertools.product(
        knowledge_graph.nodes(),
        knowledge_graph._edges.types.pandas_index,
        knowledge_graph.nodes(),
    )
    df = triple_df(*every_edge)

    # check the model can be trained on a few (uneven) batches
    model.fit(
        gen.flow(df.iloc[:7], negative_samples=2),
        validation_data=gen.flow(df.iloc[7:14], negative_samples=3),
    )

    # compute the exact values based on the model by extracting the embeddings for each element and
    # doing the Re(<e_s, w_r, conj(e_o)>) inner product
    s_idx = knowledge_graph._get_index_for_nodes(df.source)
    r_idx = knowledge_graph._edges.types.to_iloc(df.label)
    o_idx = knowledge_graph._get_index_for_nodes(df.target)

    nodes, edge_types = complex_model.embeddings()
    # the rows correspond to the embeddings for the given edge, so we can do bulk operations
    e_s = nodes[s_idx, :]
    w_r = edge_types[r_idx, :]
    e_o = nodes[o_idx, :]
    actual = (e_s * w_r * e_o.conj()).sum(axis=1).real

    # predict every edge using the model
    prediction = model.predict(gen.flow(df))

    # (use an absolute tolerance to allow for catastrophic cancellation around very small values)
    assert np.allclose(prediction[:, 0], actual, rtol=1e-3, atol=1e-14)

    # the model is stateful (i.e. it holds the weights permanently) so the predictions with a second
    # 'build' should be the same as the original one
    model2 = Model(*complex_model.build())
    prediction2 = model2.predict(gen.flow(df))
    assert np.array_equal(prediction, prediction2)


def test_complex_rankings():
    nodes = ["a", "b", "c", "d"]
    rels = ["W", "X", "Y", "Z"]
    empty = pd.DataFrame(columns=["source", "target"])

    every_edge = itertools.product(nodes, rels, nodes)
    df = triple_df(*every_edge)

    all_edges = StellarGraph(
        nodes=pd.DataFrame(index=nodes),
        edges={name: df.drop(columns="label") for name, df in df.groupby("label")},
    )

    gen = KGTripleGenerator(all_edges, 3)
    complex_model = ComplEx(gen, 5)
    x_inp, x_out = complex_model.build()
    model = Model(x_inp, x_out)

    raw = complex_model.rank_edges_against_all_nodes(gen.flow(df), all_edges)
    # basic check that the ranks are formed correctly
    assert raw.dtype == int
    assert np.all(raw >= 1)

    # check the ranks against computing them from the model predictions directly. That is, for each
    # edge, compare the rank against one computed by counting the predictions.
    predictions = model.predict(gen.flow(df))

    for (source, rel, target), score, (mod_o_rank, mod_s_rank) in zip(
        df.itertuples(index=False), predictions, raw
    ):
        mod_o_scores = predictions[(df.source == source) & (df.label == rel)]
        expected_mod_o_rank = 1 + (mod_o_scores > score).sum()
        assert mod_o_rank == expected_mod_o_rank

        mod_s_scores = predictions[(df.label == rel) & (df.target == target)]
        expected_mod_s_rank = 1 + (mod_s_scores > score).sum()
        assert mod_s_rank == expected_mod_s_rank


def test_dismult(knowledge_graph):
    # this test creates a random untrained model and predicts every possible edge in the graph, and
    # compares that to a direct implementation of the scoring method in the paper
    gen = KGTripleGenerator(knowledge_graph, 3)

    # use a random initializer with a large range, so that any differences are obvious
    init = initializers.RandomUniform(-1, 1)
    distmult_model = DistMult(gen, 5, embeddings_initializer=init)
    x_inp, x_out = distmult_model.build()

    model = Model(x_inp, x_out)

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True))

    every_edge = itertools.product(
        knowledge_graph.nodes(),
        knowledge_graph._edges.types.pandas_index,
        knowledge_graph.nodes(),
    )
    df = triple_df(*every_edge)

    # check the model can be trained on a few (uneven) batches
    model.fit(
        gen.flow(df.iloc[:7], negative_samples=2),
        validation_data=gen.flow(df.iloc[7:14], negative_samples=3),
    )

    # compute the exact values based on the model by extracting the embeddings for each element and
    # doing the y_(e_1)^T M_r y_(e_2) = <e_1, w_r, e_2> inner product
    s_idx = knowledge_graph._get_index_for_nodes(df.source)
    r_idx = knowledge_graph._edges.types.to_iloc(df.label)
    o_idx = knowledge_graph._get_index_for_nodes(df.target)

    nodes, edge_types = distmult_model.embeddings()
    # the rows correspond to the embeddings for the given edge, so we can do bulk operations
    e_s = nodes[s_idx, :]
    w_r = edge_types[r_idx, :]
    e_o = nodes[o_idx, :]
    actual = (e_s * w_r * e_o).sum(axis=1)

    # predict every edge using the model
    prediction = model.predict(gen.flow(df))

    # (use an absolute tolerance to allow for catastrophic cancellation around very small values)
    assert np.allclose(prediction[:, 0], actual, rtol=1e-3, atol=1e-14)

    # the model is stateful (i.e. it holds the weights permanently) so the predictions with a second
    # 'build' should be the same as the original one
    model2 = Model(*distmult_model.build())
    prediction2 = model2.predict(gen.flow(df))
    assert np.array_equal(prediction, prediction2)
