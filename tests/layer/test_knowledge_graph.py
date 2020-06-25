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

import tensorflow as tf
from tensorflow.keras import Model, initializers, losses as tf_losses, layers

from stellargraph import StellarGraph, StellarDiGraph, losses as sg_losses
from stellargraph.mapper.knowledge_graph import KGTripleGenerator
from stellargraph.layer.knowledge_graph import (
    KGModel,
    KGScore,
    ComplEx,
    DistMult,
    RotatE,
    RotE,
    RotH,
    _ranks_from_score_columns,
)

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


@pytest.mark.parametrize("sample_strategy", ["uniform", "self-adversarial"])
def test_complex(knowledge_graph, sample_strategy):
    # this test creates a random untrained model and predicts every possible edge in the graph, and
    # compares that to a direct implementation of the scoring method in the paper
    gen = KGTripleGenerator(knowledge_graph, 3)

    # use a random initializer with a large positive range, so that any differences are obvious
    init = initializers.RandomUniform(-1, 1)
    complex_model = ComplEx(gen, 5, embeddings_initializer=init)
    x_inp, x_out = complex_model.in_out_tensors()

    model = Model(x_inp, x_out)
    if sample_strategy == "uniform":
        loss = tf_losses.BinaryCrossentropy(from_logits=True)
    else:
        loss = sg_losses.SelfAdversarialNegativeSampling()

    model.compile(loss=loss)

    every_edge = itertools.product(
        knowledge_graph.nodes(),
        knowledge_graph._edges.types.pandas_index,
        knowledge_graph.nodes(),
    )
    df = triple_df(*every_edge)

    # check the model can be trained on a few (uneven) batches
    model.fit(
        gen.flow(df.iloc[:7], negative_samples=2, sample_strategy=sample_strategy),
        validation_data=gen.flow(
            df.iloc[7:14], negative_samples=3, sample_strategy=sample_strategy
        ),
    )

    # compute the exact values based on the model by extracting the embeddings for each element and
    # doing the Re(<e_s, w_r, conj(e_o)>) inner product
    s_idx = knowledge_graph.node_ids_to_ilocs(df.source)
    r_idx = knowledge_graph._edges.types.to_iloc(df.label)
    o_idx = knowledge_graph.node_ids_to_ilocs(df.target)

    nodes, edge_types = complex_model.embeddings()
    # the rows correspond to the embeddings for the given edge, so we can do bulk operations
    e_s = nodes[s_idx, :]
    w_r = edge_types[r_idx, :]
    e_o = nodes[o_idx, :]
    actual = (e_s * w_r * e_o.conj()).sum(axis=1).real

    # predict every edge using the model
    prediction = model.predict(gen.flow(df))

    # (use an absolute tolerance to allow for catastrophic cancellation around very small values)
    np.testing.assert_allclose(prediction[:, 0], actual, rtol=1e-3, atol=1e-6)

    # the model is stateful (i.e. it holds the weights permanently) so the predictions with a second
    # 'build' should be the same as the original one
    model2 = Model(*complex_model.in_out_tensors())
    prediction2 = model2.predict(gen.flow(df))
    np.testing.assert_array_equal(prediction, prediction2)


def test_distmult(knowledge_graph):
    # this test creates a random untrained model and predicts every possible edge in the graph, and
    # compares that to a direct implementation of the scoring method in the paper
    gen = KGTripleGenerator(knowledge_graph, 3)

    # use a random initializer with a large range, so that any differences are obvious
    init = initializers.RandomUniform(-1, 1)
    distmult_model = DistMult(gen, 5, embeddings_initializer=init)
    x_inp, x_out = distmult_model.in_out_tensors()

    model = Model(x_inp, x_out)

    model.compile(loss=tf_losses.BinaryCrossentropy(from_logits=True))

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
    s_idx = knowledge_graph.node_ids_to_ilocs(df.source)
    r_idx = knowledge_graph._edges.types.to_iloc(df.label)
    o_idx = knowledge_graph.node_ids_to_ilocs(df.target)

    nodes, edge_types = distmult_model.embeddings()
    # the rows correspond to the embeddings for the given edge, so we can do bulk operations
    e_s = nodes[s_idx, :]
    w_r = edge_types[r_idx, :]
    e_o = nodes[o_idx, :]
    actual = (e_s * w_r * e_o).sum(axis=1)

    # predict every edge using the model
    prediction = model.predict(gen.flow(df))

    # (use an absolute tolerance to allow for catastrophic cancellation around very small values)
    np.testing.assert_allclose(prediction[:, 0], actual, rtol=1e-3, atol=1e-14)

    # the model is stateful (i.e. it holds the weights permanently) so the predictions with a second
    # 'build' should be the same as the original one
    model2 = Model(*distmult_model.in_out_tensors())
    prediction2 = model2.predict(gen.flow(df))
    np.testing.assert_array_equal(prediction, prediction2)


@test_utils.flaky_xfail_mark(AssertionError, 1623)
def test_rotate(knowledge_graph):
    margin = 2.34
    norm_order = 1.234

    # this test creates a random untrained model and predicts every possible edge in the graph, and
    # compares that to a direct implementation of the scoring method in the paper
    gen = KGTripleGenerator(knowledge_graph, 3)

    # use a random initializer with a large range, so that any differences are obvious
    init = initializers.RandomUniform(-1, 1)
    rotate_model = RotatE(
        gen, 5, margin=margin, norm_order=norm_order, embeddings_initializer=init
    )
    x_inp, x_out = rotate_model.in_out_tensors()

    model = Model(x_inp, x_out)

    model.compile(loss=tf_losses.BinaryCrossentropy(from_logits=True))

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
    s_idx = knowledge_graph.node_ids_to_ilocs(df.source)
    r_idx = knowledge_graph._edges.types.to_iloc(df.label)
    o_idx = knowledge_graph.node_ids_to_ilocs(df.target)

    nodes, edge_types = rotate_model.embeddings()
    # the rows correspond to the embeddings for the given edge, so we can do bulk operations
    e_s = nodes[s_idx, :]
    w_r = edge_types[r_idx, :]
    e_o = nodes[o_idx, :]

    # every edge-type embedding should be a unit rotation
    np.testing.assert_allclose(np.abs(w_r), 1)

    actual = margin - np.linalg.norm(e_s * w_r - e_o, ord=norm_order, axis=1)

    # predict every edge using the model
    prediction = model.predict(gen.flow(df))

    # (use an absolute tolerance to allow for catastrophic cancellation around very small values)
    np.testing.assert_allclose(prediction[:, 0], actual, rtol=1e-3, atol=1e-14)

    # the model is stateful (i.e. it holds the weights permanently) so the predictions with a second
    # 'build' should be the same as the original one
    model2 = Model(*rotate_model.in_out_tensors())
    prediction2 = model2.predict(gen.flow(df))
    np.testing.assert_array_equal(prediction, prediction2)


@pytest.mark.parametrize("model_class", [RotE, RotH])
def test_rote_roth(knowledge_graph, model_class):
    # this test creates a random untrained model and predicts every possible edge in the graph, and
    # compares that to a direct implementation of the scoring method in the paper
    gen = KGTripleGenerator(knowledge_graph, 3)

    # use a random initializer with a large range, so that any differences are obvious
    init = initializers.RandomUniform(-1, 1)
    rot_model = model_class(gen, 6, embeddings_initializer=init)
    x_inp, x_out = rot_model.in_out_tensors()

    model = Model(x_inp, x_out)
    model.summary()
    model.compile(loss=tf_losses.BinaryCrossentropy(from_logits=True))

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

    # predict every edge using the model
    prediction = model.predict(gen.flow(df))

    (node_emb, node_bias), (et_emb, et_theta) = rot_model.embedding_arrays()

    if model_class is RotE:
        # compute the exact values based on the model, for RotationE (the RotationH model is too
        # hard to test directly)
        s_idx = knowledge_graph.node_ids_to_ilocs(df.source)
        r_idx = knowledge_graph.edge_type_names_to_ilocs(df.label)
        o_idx = knowledge_graph.node_ids_to_ilocs(df.target)

        # the rows correspond to the embeddings for the given edge, so we can do bulk operations
        e_s = node_emb[s_idx, :]
        b_s = node_bias[s_idx, 0]

        r_r = et_emb[r_idx, :]
        theta_r = et_theta[r_idx, :]

        e_o = node_emb[o_idx, :]
        b_o = node_bias[o_idx, 0]

        rot_r = np.cos(theta_r) + 1j * np.sin(theta_r)

        assert e_s.dtype == np.float32
        rotated = (e_s.view(np.complex64) * rot_r).view(np.float32)
        actual = -np.linalg.norm(rotated + r_r - e_o, axis=-1) ** 2 + b_s + b_o

        np.testing.assert_allclose(prediction[:, 0], actual, rtol=1e-3, atol=1e-14)

    # the model is stateful (i.e. it holds the weights permanently) so the predictions with a second
    # 'build' should be the same as the original one
    model2 = Model(*rot_model.in_out_tensors())
    prediction2 = model2.predict(gen.flow(df))
    np.testing.assert_array_equal(prediction, prediction2)


@pytest.mark.parametrize(
    "model_maker",
    [
        ComplEx,
        DistMult,
        pytest.param(RotatE, marks=test_utils.flaky_xfail_mark(AssertionError, 1623)),
        pytest.param(RotH, marks=test_utils.flaky_xfail_mark(AssertionError, 1675)),
        RotE,
    ],
)
def test_model_rankings(model_maker):
    nodes = pd.DataFrame(index=["a", "b", "c", "d"])
    rels = ["W", "X", "Y", "Z"]
    empty = pd.DataFrame(columns=["source", "target"])

    every_edge = itertools.product(nodes.index, rels, nodes.index)
    every_edge_df = triple_df(*every_edge)

    no_edges = StellarDiGraph(nodes, {name: empty for name in rels})

    # the filtering is most interesting when there's a smattering of edges, somewhere between none
    # and all; this does a stratified sample by label, to make sure there's at least one edge from
    # each label.
    one_per_label_df = (
        every_edge_df.groupby("label").apply(lambda df: df.sample(n=1)).droplevel(0)
    )
    others_df = every_edge_df.sample(frac=0.25)
    some_edges_df = pd.concat([one_per_label_df, others_df], ignore_index=True)

    some_edges = StellarDiGraph(
        nodes,
        {name: df.drop(columns="label") for name, df in some_edges_df.groupby("label")},
    )

    all_edges = StellarDiGraph(
        nodes=nodes,
        edges={
            name: df.drop(columns="label")
            for name, df in every_edge_df.groupby("label")
        },
    )

    gen = KGTripleGenerator(all_edges, 3)
    sg_model = model_maker(gen, embedding_dimension=6)
    x_inp, x_out = sg_model.in_out_tensors()
    model = Model(x_inp, x_out)

    raw_some, filtered_some = sg_model.rank_edges_against_all_nodes(
        gen.flow(every_edge_df), some_edges
    )
    # basic check that the ranks are formed correctly
    assert raw_some.dtype == int
    assert np.all(raw_some >= 1)
    # filtered ranks are never greater, and sometimes less
    assert np.all(filtered_some <= raw_some)
    assert np.any(filtered_some < raw_some)

    raw_no, filtered_no = sg_model.rank_edges_against_all_nodes(
        gen.flow(every_edge_df), no_edges
    )
    np.testing.assert_array_equal(raw_no, raw_some)
    # with no edges, filtering does nothing
    np.testing.assert_array_equal(raw_no, filtered_no)

    raw_all, filtered_all = sg_model.rank_edges_against_all_nodes(
        gen.flow(every_edge_df), all_edges
    )
    np.testing.assert_array_equal(raw_all, raw_some)
    # when every edge is known, the filtering should eliminate every possibility
    assert np.all(filtered_all == 1)

    # check the ranks against computing them from the model predictions directly. That is, for each
    # edge, compare the rank against one computed by counting the predictions. This computes the
    # filtered ranks naively too.
    predictions = model.predict(gen.flow(every_edge_df))

    for (source, rel, target), score, raw, filtered in zip(
        every_edge_df.itertuples(index=False), predictions, raw_some, filtered_some
    ):
        # rank for the subset specified by the given selector
        def rank(compare_selector):
            return 1 + (predictions[compare_selector] > score).sum()

        same_r = every_edge_df.label == rel

        same_s_r = (every_edge_df.source == source) & same_r

        expected_raw_mod_o_rank = rank(same_s_r)
        assert raw[0] == expected_raw_mod_o_rank

        known_objects = some_edges_df[
            (some_edges_df.source == source) & (some_edges_df.label == rel)
        ]
        object_is_unknown = ~every_edge_df.target.isin(known_objects.target)
        expected_filt_mod_o_rank = rank(same_s_r & object_is_unknown)
        assert filtered[0] == expected_filt_mod_o_rank

        same_r_o = same_r & (every_edge_df.target == target)

        expected_raw_mod_s_rank = rank(same_r_o)
        assert raw[1] == expected_raw_mod_s_rank

        known_subjects = some_edges_df[
            (some_edges_df.label == rel) & (some_edges_df.target == target)
        ]
        subject_is_unknown = ~every_edge_df.source.isin(known_subjects.source)
        expected_filt_mod_s_rank = rank(subject_is_unknown & same_r_o)
        assert filtered[1] == expected_filt_mod_s_rank


@pytest.mark.parametrize("tie_breaking", ["top", "bottom", "random"])
def test_tie_breaking(tie_breaking):
    pred_scores = np.array(
        [
            [1, 5, 8],  # true_modified_node_ilocs:
            [1, 3, 8],  # 1
            [1, 2, 7],  # 2
            [1, 2, 6],  # 3
        ]
    )
    known_edges_graph = StellarDiGraph(
        nodes=pd.DataFrame(index=["a", "b", "c", "d"]),
        edges=pd.DataFrame(
            [
                # preds[0, :]: edge being predicted, checking it's counted properly for 'filtered'
                ("a", "b"),
                # preds[1, :]: the other tied edge, to see the 'bottom' score move up
                ("b", "d"),
            ],
            columns=["source", "target"],
        ),
    )

    copies = 100

    rankings = [
        _ranks_from_score_columns(
            pred_scores,
            true_modified_node_ilocs=np.array([1, 2, 3]),
            unmodified_node_ilocs=np.array([0, 1, 2]),
            true_rel_ilocs=np.array([0, 0, 0]),
            modified_object=True,
            known_edges_graph=known_edges_graph,
            tie_breaking=tie_breaking,
        )
        for _ in range(copies)
    ]

    all_rankings = np.array(rankings)
    assert all_rankings.shape == (copies, 2, 3)

    top_expected = np.repeat([[[1, 3, 4], [1, 3, 4]]], copies, axis=0)
    bottom_expected = np.repeat([[[4, 4, 4], [4, 3, 4]]], copies, axis=0)

    if tie_breaking == "top":
        np.testing.assert_array_equal(all_rankings, top_expected)

    elif tie_breaking == "bottom":
        np.testing.assert_array_equal(all_rankings, bottom_expected)

    elif tie_breaking == "random":
        assert (all_rankings >= top_expected).all()
        assert (all_rankings <= bottom_expected).all()

        # check both raw and filtered results (independently) have some variation in them
        for i in range(all_rankings.shape[1]):
            raw_or_filtered = all_rankings[:, i, :]
            assert (raw_or_filtered != top_expected[:, i, :]).any()
            assert (raw_or_filtered != bottom_expected[:, i, :]).any()


def test_embedding_validation(knowledge_graph):
    class X(layers.Layer, KGScore):
        def __init__(self, emb):
            self.emb = emb

        def embeddings(self, *args, **kwargs):
            return self.emb

        def bulk_scoring(self, *args, **kwargs):
            raise NotImplementedError()

    gen = KGTripleGenerator(knowledge_graph, 3)

    e = layers.Embedding(5, 4)

    kwargs = {
        "embeddings_initializer": None,
        "embeddings_regularizer": None,
    }
    with pytest.raises(ValueError, match="scoring: .* found a sequence of length 3"):
        KGModel(gen, X((1, 2, 3)), 2, **kwargs)

    with pytest.raises(
        ValueError, match=r"scoring: .* found a pair with types \(Embedding, int\)"
    ):
        KGModel(gen, X((e, 2)), 2, **kwargs)

    with pytest.raises(
        ValueError,
        match=r"scoring: .* found a pair of lists containing types \(\[Embedding, Embedding\], \[int\]\)",
    ):
        KGModel(gen, X(([e, e], [2])), 2, **kwargs)

    # all good:
    KGModel(gen, X(([e, e], [e, e, e])), 2, **kwargs)


@pytest.mark.parametrize("model_maker", [ComplEx, DistMult, RotatE, RotH, RotE])
def test_save_load(tmpdir, knowledge_graph, model_maker):
    gen = KGTripleGenerator(knowledge_graph, 3)
    sg_model = model_maker(gen, embedding_dimension=6)
    test_utils.model_save_load(tmpdir, sg_model)
