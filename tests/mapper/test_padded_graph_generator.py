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
from stellargraph.core.graph import *
from stellargraph.mapper.padded_graph_generator import (
    PaddedGraphGenerator,
    PaddedGraphSequence,
)

import numpy as np
import pytest
from ..test_utils.graphs import example_graph_random, example_graph, example_hin_1

graphs = [
    example_graph_random(feature_size=4, n_nodes=6),
    example_graph_random(feature_size=4, n_nodes=5),
    example_graph_random(feature_size=4, n_nodes=3),
]


def _mask(valid, total):
    return np.repeat([True, False], (valid, total - valid))


def test_generator_init():
    generator = PaddedGraphGenerator(graphs=graphs)
    assert len(generator.graphs) == len(graphs)


def test_generator_different_feature_numbers():
    graphs_diff_num_features = [
        example_graph_random(feature_size=2, n_nodes=6),
        example_graph_random(feature_size=4, n_nodes=5),
    ]

    with pytest.raises(
        ValueError,
        match="graphs: expected node features for all graph to have same dimensions,.*2.*4",
    ):
        generator = PaddedGraphGenerator(graphs=graphs_diff_num_features)

    generator = PaddedGraphGenerator(graphs=graphs_diff_num_features[:1])
    with pytest.raises(
        ValueError,
        match="graphs: expected node features for all graph to have same dimensions,.*2.*4",
    ):
        seq = generator.flow(graphs_diff_num_features)


def test_generator_init_nx_graph():
    graphs_nx = [
        example_graph_random(feature_size=4, n_nodes=3, is_directed=False),
        example_graph_random(
            feature_size=4, n_nodes=2, is_directed=False
        ).to_networkx(),
    ]

    with pytest.raises(
        TypeError, match="graphs: expected.*StellarGraph.*found MultiGraph."
    ):
        generator = PaddedGraphGenerator(graphs=graphs_nx)


def test_generator_hin():
    graphs_mixed = [
        example_graph_random(feature_size=2, n_nodes=6),
        example_hin_1(is_directed=False),
    ]

    with pytest.raises(
        ValueError,
        match="graphs: expected only graphs with a single node type.*found.*'A', 'B'",
    ):
        generator = PaddedGraphGenerator(graphs=graphs_mixed)

    generator = PaddedGraphGenerator(graphs=graphs_mixed[:1])
    with pytest.raises(
        ValueError,
        match="graphs: expected only graphs with a single node type.*found.*'A', 'B'",
    ):
        seq = generator.flow(graphs_mixed)


def test_generator_empty():
    graphs = [
        example_graph_random(feature_size=2, n_nodes=4),
        example_graph_random(feature_size=2, node_types=0, edge_types=0),
    ]

    with pytest.raises(
        ValueError,
        match="graphs: expected every graph to be non-empty, found graph with no nodes",
    ):
        generator = PaddedGraphGenerator(graphs=graphs)

    generator = PaddedGraphGenerator(graphs=graphs[:1])
    with pytest.raises(
        ValueError,
        match="graphs: expected every graph to be non-empty, found graph with no nodes",
    ):
        seq = generator.flow(graphs)


def test_generator_flow_invalid_batch_size():
    with pytest.raises(
        ValueError, match="expected batch_size.*strictly positive integer, found -1"
    ):
        PaddedGraphGenerator(graphs=graphs).flow(graphs=[0], batch_size=-1)

    with pytest.raises(
        TypeError, match="expected batch_size.*integer type, found float"
    ):
        PaddedGraphGenerator(graphs=graphs).flow(graphs=[0], batch_size=2.0)

    with pytest.raises(
        ValueError, match="expected batch_size.*strictly positive integer, found 0"
    ):
        PaddedGraphGenerator(graphs=graphs).flow(graphs=[0], batch_size=0)


def test_generator_flow_incorrect_targets():

    generator = PaddedGraphGenerator(graphs=graphs)

    with pytest.raises(
        ValueError, match="expected targets to be the same length as node_ids,.*1 vs 2"
    ):
        generator.flow(graphs=[0, 1], targets=np.array([0]))

    with pytest.raises(
        TypeError, match="targets: expected an iterable or None object, found int"
    ):
        generator.flow(graphs=[0, 1], targets=1)


def test_generator_flow_invalid_shape():
    generator = PaddedGraphGenerator(graphs=graphs)

    with pytest.raises(
        ValueError, match=r"graphs: expected a shape .* found shape \(\)"
    ):
        generator.flow(0)

    with pytest.raises(
        ValueError, match=r"graphs: expected a shape .* found shape \(2, 3, 4\)"
    ):
        generator.flow(np.ones((2, 3, 4)))


def test_generator_flow_no_targets():

    generator = PaddedGraphGenerator(graphs=graphs)

    seq = generator.flow(graphs=[0, 1, 2], batch_size=2)
    assert isinstance(seq, PaddedGraphSequence)

    assert len(seq) == 2  # two batches

    # The first batch should be size 2 and the second batch size 1
    values_0, targets_0 = seq[0]

    assert len(values_0) == 3
    assert values_0[0].shape[0] == 2
    assert values_0[1].shape[0] == 2
    assert values_0[2].shape[0] == 2
    assert targets_0 is None

    values_1, targets_1 = seq[1]

    assert len(values_1) == 3
    assert values_1[0].shape[0] == 1
    assert values_1[1].shape[0] == 1
    assert values_1[2].shape[0] == 1
    assert targets_1 is None


def test_generator_flow_check_padding():

    generator = PaddedGraphGenerator(graphs=graphs)

    seq = generator.flow(graphs=[0, 2], batch_size=2)
    assert isinstance(seq, PaddedGraphSequence)

    assert len(seq) == 1

    # The largest graph has 6 nodes vs 3 for the smallest one.
    # Check that the data matrices have the correct size 6
    batch = seq[0]

    assert batch[0][0].shape == (2, 6, 4)
    assert batch[0][1].shape == (2, 6)
    assert batch[0][2].shape == (2, 6, 6)

    for mask in batch[0][1]:
        assert np.sum(mask) == 6 or np.sum(mask) == 3


def test_generator_flow_with_targets():

    generator = PaddedGraphGenerator(graphs=graphs)

    seq = generator.flow(graphs=[1, 2], targets=np.array([0, 1]), batch_size=1)
    assert isinstance(seq, PaddedGraphSequence)

    for batch in seq:
        assert batch[0][0].shape[0] == 1
        assert batch[0][1].shape[0] == 1
        assert batch[0][2].shape[0] == 1
        assert batch[1].shape[0] == 1


@pytest.mark.parametrize("symmetric_normalization", [True, False])
@pytest.mark.parametrize("weighted", [True, False])
def test_generator_adj_normalisation(symmetric_normalization, weighted):

    graph = example_graph(feature_size=4, edge_weights=True)

    generator = PaddedGraphGenerator(graphs=[graph])
    seq = generator.flow(
        graphs=[0], symmetric_normalization=symmetric_normalization, weighted=weighted
    )

    adj_norm_seq = seq.normalized_adjs[0].todense()

    adj = np.array(graph.to_adjacency_matrix(weighted=weighted).todense())
    np.fill_diagonal(adj, 1)
    if symmetric_normalization:
        inv_deg = np.diag(np.sqrt(1.0 / adj.sum(axis=1)))
        adj_norm = inv_deg.dot(adj).dot(inv_deg)
    else:
        inv_deg = np.diag(1.0 / adj.sum(axis=1))
        adj_norm = inv_deg.dot(adj)

    np.testing.assert_allclose(np.asarray(adj_norm_seq), adj_norm)


def test_generator_flow_shuffle():

    generator = PaddedGraphGenerator(graphs=graphs)
    num_epochs_to_check = 5

    def get_batches(seq):
        return [seq[i][0] for i in range(len(seq))]

    def batches_all_equal(batches, other_batches):
        checks = [
            inp.shape == other_inp.shape and np.allclose(inp, other_inp)
            for batch, other_batch in zip(batches, other_batches)
            for inp, other_inp in zip(batch, other_batch)
        ]
        return all(checks)

    def get_next_epoch_batches(seq):
        seq.on_epoch_end()
        return get_batches(seq)

    # shuffle = False
    seq = generator.flow(graphs=[0, 1, 2], batch_size=2, shuffle=False)
    batches = get_batches(seq)
    for _ in range(num_epochs_to_check):
        assert batches_all_equal(batches, get_next_epoch_batches(seq))

    # shuffle = True, fixed seed
    seq = generator.flow(graphs=[0, 1, 2], batch_size=2, shuffle=True, seed=0)
    batches = get_batches(seq)
    at_least_one_different = False
    for _ in range(num_epochs_to_check):
        if not batches_all_equal(batches, get_next_epoch_batches(seq)):
            at_least_one_different = True
    assert at_least_one_different


def test_generator_flow_StellarGraphs():
    generator = PaddedGraphGenerator(graphs=graphs)
    graph_ilocs = [1, 2, 0]

    seq_1 = generator.flow(graph_ilocs)
    seq_2 = generator.flow([graphs[1], graphs[2], graphs[0]])

    assert len(seq_1) == len(seq_2) == 3

    for (values_1, targets_1), (values_2, targets_2) in zip(seq_1, seq_2):
        assert len(values_1) == len(values_2) == 3
        assert targets_1 is targets_2 is None

        for arr_1, arr_2 in zip(values_1, values_2):
            np.testing.assert_array_equal(arr_1, arr_2)


@pytest.mark.parametrize("use_targets", [False, True])
@pytest.mark.parametrize("use_ilocs", [False, True])
def test_generator_pairs(use_targets, use_ilocs):
    generator = PaddedGraphGenerator(graphs=graphs)

    targets = [12, 34, 56] if use_targets else None
    ilocs = [(1, 0), (0, 2), (2, 1)]
    input = ilocs if use_ilocs else [[graphs[x] for x in pair] for pair in ilocs]

    seq = generator.flow(input, targets=targets, batch_size=2)

    assert len(seq) == 2

    values_0, targets_0 = seq[0]
    assert len(values_0) == 6
    assert values_0[0].shape == (2, 6, 4)
    assert values_0[3].shape == (2, 6, 4)
    np.testing.assert_array_equal(values_0[1], [_mask(5, 6), _mask(6, 6)])
    np.testing.assert_array_equal(values_0[4], [_mask(6, 6), _mask(3, 6)])
    assert values_0[2].shape == (2, 6, 6)
    assert values_0[5].shape == (2, 6, 6)
    if use_targets:
        np.testing.assert_array_equal(targets_0, [12, 34])
    else:
        assert targets_0 is None

    values_1, targets_1 = seq[1]
    assert len(values_1) == 6
    assert values_1[0].shape == (1, 5, 4)
    assert values_1[3].shape == (1, 5, 4)
    np.testing.assert_array_equal(values_1[1], [_mask(3, 5)])
    np.testing.assert_array_equal(values_1[4], [_mask(5, 5)])
    assert values_1[2].shape == (1, 5, 5)
    assert values_1[5].shape == (1, 5, 5)
    if use_targets:
        np.testing.assert_array_equal(targets_1, [56])
    else:
        assert targets_1 is None
