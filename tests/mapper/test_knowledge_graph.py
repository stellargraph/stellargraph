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

import pytest

import pandas as pd
import numpy as np

from stellargraph.mapper.knowledge_graph import KGTripleGenerator, KGTripleSequence

from .. import test_utils
from ..test_utils.graphs import knowledge_graph

pytestmark = test_utils.ignore_stellargraph_experimental_mark


def check_sequence_output(
    output,
    batch_size,
    negatives,
    max_node_iloc=None,
    source_ilocs=None,
    rel_ilocs=None,
    target_ilocs=None,
):
    s, r, o = output[0]
    l = output[1] if len(output) == 2 else None

    expected_length = batch_size * (1 + (negatives or 0))
    assert len(s) == len(r) == len(o) == expected_length

    if source_ilocs is not None:
        expected_pos = set(zip(source_ilocs, rel_ilocs, target_ilocs))
        actual_pos = set(zip(s[:batch_size], r[:batch_size], o[:batch_size]))
        # the positive edges should all be edges passed in originally
        assert actual_pos <= expected_pos

    if negatives is None:
        assert l is None
    else:
        assert len(l) == expected_length
        assert set(l[:batch_size]) == {1}
        assert set(l[batch_size:]) == {0}

        if max_node_iloc is not None:
            assert np.all((0 <= s) & (s <= max_node_iloc))
            assert np.all((0 <= r) & (r <= max_node_iloc))


def triple_df(*values):
    return pd.DataFrame(values, columns=["source", "label", "target"])


def test_kg_triple_generator(knowledge_graph):
    gen = KGTripleGenerator(knowledge_graph, 2)

    edges = triple_df(("a", "W", "b"), ("c", "X", "a"), ("d", "Y", "c"))

    seq = gen.flow(edges)
    check_sequence_output(seq[0], 2, None)
    check_sequence_output(seq[1], 1, None)

    seq = gen.flow(edges, negative_samples=10)
    check_sequence_output(seq[0], 2, 10, knowledge_graph.number_of_nodes())


def test_kg_triple_generator_errors(knowledge_graph):
    gen = KGTripleGenerator(knowledge_graph, 10)

    with pytest.raises(TypeError, match="edges: expected.*found int"):
        gen.flow(1)

    with pytest.raises(KeyError, match="fake"):
        gen.flow(triple_df(("fake", "W", "b")))

    with pytest.raises(KeyError, match="fake"):
        gen.flow(triple_df(("a", "fake", "b")))

    with pytest.raises(KeyError, match="fake"):
        gen.flow(triple_df(("a", "W", "fake")))

    with pytest.raises(TypeError, match="negative_samples: expected.*found str"):
        gen.flow(triple_df(), negative_samples="foo")

    with pytest.raises(ValueError, match="negative_samples: expected.*found -1"):
        gen.flow(triple_df(), negative_samples=-1)


@pytest.mark.parametrize("negative_samples", [None, 1, 10])
def test_kg_triple_sequence_batches(negative_samples):
    s = [0, 1, 2, 3, 4]
    r = [5, 6, 7, 8, 9]
    t = [10, 11, 12, 13, 14]
    seq = KGTripleSequence(
        max_node_iloc=20,
        source_ilocs=s,
        rel_ilocs=r,
        target_ilocs=t,
        batch_size=3,
        shuffle=False,
        negative_samples=negative_samples,
        seed=None,
    )
    assert len(seq) == 2
    check_sequence_output(seq[0], 3, negative_samples, 20, s, r, t)
    check_sequence_output(seq[1], 2, negative_samples, 20, s, r, t)


def epoch_sample_equal(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a[0], b[0]))


@pytest.mark.parametrize("shuffle", [False, True])
def test_kg_triple_sequence_shuffle(shuffle):
    seq = KGTripleSequence(
        max_node_iloc=10,
        source_ilocs=[0, 1, 2, 3, 4],
        rel_ilocs=[0, 1, 0, 1, 0],
        target_ilocs=[4, 3, 2, 1, 0],
        batch_size=5,
        shuffle=shuffle,
        negative_samples=None,
        seed=None,
    )
    assert len(seq) == 1

    def sample():
        ret = seq[0]
        seq.on_epoch_end()
        return ret

    # with 20 epochs, it's very unlikely ((1/5!)**20 ≈ 2.6e-42) they will all be the same, if
    # (uniform) shuffling is happening
    first, *rest = [sample() for _ in range(20)]

    should_be_equal = not shuffle
    assert all(epoch_sample_equal(first, r) for r in rest) == should_be_equal


def test_kg_triple_sequence_negative_samples():
    max_node_iloc = 1234567
    negative_sampless = 100
    s = [0, 1]
    r = [2, 3]
    t = [4, 5]
    seq = KGTripleSequence(
        max_node_iloc=max_node_iloc,
        source_ilocs=s,
        rel_ilocs=r,
        target_ilocs=t,
        batch_size=2,
        shuffle=False,
        negative_samples=negative_sampless,
        seed=None,
    )

    sample = seq[0]
    check_sequence_output(sample, 2, negative_sampless, max_node_iloc, s, r, t)


def test_kg_triple_sequence_seed_shuffle_negative_samples():
    def mk(seed):
        return KGTripleSequence(
            max_node_iloc=10000,
            source_ilocs=[0, 1],
            rel_ilocs=[2, 3],
            target_ilocs=[4, 5],
            batch_size=1,
            shuffle=True,
            negative_samples=5,
            seed=seed,
        )

    def run(a, b):
        sample_a = a[0]
        sample_b = b[0]
        a.on_epoch_end()
        b.on_epoch_end()
        return epoch_sample_equal(sample_a, sample_b)

    # the same seed should give the same sequence
    seq0_1 = mk(0)
    seq0_2 = mk(0)
    assert all(run(seq0_1, seq0_2) for _ in range(20))

    # different seeds should give different sequences
    seq1 = mk(1)
    seq2 = mk(2)
    assert not all(run(seq1, seq2) for _ in range(20))
