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
import numpy as np

from stellargraph.mapper import *
from ..test_utils.graphs import example_graph_random


class DummyGenerator(Generator):
    def __init__(self, indices, batch_dims=1, data=[], check_flow_args=False):
        self.indices = indices
        self.batch_dims = batch_dims
        self.data = data
        self.check_flow_args = check_flow_args

    def flow(self, *args, **kwargs):
        if self.check_flow_args:
            # validate that CorruptedGenerator passes through the arguments right
            assert args == ("some",)
            assert kwargs == {"args": 1}

        return [(x, None) for x in self.data]

    def num_batch_dims(self):
        return self.batch_dims

    def default_corrupt_input_index_groups(self):
        return self.indices


def test_corrupted_generator_invalid():
    with pytest.raises(
        TypeError, match="base_generator: expected a Generator subclass, found NoneType"
    ):
        CorruptedGenerator(None)

    with pytest.raises(
        TypeError,
        match="base_generator: expected a Generator that supports corruption .* found DummyGenerator",
    ):
        CorruptedGenerator(DummyGenerator(None))

    # incorrect type for the indices, also checking that get the indices out of the base generator
    # or from the top-level parameter are validated the same
    with pytest.raises(
        TypeError, match="corrupt_index_groups: expected list or tuple, found str"
    ):
        CorruptedGenerator(DummyGenerator("foo"))

    with pytest.raises(
        TypeError, match="corrupt_index_groups: expected list or tuple, found str"
    ):
        CorruptedGenerator(DummyGenerator(None), corrupt_index_groups="foo")

    with pytest.raises(
        TypeError,
        match="corrupt_index_groups: expected each group to be a list or tuple, found str for group number 0",
    ):
        CorruptedGenerator(DummyGenerator(["foo"]))

    # incorrect contents in the indices, in various forms:

    with pytest.raises(
        TypeError,
        match=r"corrupt_index_groups: expected each index to be a non-negative integer, found str \('foo'\) in group number 1",
    ):
        CorruptedGenerator(DummyGenerator([[0], ["foo"]]))

    with pytest.raises(
        TypeError,
        match=r"corrupt_index_groups: expected each index to be a non-negative integer, found int \(-123\) in group number 2",
    ):
        CorruptedGenerator(DummyGenerator([[123], [0], [1, -123]]))

    with pytest.raises(
        ValueError,
        match="corrupt_index_groups: expected each group to have at least one index, found empty group number 3",
    ):
        CorruptedGenerator(DummyGenerator([[0], [1], [2], []]))

    with pytest.raises(
        ValueError,
        match=r"corrupt_index_groups: expected each index to appear at most once, found two occurrences of 123 \(in group numbers 0 and 0\)",
    ):
        CorruptedGenerator(DummyGenerator([[123, 123]]))

    with pytest.raises(
        ValueError,
        match=r"corrupt_index_groups: expected each index to appear at most once, found two occurrences of 123 \(in group numbers 0 and 1\)",
    ):
        CorruptedGenerator(DummyGenerator([[123], [1, 123]]))


def _data(num, shape=(3, 4, 5, 6)):
    per_array = np.product(shape)
    return np.arange(per_array * num, per_array * (num + 1)).reshape(shape)


def test_corrupted_invalid_index():
    batch = [_data(0), _data(1)]
    corr_gen = CorruptedGenerator(DummyGenerator([[0], [2, 1, 3]], data=[batch]))

    corr_seq = corr_gen.flow()

    with pytest.raises(
        ValueError,
        match=r"corrupt_index_groups \(group number 1\): expected valid .* 2 input tensors, found some too large: 2, 3",
    ):
        corr_seq[0]


def test_corrupted_flow_args():
    base_gen = DummyGenerator([[0]], check_flow_args=True)
    corr_gen = CorruptedGenerator(base_gen)
    # this hooks into the asserts in `DummyGenerator`
    corr_gen.flow("some", args=1)


def _rank2(array):
    return array.reshape((-1, array.shape[-1]))


def _sorted_feats(arrays):
    flat = np.concatenate([_rank2(arr) for arr in arrays])
    return np.sort(flat, axis=0)


@pytest.mark.parametrize("group_param", ["default", "override"])
def test_corrupted_groups(group_param):
    batch = [_data(0), _data(1), _data(2), _data(3, shape=(3, 10, 10, 6))]
    groups = [[0], [2, 3]]

    if group_param == "default":
        base_gen = DummyGenerator(groups, batch_dims=1, data=[batch])
        corr_gen = CorruptedGenerator(base_gen)
    else:
        base_gen = DummyGenerator(None, batch_dims=1, data=[batch])
        corr_gen = CorruptedGenerator(base_gen, corrupt_index_groups=groups)

    corr_seq = corr_gen.flow()
    assert len(corr_seq) == 1

    (corr0, corr2, corr3, *orig), targets = corr_seq[0]
    assert len(orig) == len(batch)
    for o, b in zip(orig, batch):
        np.testing.assert_array_equal(o, b)

    np.testing.assert_array_equal(_sorted_feats([corr0]), _sorted_feats([orig[0]]))
    np.testing.assert_array_equal(
        _sorted_feats([corr2, corr3]), _sorted_feats([orig[2], orig[3]])
    )
    # check that the corruption move some things between the groups (this is exceedingly likely to
    # happen, with this size of data)
    assert np.any(_sorted_feats([corr2]) != _sorted_feats([orig[2]]))
    assert np.any(_sorted_feats([corr3]) != _sorted_feats([orig[3]]))

    np.testing.assert_array_equal(targets, [[1, 0], [1, 0], [1, 0]])


@pytest.mark.parametrize("batch_dims", [1, 2, 3])
def test_corrupted_batching(batch_dims):
    batch0_shape = (10, 4, 5, 6)
    batch0 = [_data(0, shape=batch0_shape)]
    # different batch sizes
    batch1_shape = (5, 4, 5, 6)
    batch1 = [_data(1, shape=batch1_shape)]

    base_gen = DummyGenerator([[0]], batch_dims=batch_dims, data=[batch0, batch1])
    corr_gen = CorruptedGenerator(base_gen)

    corr_seq = corr_gen.flow()
    assert len(corr_seq) == 2

    def check(index, shape):
        (corr, orig), targets = corr_seq[index]

        assert targets.shape == (*shape[:batch_dims], 2)
        np.testing.assert_array_equal(targets[..., 0], 1)
        np.testing.assert_array_equal(targets[..., 1], 0)

    check(0, batch0_shape)
    check(1, batch1_shape)


@pytest.mark.parametrize("num_nodes", [10, 20])
@pytest.mark.parametrize("sparse", [True, False])
def test_corrupt_full_batch_generator(sparse, num_nodes):

    G = example_graph_random(n_nodes=20)

    generator = FullBatchNodeGenerator(G, sparse=sparse)

    nodes = G.nodes()[:num_nodes]
    gen = CorruptedGenerator(generator).flow(nodes)

    [shuffled_feats, features, *_], targets = gen[0]

    assert features.shape == shuffled_feats.shape

    # check shuffled_feats are feats
    assert not np.array_equal(features, shuffled_feats)

    # check that all feature vecs in shuffled_feats correspond to a feature vec in features
    np.testing.assert_array_equal(
        _sorted_feats([shuffled_feats]), _sorted_feats([features])
    )

    assert targets.shape == (1, num_nodes, 2)


@pytest.mark.parametrize("is_directed", [True, False])
def test_corrupt_graphsage_generator(is_directed):

    G = example_graph_random(n_nodes=20, is_directed=is_directed)

    if is_directed:
        generator = DirectedGraphSAGENodeGenerator(
            G, batch_size=5, in_samples=[2, 3], out_samples=[4, 1]
        )
    else:
        generator = GraphSAGENodeGenerator(G, batch_size=5, num_samples=[2, 3])

    base_gen = generator.flow(G.nodes())
    gen = CorruptedGenerator(generator).flow(G.nodes())

    x, targets = gen[0]
    clean_feats, _ = base_gen[0]

    shuffled_feats = x[: (len(x) // 2)]
    features = x[(len(x) // 2) :]

    assert len(clean_feats) == len(features)
    assert len(x) == 2 * len(clean_feats)
    assert len(features) == len(shuffled_feats)
    assert all(f.shape == s.shape for f, s in zip(features, shuffled_feats))

    # check shuffled_feats do seem to be shuffled
    assert not all(
        np.array_equal(_sorted_feats([shuf]), _sorted_feats(feat))
        for shuf, feat in zip(shuffled_feats, features)
    )

    # check that all feature vecs in shuffled_feats correspond to a feature vec in features
    np.testing.assert_array_equal(
        _sorted_feats(shuffled_feats), _sorted_feats(features)
    )
