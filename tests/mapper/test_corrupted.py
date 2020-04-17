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


@pytest.mark.parametrize("sparse", [True, False])
def test_corrupt_full_batch_generator(sparse):

    G = example_graph_random(n_nodes=20)

    generator = FullBatchNodeGenerator(G, sparse=sparse)

    base_gen = generator.flow(G.nodes())
    gen = CorruptedNodeSequence(base_gen)

    [shuffled_feats, features, *_], targets = gen[0]

    assert features.shape == shuffled_feats.shape

    # check shuffled_feats are feats
    assert not np.array_equal(features, shuffled_feats)

    # check that all feature vecs in shuffled_feats correspond to a feature vec in features
    assert all(
        any(
            np.array_equal(shuffled_feats[:, i, :], features[:, j, :])
            for j in range(features.shape[1])
        )
        for i in range(shuffled_feats.shape[1])
    )


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
    gen = CorruptedNodeSequence(base_gen)

    x, targets = gen[0]
    clean_feats, _ = base_gen[0]

    shuffled_feats = x[: (len(x) // 2)]
    features = x[(len(x) // 2) :]

    assert len(clean_feats) == len(features)
    assert len(x) == 2 * len(clean_feats)
    assert len(features) == len(shuffled_feats)
    assert all(f.shape == s.shape for f, s in zip(features, shuffled_feats))

    features = np.concatenate(features, axis=1).reshape(-1, features[0][0].shape[-1])
    shuffled_feats = np.concatenate(shuffled_feats, axis=1).reshape(
        -1, features.shape[-1]
    )

    # check shuffled_feats are feats
    assert not np.array_equal(features, shuffled_feats)

    # check that all feature vecs in shuffled_feats correspond to a feature vec in features
    assert all(
        any(
            np.array_equal(shuffled_feats[i, :], features[j, :])
            for j in range(features.shape[0])
        )
        for i in range(shuffled_feats.shape[0])
    )
