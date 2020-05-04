# -*- coding: utf-8 -*-
#
# Copyright 2017-2020 Data61, CSIRO
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
import numpy as np
from stellargraph.core.element_data import ExternalIdIndex


@pytest.mark.parametrize(
    "count,expected_missing",
    [(0, 0xFF), (255, 0xFF), (256, 0xFFFF), (65535, 0xFFFF), (65536, 0xFFFF_FFFF)],
)
def test_external_id_index_to_iloc(count, expected_missing):
    values = [f"id{x}" for x in range(count)]
    idx = ExternalIdIndex(values)

    all_ilocs = idx.to_iloc(values)
    assert (all_ilocs == list(range(count))).all()
    # the value for a missing ID should be larger than everything else, so that indexing with it
    # into an array indexed by the current ExternalIdIndex will fail
    assert (all_ilocs < expected_missing).all()

    if count <= 256:
        # only do individual lookups when there's a few IDs, and assume that if those work, then large ones will too
        for i, x in enumerate(values):
            assert np.array_equal(idx.to_iloc([x]), [i])

    # missing value
    assert idx.to_iloc(["A"]) == expected_missing


def test_benchmark_external_id_index_from_iloc(benchmark):
    N = 1000
    SIZE = 100
    idx = ExternalIdIndex(np.arange(N))
    x = np.random.randint(0, N, size=SIZE)

    def f():
        idx.from_iloc(x)

    benchmark(f)
