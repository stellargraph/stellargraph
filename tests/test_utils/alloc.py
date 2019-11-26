# -*- coding: utf-8 -*-
#
# Copyright 2017-2018 Data61, CSIRO
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

import gc
import pytest
import tracemalloc


class MallocInstant:
    """
    This class wraps allocation tracing in a way that pretends to be a
    clock, that is, one can ask for "now" snapshots, and then
    difference them to work out the "elapsed" allocations (that is,
    the total size difference) between them.

    The __sub__ (-) operator counts both allocations (positive) and
    frees (negative) between the two "MallocInstant"s, and so gives
    the delta in the size of the allocations, not the total size of
    new allocations.
    """

    def __init__(self, snapshot: tracemalloc.Snapshot):
        self._snapshot = snapshot

    def __sub__(self, other: "MallocInstant") -> int:
        diff = self._snapshot.compare_to(other._snapshot, "lineno")
        return sum(elem.size_diff for elem in diff)


def snapshot() -> MallocInstant:
    """
    Take a snapshot of the current "malloc instant", similar to an allocation version of time.perf_counter() (etc.).
    """
    return MallocInstant(tracemalloc.take_snapshot())


@pytest.fixture
def allocation_benchmark(request, benchmark):
    # make sure the user specified the "snapshot" timer
    marker = request.node.get_closest_marker("benchmark")
    options = dict(marker.kwargs) if marker else {}
    correct_timer = "timer" in options and options["timer"] is snapshot
    if not correct_timer:
        raise ValueError(
            "allocation_benchmark fixture can only be used in functions with @pytest.mark.benchmark(..., timer=%s.%s, ...)"
            % (__name__, snapshot.__name__)
        )

    benchmark.extra_info["allocation_benchmark"] = True

    def setup():
        # this is somewhat expensive, but dramatically increases
        # consistency, by ensuring there's not many deallocations of
        # "random" other objects that happen during the benchmark
        # period: the standard deviation drops by 10-100x and IQR
        # drops to 0, typically.
        gc.collect()

    def run_it(f):
        result = benchmark.pedantic(f, setup=setup, iterations=1, rounds=10)
        if result is None:
            raise ValueError(
                "benchmark function returned None: allocation benchmarking is only reliable if the object(s) of interest is created inside and returned from the function being benchmarked"
            )

    tracemalloc.start()
    yield run_it
    tracemalloc.stop()
