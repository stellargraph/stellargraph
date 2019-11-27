# -*- coding: utf-8 -*-
#
# Copyright 2019 Data61, CSIRO
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


def snapshot(gc_generation=0) -> MallocInstant:
    """
    Take a snapshot of the current "malloc instant", similar to an allocation version of time.perf_counter() (etc.).
    """
    if gc_generation is not None:
        gc.collect(gc_generation)
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

    def run_it(f):
        result = benchmark.pedantic(f, iterations=1, rounds=20, warmup_rounds=3)
        if result is None:
            raise ValueError(
                "benchmark function returned None: allocation benchmarking is only reliable if the object(s) of interest is created inside and returned from the function being benchmarked"
            )

    # Running without a GC for an memory benchmark? This ensures that all objects created in the
    # benchmarked function get placed into the young generation (0), and so cleaning those up to
    # leave only the long-lived objects from each function execution is just a `gc.collect(0)` which
    # is very fast (much faster than `gc.collect()`). For measuring peak memory use this gives the
    # worst-case peak: the case when no GC runs at all during the benchmarked-function and so only
    # obvious deallocations occur.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    tracemalloc.start()
    yield run_it
    tracemalloc.stop()

    if gc_was_enabled:
        gc.enable()
        # forcibly clean up anything left over in this benchmark
        gc.collect()
