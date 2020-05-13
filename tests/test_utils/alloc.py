# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
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
        # By collecting, we ensure we're only recording long term objects; this works best when GC
        # is disabled, see the comment in `allocation_benchmark` below.
        gc.collect(gc_generation)
    return MallocInstant(tracemalloc.take_snapshot())


class MallocPeak:
    """
    This class wraps peak memory usage in a way that pretends to be a clock.

    The __sub__ (-) operator ignores the "before" argument, because this assumes that each peak
    recorded independently, by calling `tracemalloc.clear_traces()` to reset the traces.
    """

    def __init__(self, peak: int):
        self._peak = peak

    def __sub__(self, other: "MallocPeak") -> int:
        return self._peak


def peak() -> int:
    """
    Take a snapshot of the current "peak memory usage", similar to an allocation version of time.perf_counter() (etc.).
    """
    _, peak = tracemalloc.get_traced_memory()
    # reset the peak for the next recording
    tracemalloc.clear_traces()
    return MallocPeak(peak)


@pytest.fixture
def allocation_benchmark(request, benchmark):
    # make sure the user specified the "snapshot" timer
    marker = request.node.get_closest_marker("benchmark")
    options = dict(marker.kwargs) if marker else {}
    allowed_timers = [snapshot, peak]
    timer = options.get("timer")
    if timer not in allowed_timers:
        allowed_str = ", ".join(
            f"{__name__}.{allowed.__name__}" for allowed in allowed_timers
        )
        raise ValueError(
            f"allocation_benchmark fixture can only be used in functions with @pytest.mark.benchmark(..., timer=T, ...) where T is one of: {allowed_str}"
        )

    # Put a note into the saved JSON files, so that future analysis can tell that these are special
    # and the "times" aren't actually times.
    benchmark.extra_info["allocation_benchmark"] = True

    def run_it(f):
        result = benchmark.pedantic(f, iterations=1, rounds=20, warmup_rounds=3)
        if result is None and timer is snapshot:
            raise ValueError(
                "benchmark function returned None: allocation benchmarking with 'snapshot' is only reliable if the object(s) of interest is created inside and returned from the function being benchmarked"
            )

    # Running with GC disabled for an memory benchmark?
    #
    # These benchmarks are designed to measure only the memory use of residual/long-lived objects
    # created by each measurement run. They thus need to clean up any short lived objects before
    # recording the final memory use (in `snapshot`), which can be done with `gc.collect`. Python's
    # GC has 3 generations (called 0 (young), 1, 2): `gc.collect(0)` (collecting gen 0) is generally
    # faster than `gc.collect(1)` (gens 0 and 1), and that's faster than
    # `gc.collect()`/`gc.collect(2)` (a full collection of gens 0, 1 and 2). Thus, the benchmarks
    # will run fastest if they can get away with `gc.collect(0)`. Objects start by being allocated
    # into gen 0, and every time they survive a collection (that is, are still reachable when the
    # collection occurs) they move to the next generation. That is, objects move out of gen 0 when
    # they survive their first collection.
    #
    # In summary, the benchmarks can use the fastest `gc.collect(0)` to clear out short-lived
    # objects at the end of a measurement run (just before recording the memory use in `snapshot`)
    # if no collections occur while the benchmark runs, and this can be guaranteed by disabling
    # automatic collection.
    #
    # Disabling GC is beneficial for measuring the peak memory use too, if that was implemented
    # (e.g. via tracemalloc.get_traced_memory), as it gives us the worst-case peak: the case when no
    # GC runs at all during the benchmarked-function and so only obvious deallocations occur.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    tracemalloc.start()
    yield run_it
    tracemalloc.stop()

    if gc_was_enabled:
        gc.enable()

    # This benchmark is finished, so now clean up any longer-lived garbage left over from the
    # small/partial collections of short-term objects that happened during the benchmark (in
    # `snapshot`).
    gc.collect()
