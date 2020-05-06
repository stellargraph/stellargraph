# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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

import pandas as pd
import pytest
import numpy as np
import networkx as nx
from stellargraph.data.explorer import TemporalRandomWalk
from stellargraph.core.graph import StellarGraph


@pytest.fixture()
def temporal_graph():
    nodes = [1, 2, 3, 4, 5, 6]
    edges = [(1, 2, 5), (2, 3, 2), (2, 4, 10), (4, 5, 3), (4, 6, 12)]
    edge_cols = ["source", "target", "weight"]
    return StellarGraph(
        nodes=pd.DataFrame(index=nodes), edges=pd.DataFrame(edges, columns=edge_cols),
    )


def test_temporal_walks(temporal_graph):

    """
    valid time respecting walks (node -[time]-> node):

        1 -[2]-> 2 -[10]-> 4
        2 -[10]-> 4 -[12]-> 6
        3 -[2]-> 2 -[10]-> 4
        5 -[4]-> 4 -[12]-> 6
        1 -[2]-> 2 -[10]-> 4 -[12]-> 6
        3 -[2]-> 2 -[10]-> 4 -[12] -> 6
    """
    expected = {(1, 2, 4), (2, 4, 6), (3, 2, 4), (5, 4, 6), (1, 2, 4, 6), (3, 2, 4, 6)}

    rw = TemporalRandomWalk(temporal_graph)
    num_cw = 20  # how many walks to be sure we're getting valid temporal walks

    for walk in rw.run(num_cw=num_cw, cw_size=3, max_walk_length=4, seed=None):
        assert tuple(walk) in expected


def test_not_progressing_enough(temporal_graph):

    rw = TemporalRandomWalk(temporal_graph)
    cw_size = 5  # no valid temporal walks of this size

    with pytest.raises(RuntimeError, match=r".* discarded .*"):
        rw.run(num_cw=1, cw_size=cw_size, max_walk_length=cw_size, seed=None)


def test_exp_biases(temporal_graph):
    rw = TemporalRandomWalk(temporal_graph)
    times = np.array([1, 2, 3])
    t_0 = 1
    expected = np.exp(t_0 - times) / sum(np.exp(t_0 - times))
    biases = rw._exp_biases(times, t_0, decay=True)
    assert np.allclose(biases, expected)


def test_exp_biases_extreme(temporal_graph):
    rw = TemporalRandomWalk(temporal_graph)

    large_times = [100000, 100001]
    biases = rw._exp_biases(large_times, t_0=0, decay=True)
    assert sum(biases) == pytest.approx(1)

    small_times = [0.000001, 0.000002]
    biases = rw._exp_biases(small_times, t_0=0, decay=True)
    assert sum(biases) == pytest.approx(1)


@pytest.mark.parametrize("cw_size", [-1, 1, 2, 4])
def test_cw_size_and_walk_length(temporal_graph, cw_size):
    rw = TemporalRandomWalk(temporal_graph)
    num_cw = 5
    max_walk_length = 3

    def run():
        return rw.run(num_cw=num_cw, cw_size=cw_size, max_walk_length=max_walk_length)

    if cw_size < 2:
        with pytest.raises(ValueError, match=r".* context window size .*"):
            run()
    elif max_walk_length < cw_size:
        with pytest.raises(ValueError, match=r".* maximum walk length .*"):
            run()
    else:
        walks = run()
        num_cw_obtained = sum([len(walk) - cw_size + 1 for walk in walks])
        assert num_cw == num_cw_obtained
        assert max(map(len, walks)) <= max_walk_length


def test_init_parameters(temporal_graph):

    num_cw = 5
    cw_size = 3
    max_walk_length = 3
    seed = 0

    rw = TemporalRandomWalk(
        temporal_graph, cw_size=cw_size, max_walk_length=max_walk_length, seed=seed
    )
    rw_no_params = TemporalRandomWalk(temporal_graph)

    run_1 = rw.run(num_cw=num_cw)
    run_2 = rw_no_params.run(
        num_cw=num_cw, cw_size=cw_size, max_walk_length=max_walk_length, seed=seed
    )

    assert np.array_equal(run_1, run_2)
