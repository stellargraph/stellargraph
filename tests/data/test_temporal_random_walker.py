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
