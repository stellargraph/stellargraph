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

import numpy as np
import pytest
import networkx as nx
from stellargraph.data.explorer import TemporalRandomWalk
from stellargraph.core.graph import StellarGraph


class TestTemporalRandomWalk(object):
    def test_temporal_walks(self):

        g = nx.MultiGraph()
        edges = [(1, 2, 5), (2, 3, 2), (2, 4, 10), (4, 5, 3), (4, 6, 12)]

        """
        valid time respecting walks (node -[time]-> node):

            1 -[2]-> 2 -[10]-> 4
            2 -[10]-> 4 -[12]-> 6
            3 -[2]-> 2 -[10]-> 4
            5 -[4]-> 4 -[12]-> 6
        """
        expected = {(1, 2, 4), (2, 4, 6), (3, 2, 4), (5, 4, 6)}

        g.add_weighted_edges_from(edges)
        g = StellarGraph(g)

        temporalrw = TemporalRandomWalk(g)
        num_cw = 10  # how many walks to obtain

        for walk in temporalrw.run(
            num_cw=num_cw, cw_size=3, max_walk_length=3, seed=None
        ):
            assert tuple(walk) in expected
