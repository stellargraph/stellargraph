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


from stellargraph import StellarGraph
import networkx as nx
import pandas as pd
import numpy as np
import pytest


def node_features(seed=0) -> pd.DataFrame:
    random = np.random.RandomState(seed)
    node_data_np = random.rand(10, 10)
    return pd.DataFrame(node_data_np)


@pytest.fixture
def petersen_graph() -> StellarGraph:
    nxg = nx.petersen_graph()
    return StellarGraph(nxg, node_features=node_features())


@pytest.fixture
def simple_graph() -> StellarGraph:
    nxg = nx.MultiGraph()
    nxg.add_nodes_from(range(10))
    nxg.add_edges_from([(i, i + 1) for i in range(9)])
    return StellarGraph(nxg, node_features=node_features())
