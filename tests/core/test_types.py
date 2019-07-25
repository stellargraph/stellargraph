# -*- coding: utf-8 -*-
#
# Copyright 2018-2019 Data61, CSIROß
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


"""
Utils tests:

"""
import pytest
import random
import networkx as nx
import numpy as np
import scipy as sp

from stellargraph.core.utils import *
from stellargraph.core.graph import *


def example_graph(label=None):
    G = StellarGraph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]
    if label:
        G.add_edges_from(elist, label=label)
    else:
        G.add_edges_from(elist)
    return G


def test_edges_with_type():
    G = example_graph("edge")
    info = G.info()


def test_edges_without_type():
    G = example_graph()
    info = G.info()
