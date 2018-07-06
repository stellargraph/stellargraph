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

import pytest
import networkx as nx
from stellar.data.stellargraph import *

def test_graph_constructor():
    sg = StellarGraph()
    assert sg.is_directed() == False

def test_digraph_constructor():
    sg = StellarDiGraph()
    assert sg.is_directed() == True

def test_graph_schema():
    sg = StellarGraph()
    sg.add_nodes_from([0,1,2,3], label='movie')
    sg.add_nodes_from([4,5], label='user')
    sg.add_edges_from([(0,4), (1,4), (1,5), (2,4), (3,5)], label='rating')

    schema = sg.create_graph_schema()

    assert 'movie' in schema.schema
    assert 'user' in schema.schema
    assert len(schema.schema['movie']) == 1
    assert len(schema.schema['user']) == 1

def test_digraph_schema():
    sg = StellarDiGraph()
    sg.add_nodes_from([0, 1, 2, 3], label='movie')
    sg.add_nodes_from([4, 5], label='user')
    sg.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label='rating')

    schema = sg.create_graph_schema()

    assert 'movie' in schema.schema
    assert 'user' in schema.schema
    assert len(schema.schema['movie']) == 1
    assert len(schema.schema['user']) == 0
