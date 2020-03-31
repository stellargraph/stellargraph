# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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

from stellargraph.datasets import *


def test_ogbn_proteins_load():
    graph, labels, train_idx, valid_idx, test_idx = OgbNProteins().load()

    print(graph.info())  # debugging

    nodes = 132534
    edges = 39561252 * 2  # undirected

    assert graph.number_of_nodes() == nodes
    assert graph.number_of_edges() == edges

    assert labels.shape == (nodes, 112)
    assert len(train_idx) + len(valid_idx) + len(test_idx) == nodes


def test_ogbn_products_load():
    graph, labels, train_idx, valid_idx, test_idx = OgbNProteins().load()

    print(graph.info())  # debugging

    nodes = 2449029
    edges = 61859140 * 2  # undirected

    assert graph.number_of_nodes() == nodes
    assert graph.number_of_edges() == edges

    assert labels.shape == (nodes, 1)
    assert len(train_idx) + len(valid_idx) + len(test_idx) == nodes


def test_ogbg_molhiv_load():
    # graphs, labels, train_idx, valid_idx, test_idx = OgbGMolHIV().load()
    pass
