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
from stellargraph.core.graph import *
from stellargraph.mapper.graph_generator import GraphGenerator, GraphSequence

import numpy as np
import pytest
from ..test_utils.graphs import example_graph_random, example_hin_1


class Test_GraphGenerator:

    graphs = [
        example_graph_random(feature_size=4, n_nodes=6),
        example_graph_random(feature_size=4, n_nodes=5),
        example_graph_random(feature_size=4, n_nodes=3),
    ]

    graphs_nx = [
        example_graph_random(feature_size=4, n_nodes=3, is_directed=False),
        example_graph_random(
            feature_size=4, n_nodes=2, is_directed=False
        ).to_networkx(),
    ]

    graphs_diff_num_features = [
        example_graph_random(feature_size=2, n_nodes=6),
        example_graph_random(feature_size=4, n_nodes=5),
    ]

    graphs_mixed = [
        example_graph_random(feature_size=2, n_nodes=6),
        example_hin_1(is_directed=False),
    ]

    def test_generator_init(self):
        generator = GraphGenerator(graphs=self.graphs)
        assert len(generator.graphs) == len(self.graphs)

    def test_generator_init_different_feature_numbers(self):
        with pytest.raises(ValueError):
            generator = GraphGenerator(graphs=self.graphs_diff_num_features)

    def test_generator_init_nx_graph(self):
        with pytest.raises(TypeError):
            generator = GraphGenerator(graphs=self.graphs_nx)

    def test_generator_init_hin(self):
        with pytest.raises(ValueError):
            generator = GraphGenerator(graphs=self.graphs_mixed)

    def test_generator_flow_incorrect_targets(self):

        generator = GraphGenerator(graphs=self.graphs)

        with pytest.raises(ValueError):
            generator.flow(graph_ilocs=[0, 1], targets=np.array([0]))

        with pytest.raises(TypeError):
            generator.flow(graph_ilocs=[0, 1], targets=1)

    def test_generator_flow_no_targets(self):

        generator = GraphGenerator(graphs=self.graphs)

        seq = generator.flow(graph_ilocs=[0, 1, 2], batch_size=2)
        assert isinstance(seq, GraphSequence)

        assert len(seq) == 2  # two batches

        # The first batch should be size 2 and the second batch size 1
        batch_0 = seq[0]
        assert batch_0[0][0].shape[0] == 2
        assert batch_0[0][1].shape[0] == 2
        assert batch_0[0][2].shape[0] == 2
        assert batch_0[1] is None

        batch_1 = seq[1]
        assert batch_1[0][0].shape[0] == 1
        assert batch_1[0][1].shape[0] == 1
        assert batch_1[0][2].shape[0] == 1
        assert batch_1[1] is None

    def test_generator_flow_check_padding(self):

        generator = GraphGenerator(graphs=self.graphs)

        seq = generator.flow(graph_ilocs=[0, 2], batch_size=2)
        assert isinstance(seq, GraphSequence)

        assert len(seq) == 1

        # The largest graph has 6 nodes vs 3 for the smallest one.
        # Check that the data matrices have the correct size 6
        batch = seq[0]

        assert batch[0][0].shape[1] == 6
        assert batch[0][1].shape[1] == 6
        assert batch[0][2].shape[1] == 6

    def test_generator_flow_with_targets(self):

        generator = GraphGenerator(graphs=self.graphs)

        seq = generator.flow(graph_ilocs=[1, 2], targets=np.array([0, 1]), batch_size=1)
        assert isinstance(seq, GraphSequence)

        for batch in seq:
            assert batch[0][0].shape[0] == 1
            assert batch[0][1].shape[0] == 1
            assert batch[0][2].shape[0] == 1
            assert batch[1].shape[0] == 1
