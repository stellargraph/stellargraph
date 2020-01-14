# -*- coding: utf-8 -*-
#
# Copyright 2017-2020 Data61, CSIRO
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

import numpy as np
from collections import defaultdict
from stellargraph.data.unsupervised_sampler import UnsupervisedSampler
from ..test_utils.graphs import simple_graph


class TestUnsupervisedSampler(object):
    def test_UnsupervisedSampler_parameter(self, simple_graph):

        # if no graph is provided
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=None)

        # walk must have length strictly greater than 1
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=simple_graph, length=1)

        # at least 1 walk from each root node
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=simple_graph, number_of_walks=0)

        # nodes nodes parameter should be an iterable of node IDs
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=simple_graph, nodes=1)

        # if no root nodes are provided for sampling defaulting to using all nodes as root nodes
        sampler = UnsupervisedSampler(G=simple_graph, nodes=None)
        assert sampler.nodes == list(simple_graph.nodes())

        # if the seed value is provided check
        # that the random choices is reproducable
        sampler = UnsupervisedSampler(G=simple_graph, seed=1)
        assert sampler.random.choices(range(100), k=10) == [
            13,
            84,
            76,
            25,
            49,
            44,
            65,
            78,
            9,
            2,
        ]

    def test_run_batch_sizes(self, simple_graph):
        batch_size = 4
        sampler = UnsupervisedSampler(G=simple_graph, length=2, number_of_walks=2)
        batches = sampler.run(batch_size)

        # check batch sizes
        assert len(batches) == np.ceil(len(simple_graph.nodes()) * 4 / batch_size)
        for ids, labels in batches[:-1]:
            assert len(ids) == len(labels) == batch_size

        # last batch can be smaller
        ids, labels = batches[-1]
        assert len(ids) == len(labels)
        assert len(ids) <= batch_size

    def test_run_context_pairs(self, simple_graph):
        batch_size = 4
        sampler = UnsupervisedSampler(G=simple_graph, length=2, number_of_walks=2)
        batches = sampler.run(batch_size)

        grouped_by_target = defaultdict(list)

        for ids, labels in batches:
            for (target, context), label in zip(ids, labels):
                grouped_by_target[target].append((context, label))

        assert len(grouped_by_target) == len(simple_graph.nodes())

        for target, sampled in grouped_by_target.items():
            # exactly 2 positive and 2 negative context pairs for each target node
            assert len(sampled) == 4

            # since each walk has length = 2, there must be an edge between each positive context pair
            for context, label in sampled:
                if label == 1:
                    assert context in set(simple_graph.neighbors(target))
