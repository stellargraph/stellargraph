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
from stellargraph.data.explorer import UniformRandomWalk
from ..test_utils.graphs import line_graph


def test_init_parameters(line_graph):

    # if no graph is provided
    with pytest.raises(ValueError):
        UnsupervisedSampler(G=None)

    # walk must have length strictly greater than 1
    with pytest.raises(ValueError):
        UnsupervisedSampler(G=line_graph, length=1)

    # at least 1 walk from each root node
    with pytest.raises(ValueError):
        UnsupervisedSampler(G=line_graph, number_of_walks=0)

    # nodes nodes parameter should be an iterable of node IDs
    with pytest.raises(ValueError):
        UnsupervisedSampler(G=line_graph, nodes=1)

    # if no root nodes are provided for sampling defaulting to using all nodes as root nodes
    sampler = UnsupervisedSampler(G=line_graph, nodes=None)
    assert sampler.nodes == list(line_graph.nodes())


def test_run_batch_sizes(line_graph):
    batch_size = 4
    sampler = UnsupervisedSampler(G=line_graph, length=2, number_of_walks=2)
    batches = sampler.run(batch_size)

    # check batch sizes
    assert len(batches) == np.ceil(len(line_graph.nodes()) * 4 / batch_size)
    for ids, labels in batches[:-1]:
        assert len(ids) == len(labels) == batch_size

    # last batch can be smaller
    ids, labels = batches[-1]
    assert len(ids) == len(labels)
    assert len(ids) <= batch_size


def test_run_context_pairs(line_graph):
    batch_size = 4
    sampler = UnsupervisedSampler(G=line_graph, length=2, number_of_walks=2)
    batches = sampler.run(batch_size)

    grouped_by_target = defaultdict(list)

    for ids, labels in batches:
        for (target, context), label in zip(ids, labels):
            grouped_by_target[target].append((context, label))

    assert len(grouped_by_target) == len(line_graph.nodes())

    for target, sampled in grouped_by_target.items():
        # exactly 2 positive and 2 negative context pairs for each target node
        assert len(sampled) == 4

        # since each walk has length = 2, there must be an edge between each positive context pair
        for context, label in sampled:
            if label == 1:
                assert context in set(line_graph.neighbors(target))


def test_walker_uniform_random(line_graph):
    length = 3
    number_of_walks = 2
    batch_size = 4

    walker = UniformRandomWalk(line_graph, n=number_of_walks, length=length)
    sampler = UnsupervisedSampler(line_graph, walker=walker)

    batches = sampler.run(batch_size)

    # batches should match the parameters used to create the walker object, instead of the defaults
    # for UnsupervisedSampler
    expected_num_batches = np.ceil(
        line_graph.number_of_nodes() * number_of_walks * (length - 1) * 2 / batch_size
    )
    assert len(batches) == expected_num_batches


class CustomWalker:
    def run(self, nodes):
        return [[node, node] for node in nodes]


def test_walker_custom(line_graph):
    walker = CustomWalker()
    sampler = UnsupervisedSampler(line_graph, walker=walker)
    batches = sampler.run(2)

    assert len(batches) == line_graph.number_of_nodes()

    # all positive examples should be self loops, since we defined our custom walker this way
    for context_pairs, labels in batches:
        for node, neighbour in context_pairs[labels == 1]:
            assert node == neighbour


def test_ignored_param_warning(line_graph):
    walker = UniformRandomWalk(line_graph, n=2, length=3)
    with pytest.raises(ValueError, match="cannot specify both 'walker' and 'length'"):
        UnsupervisedSampler(line_graph, walker=walker, length=5)

    with pytest.raises(
        ValueError, match="cannot specify both 'walker' and 'number_of_walks'"
    ):
        UnsupervisedSampler(line_graph, walker=walker, number_of_walks=5)

    with pytest.raises(ValueError, match="cannot specify both 'walker' and 'seed'"):
        UnsupervisedSampler(line_graph, walker=walker, seed=1)
