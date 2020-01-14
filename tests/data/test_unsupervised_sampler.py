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

from stellargraph.data.unsupervised_sampler import UnsupervisedSampler
from stellargraph.core.graph import StellarGraph

from ..test_utils.graphs import example_graph_2, create_test_graph_nx


class TestUnsupervisedSampler(object):
    def test_UnsupervisedSampler_parameter(self):

        g = create_test_graph_nx()
        # rw = UniformRandomWalk(StellarGraph(g))

        # if no graph is provided
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=None)

        # graph has to be a Stellargraph object
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=g)

        g = StellarGraph(g)

        """
        # only Uniform random walk is supported at the moment
        with pytest.raises(TypeError):
            UnsupervisedSampler(G=g, walker="any random walker")

        # if no walker is provided, default to Uniform Random Walk
        sampler = UnsupervisedSampler(G=g)
        assert isinstance(sampler.walker, UniformRandomWalk)

        """

        # walk must have length strictly greater than 1
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=g, length=1)

        # at least 1 walk from each root node
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=g, number_of_walks=0)

        # nodes nodes parameter should be an iterable of node IDs
        with pytest.raises(ValueError):
            UnsupervisedSampler(G=g, nodes=1)

        # if no root nodes are provided for sampling defaulting to using all nodes as root nodes
        sampler = UnsupervisedSampler(G=g, nodes=None)
        assert sampler.nodes == list(g.nodes())

        # if the seed value is provided check
        # that the random choices is reproducable
        sampler = UnsupervisedSampler(G=g, seed=1)
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

    def test_generator_parameter(self):

        g = create_test_graph_nx()
        g = StellarGraph(g)
        # rw = UniformRandomWalk(g)
        sampler = UnsupervisedSampler(G=g)

        # generator should be provided with a valid batch size. i.e. an integer >=1

        sample_gen = sampler.generator(batch_size=None)
        with pytest.raises(ValueError):
            next(sample_gen)

        sample_gen = sampler.generator(batch_size="x")
        with pytest.raises(TypeError):
            next(sample_gen)

        sample_gen = sampler.generator(batch_size=0)
        with pytest.raises(ValueError):
            next(sample_gen)

        sample_gen = sampler.generator(batch_size=3)
        with pytest.raises(ValueError):
            next(sample_gen)

    def test_generator_samples(self):

        n_feat = 4
        batch_size = 4

        G = example_graph_2(feature_size=n_feat)

        sampler = UnsupervisedSampler(G=G)

        sample_gen = sampler.generator(batch_size)

        samples = next(sample_gen)

        # return two lists: [(target,context)] pairs and [1/0] binary labels
        assert len(samples) == 2

        # each (target, context) pair has a matching label
        assert len(samples[0]) == len(samples[1])

        # batch-size number of samples are returned if batch_size is even
        assert len(samples[0]) == batch_size

    def test_generator_multiple_batches(self):

        n_feat = 4
        batch_size = 4
        number_of_batches = 3

        G = example_graph_2(feature_size=n_feat)

        sampler = UnsupervisedSampler(G=G)

        sample_gen = sampler.generator(batch_size)

        batches = []
        for batch in range(number_of_batches):
            batches.append(next(sample_gen))

        assert len(batches) == number_of_batches
