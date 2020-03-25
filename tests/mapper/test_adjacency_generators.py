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

from stellargraph.core.utils import normalize_adj
from stellargraph.mapper.adjacency_generators import AdjacencyPowerGenerator
from ..test_utils.graphs import barbell
import tensorflow as tf


import numpy as np
import pytest


def test_init(barbell):

    generator = AdjacencyPowerGenerator(barbell, num_powers=5)
    num_nodes = len(barbell.nodes())

    assert generator.num_powers == 5
    assert generator.Aadj_T.shape == (num_nodes, num_nodes)
    assert generator.transition_matrix_T.shape == (num_nodes, num_nodes)


def test_bad_init(barbell):

    with pytest.raises(TypeError):
        generator = AdjacencyPowerGenerator(None, num_powers=5)

    with pytest.raises(ValueError, match="num_powers: expected.*found -1"):
        generator = AdjacencyPowerGenerator(barbell, num_powers=-1)

    with pytest.raises(TypeError, match="num_powers: expected.*found float"):
        generator = AdjacencyPowerGenerator(barbell, num_powers=1.0)


def test_flow(barbell):

    generator = AdjacencyPowerGenerator(barbell, num_powers=5)
    dataset = generator.flow(batch_size=1)
    assert tf.data.experimental.cardinality(dataset).numpy() == -1


def test_bad_flow(barbell):

    generator = AdjacencyPowerGenerator(barbell, num_powers=5)

    with pytest.raises(ValueError, match="batch_size: expected.*found 0"):
        generator.flow(0, num_parallel_calls=1)

    with pytest.raises(TypeError, match="batch_size: expected.*found float"):
        generator.flow(1.0, num_parallel_calls=1)

    with pytest.raises(ValueError, match="num_parallel_calls: expected.*found 0"):
        generator.flow(1, num_parallel_calls=0)

    with pytest.raises(TypeError, match="num_parallel_calls: expected.*found float"):
        generator.flow(1, num_parallel_calls=1.01)


@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_flow_batch_size(barbell, batch_size):

    num_powers = 5
    generator = AdjacencyPowerGenerator(barbell, num_powers=num_powers)
    for x, y in generator.flow(batch_size=batch_size).take(1):

        assert x[0].shape == (batch_size,)
        assert y.shape == (batch_size, 1, len(barbell.nodes()))
        assert x[1].shape == (batch_size, num_powers, len(barbell.nodes()))


@pytest.mark.parametrize("num_powers", [2, 4, 8])
def test_flow_batch_size(barbell, num_powers):

    batch_size = 2
    generator = AdjacencyPowerGenerator(barbell, num_powers=num_powers)
    for x, y in generator.flow(batch_size=batch_size).take(1):

        assert x[0].shape == (batch_size,)
        assert y.shape == (batch_size, 1, len(barbell.nodes()))
        assert x[1].shape == (batch_size, num_powers, len(barbell.nodes()))


@pytest.mark.parametrize("num_powers", [2, 4, 8])
def test_partial_powers(barbell, num_powers):

    Aadj = normalize_adj(barbell.to_adjacency_matrix(), symmetric=False).todense()
    actual_powers = [Aadj]
    for _ in range(num_powers - 1):
        actual_powers.append(actual_powers[-1].dot(Aadj))

    generator = AdjacencyPowerGenerator(barbell, num_powers=num_powers)
    dataset = generator.flow(batch_size=1)
    for i, (x, y) in enumerate(dataset.take(barbell.number_of_nodes())):

        partial_powers = x[1].numpy()
        for j in range(num_powers):
            print(i, j)
            assert np.allclose(partial_powers[0, j, :], actual_powers[j][i, :])
