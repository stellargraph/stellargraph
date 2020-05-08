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

import pytest

import numpy as np
from stellargraph.utils.hyperbolic import *


@pytest.fixture
def seeded():
    seed = np.random.randint(2 ** 32)
    # log for reproducibility
    print("seed:", seed)
    np.random.seed(seed)


def _generate(num_vectors, norm_range=(0, 1)):
    c = np.random.random() * 10
    n = np.random.choice([1, 2, 5, 10, 30, 100])
    extra_axes = np.random.choice([0, 1, 3])

    small, big = norm_range
    norms = (small + np.random.random(size=num_vectors) * (big - small)) / c ** 0.5

    raws = 2 * np.random.random(size=(num_vectors, n)) - 1
    scale = norms / np.linalg.norm(raws, axis=1)
    vs = raws * scale[:, None]
    return c, vs.astype(np.float32)[(None,) * extra_axes]


def test_poincare_ball_exp_map0_specialisation(seeded):
    for _ in range(100):
        c, vs = _generate(17)

        specialised = poincare_ball_exp_map0(c, vs)
        assert specialised.shape == vs.shape

        actual = poincare_ball_exp_map(np.zeros_like(vs), c, vs)
        np.testing.assert_allclose(specialised.numpy(), actual.numpy())


def test_poincare_ball_distance_vs_euclidean(seeded):
    for _ in range(100):
        # d_c(0, x) is approximtely 2||x||_2 for sufficiently small x
        c, vs = _generate(17, norm_range=(0, 0.01))
        zeros = np.zeros_like(vs)
        hyperbolic = poincare_ball_distance(c, zeros, vs)
        assert hyperbolic.shape == vs.shape[:-1]

        euclidean = np.linalg.norm(vs, axis=-1)
        np.testing.assert_allclose(hyperbolic, 2 * euclidean, rtol=1e-3, atol=1e-15)

        # d_c(0, x) is much larger than 2||x||_2 for sufficiently large x
        c, vs = _generate(17, norm_range=(0.99, 1))
        zeros = np.zeros_like(vs)
        hyperbolic = poincare_ball_distance(c, zeros, vs)
        assert hyperbolic.shape == vs.shape[:-1]

        euclidean = np.linalg.norm(vs, axis=-1)
        np.testing.assert_array_less(4 * euclidean, hyperbolic)
