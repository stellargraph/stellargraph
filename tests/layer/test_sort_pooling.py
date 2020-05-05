# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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
from stellargraph.layer.sort_pooling import SortPooling


def test_sorting_padding():

    data = np.array([[3, 4, 0], [1, 2, 2], [5, 0, 1]], dtype=int).reshape((1, 3, 3))
    mask = np.array([[True, True, True]])
    data_sorted = np.array(
        [[1, 2, 2], [5, 0, 1], [3, 4, 0], [0, 0, 0]], dtype=int
    ).reshape((1, 4, 3))

    layer = SortPooling(k=4)

    data_out = layer(data, mask=mask)

    assert np.array_equal(data_out, data_sorted)

    # for mini-batch of size > 1
    data = np.array([[3, 1], [1, 2], [5, 0], [0, -4]], dtype=int).reshape((2, 2, 2))
    mask = np.array([[True, True], [True, True]])
    data_sorted = np.array(
        [[1, 2], [3, 1], [0, 0], [5, 0], [0, -4], [0, 0]], dtype=int
    ).reshape((2, 3, 2))

    layer = SortPooling(k=3)

    data_out = layer(data, mask=mask)

    assert np.array_equal(data_out, data_sorted)


def test_sorting_truncation():
    data = np.array([[3, 4, 0], [1, 2, 2], [5, 0, 1]], dtype=int).reshape((1, 3, 3))
    mask = np.array([[True, True, True]])

    data_sorted = np.array([[1, 2, 2], [5, 0, 1]], dtype=int).reshape((1, 2, 3))

    layer = SortPooling(k=2)

    data_out = layer(data, mask=mask)

    assert np.array_equal(data_out, data_sorted)

    # for mini-batch of size > 1
    data = np.array([[3, 1], [1, 2], [5, 0], [0, -4]], dtype=int).reshape((2, 2, 2))
    mask = np.array([[True, True], [True, True]])

    data_sorted = np.array([[1, 2], [5, 0]], dtype=int).reshape((2, 1, 2))

    layer = SortPooling(k=1)

    data_out = layer(data, mask=mask)

    assert np.array_equal(data_out, data_sorted)


def test_sorting_negative_values():

    data = np.array([[3, 4, 0], [1, 2, -1], [5, 0, 1]], dtype=int).reshape((1, 3, 3))
    mask = np.array([[True, True, True]])

    data_sorted = np.array([[5, 0, 1], [3, 4, 0], [1, 2, -1]], dtype=int).reshape(
        (1, 3, 3)
    )

    layer = SortPooling(k=3)

    data_out = layer(data, mask=mask)

    assert np.array_equal(data_out, data_sorted)


def test_mask():
    data = np.array([[3, 4, 0], [1, 2, -1], [5, 0, 1]], dtype=int).reshape((1, 3, 3))
    mask = np.array([[True, True, True]])

    data_sorted = np.array([[5, 0, 1], [3, 4, 0]], dtype=int).reshape((1, 2, 3))

    layer = SortPooling(k=2)

    data_out = layer(data, mask=mask)

    assert np.array_equal(data_out, data_sorted)

    mask = np.array([[True, True, False]])
    data_sorted = np.array(
        [[3, 4, 0], [1, 2, -1], [0, 0, 0], [0, 0, 0]], dtype=int
    ).reshape((1, 4, 3))

    layer = SortPooling(k=4)

    data_out = layer(data, mask=mask)

    assert np.array_equal(data_out, data_sorted)


def test_flatten_output():
    data = np.array([[3, 1], [1, 2], [5, 0], [0, -4]], dtype=int).reshape((2, 2, 2))
    mask = np.array([[True, True], [True, True]])

    data_sorted = np.array([[1, 2, 3, 1], [5, 0, 0, -4]], dtype=int).reshape((2, 4, 1))

    layer = SortPooling(k=2, flatten_output=True)

    data_out = layer(data, mask=mask)

    assert np.array_equal(data_out, data_sorted)


def test_invalid_k():
    with pytest.raises(TypeError, match="k: expected int, found str"):
        SortPooling(k="false")

    with pytest.raises(ValueError, match="k: expected integer >= 1, found 0"):
        SortPooling(k=0)
