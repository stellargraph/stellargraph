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

import numpy as np
import pandas as pd
import pytest

from stellargraph.core.columns import Columns


def test_constructor_validation():
    with pytest.raises(TypeError, match="expected dict"):
        Columns(1)

    with pytest.raises(TypeError, match=r"columns\['b'\]: expected ndarray"):
        Columns({"a": np.array([]), "b": 1})

    with pytest.raises(ValueError, match=r"columns\['b'\]: expected length 1.*found 2"):
        Columns({"a": np.array([1]), "b": np.array([3, 4])})


_DATA = Columns(
    {"a": np.array([1, 2]), "b": np.array([3.4, 5.6]), "c": np.array(["a", "b"])}
)


def test_column_names():
    assert Columns({}).column_names == set()
    assert _DATA.column_names == {"a", "b", "c"}


def test_column():
    assert np.array_equal(_DATA.column("a"), [1, 2])

    with pytest.raises(KeyError):
        _DATA.column("x")


def test_columns():
    a, b = _DATA.columns("a", "b")
    assert np.array_equal(a, [1, 2])
    assert np.array_equal(b, [3.4, 5.6])

    # it respects order
    b2, a2 = _DATA.columns("b", "a")
    assert np.array_equal(a2, [1, 2])
    assert np.array_equal(b2, [3.4, 5.6])

    with pytest.raises(KeyError):
        _DATA.columns("a", "b", "x")


def test_add_columns():
    new = _DATA.add_columns(
        {
            # overwrites the old c
            "c": np.array([12, 34]),
            # adds a new column
            "d": np.array([56, 78]),
        }
    )
    assert new.column_names == {"a", "b", "c", "d"}
    assert np.array_equal(new.column("c"), [12, 34])
    assert np.array_equal(new.column("d"), [56, 78])


def test_drop_columns():
    new = _DATA.drop_columns("c")
    assert new.column_names == {"a", "b"}

    # dropping a non-existant column does nothing
    _DATA.drop_columns("x")


def test_select_columns():
    new = _DATA.select_columns("a", "b")
    assert new.column_names == {"a", "b"}

    with pytest.raises(KeyError):
        _DATA.select_columns("a", "b", "x")


def test_select_rows():
    sliced = _DATA.select_rows(slice(0, 1))
    assert np.array_equal(sliced.column("a"), [1])
    assert np.array_equal(sliced.column("b"), [3.4])

    indices_list = [1, 1, 0, 1]
    indices_array = np.array(indices_list)
    for idx in [indices_list, indices_array]:
        selected = _DATA.select_rows(idx)
        assert np.array_equal(selected.column("a"), [2, 2, 1, 2])
        assert np.array_equal(selected.column("b"), [5.6, 5.6, 3.4, 5.6])

    booled = _DATA.select_rows([False, True])
    assert np.array_equal(booled.column("a"), [2])
    assert np.array_equal(booled.column("b"), [5.6])


def test_iter_rows():
    ab = list(_DATA.iter_rows("a", "b"))
    assert ab == [(1, 3.4), (2, 5.6)]

    ba = list(_DATA.iter_rows("b", "a"))
    assert ba == [(3.4, 1), (5.6, 2)]
