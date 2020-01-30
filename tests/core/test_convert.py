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

from stellargraph.core.convert import ColumnarConverter, convert_nodes, convert_edges

_EMPTY_DF = pd.DataFrame([], index=[1, 2])


def test_columnar_convert_type_default():
    converter = ColumnarConverter("some_name", "foo", {}, {}, False)
    shared, features = converter.convert(_EMPTY_DF)
    assert "foo" in shared
    assert "foo" in features


def test_columnar_convert_selected_columns():
    df = _EMPTY_DF.assign(before="abc", same=10)

    converter = ColumnarConverter(
        "some_name", "foo", {}, {"before": "after", "same": "same"}, False
    )
    shared, features = converter.convert({"x": df, "y": df})

    assert "x" in shared
    assert "y" in shared

    for df in shared.values():
        assert "before" not in df
        assert all(df["after"] == "abc")
        assert all(df["same"] == 10)


def test_columnar_convert_selected_columns_missing():
    converter = ColumnarConverter(
        "some_name", "foo", {}, {"before": "after", "same": "same"}, False
    )

    with pytest.raises(
        ValueError, match=r"some_name\['x'\]: expected 'before', 'same' columns, found:"
    ):
        converter.convert({"x": _EMPTY_DF})


def test_columnar_convert_column_default():
    converter = ColumnarConverter("some_name", "foo", {"before": 123}, {}, False)
    shared, features = converter.convert({"x": _EMPTY_DF, "y": _EMPTY_DF})

    assert "x" in shared
    assert "y" in shared

    for df in shared.values():
        assert all(df["before"] == 123)


def test_columnar_convert_column_default_selected_columns():
    # the defaulting happens before the renaming
    converter = ColumnarConverter(
        "x", "foo", {"before": 123}, {"before": "after"}, False
    )
    shared, features = converter.convert({"x": _EMPTY_DF, "y": _EMPTY_DF})

    assert "x" in shared
    assert "y" in shared

    for df in shared.values():
        assert "before" not in df
        assert all(df["after"] == 123)


def test_columnar_convert_features():
    converter = ColumnarConverter("some_name", "foo", {}, {"x": "x"}, True)
    df = _EMPTY_DF.assign(a=[1, 2], b=[100, 200], x=123)
    shared, features = converter.convert(df)

    assert all(shared["foo"]["x"] == 123)
    assert np.array_equal(features["foo"], [[1, 100], [2, 200]])


def test_columnar_convert_disallow_features():
    converter = ColumnarConverter("some_name", "foo", {}, {}, False)
    df = _EMPTY_DF.assign(a=1)
    with pytest.raises(ValueError, match="expected zero feature columns, found 'a'"):
        shared, features = converter.convert(df)


def test_columnar_convert_invalid_input():
    converter = ColumnarConverter("some_name", "foo", {}, {}, False)

    with pytest.raises(
        TypeError, match="some_name: expected dict, found <class 'int'>"
    ):
        converter.convert(1)

    with pytest.raises(
        TypeError,
        match=r"some_name\['x'\]: expected pandas DataFrame, found <class 'int'>",
    ):
        converter.convert({"x": 1})
