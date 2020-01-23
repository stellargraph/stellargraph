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

import pandas as pd

from stellargraph.core.convert import ColumnarConverter, convert_nodes, convert_edges

_EMPTY_DF = pd.DataFrame([], index=[1, 2])


def test_convert_type_default():
    converter = ColumnarConverter("x", "foo", {}, {}, False)
    shared, features = converter.convert(_EMPTY_DF)
    assert "foo" in shared
    assert "foo" in features


def test_convert_selected_columns():
    df = _EMPTY_DF.assign(before="abc")

    converter = ColumnarConverter("x", "foo", {}, {"before": "after"}, False)
    shared, features = converter.convert({"x": df, "y": df})

    assert "x" in shared
    assert "y" in shared

    for df in shared.values():
        assert "before" not in df
        assert all(df["after"] == "abc")


def test_convert_column_default():
    converter = ColumnarConverter("x", "foo", {"before": 123}, {}, False)
    shared, features = converter.convert({"x": _EMPTY_DF, "y": _EMPTY_DF})

    assert "x" in shared
    assert "y" in shared

    for df in shared.values():
        assert all(df["before"] == 123)


def test_convert_column_default_selected():
    converter = ColumnarConverter(
        "x", "foo", {"before": 123}, {"before": "after"}, False
    )
    shared, features = converter.convert({"x": _EMPTY_DF, "y": _EMPTY_DF})

    assert "x" in shared
    assert "y" in shared

    for df in shared.values():
        assert "before" not in df
        assert all(df["after"] == 123)
