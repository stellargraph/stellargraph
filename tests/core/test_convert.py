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

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from stellargraph.core.convert import (
    ColumnarConverter,
    convert_nodes,
    convert_edges,
    from_networkx,
)

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


def from_networkx_for_testing(g, node_features=None, dtype="float32"):
    return from_networkx(
        g,
        node_type_attr="n",
        edge_type_attr="e",
        node_type_default="a",
        edge_type_default="x",
        edge_weight_attr="w",
        node_features=node_features,
        dtype=dtype,
    )


def test_from_networkx_empty():
    nodes, edges = from_networkx_for_testing(nx.DiGraph())
    assert nodes == {}
    assert edges == {}


def assert_dataframe_dict_equal(new, expected):
    assert sorted(new.keys()) == sorted(expected.keys())

    for k, expected_value in expected.items():
        pd.testing.assert_frame_equal(new[k], expected_value)


# default value for edge weights
W = np.float32(1)


def test_from_networkx_graph_only():
    raw_edges = [(0, 1), (0, 2), (0, 2), (1, 2), (1, 2)]
    expected_nodes = {"a": pd.DataFrame(columns=range(0), index=[0, 1, 2])}
    expected_edges = {
        "x": pd.DataFrame(raw_edges, columns=["source", "target"]).assign(w=W)
    }

    g = nx.MultiDiGraph()
    g.add_edges_from(raw_edges)

    nodes, edges = from_networkx_for_testing(g)
    assert_dataframe_dict_equal(nodes, expected_nodes)
    assert_dataframe_dict_equal(edges, expected_edges)


def test_from_networkx_ignore_unknown_attrs():
    raw_nodes = [(0, {"foo": 123})]
    raw_edges = [(0, 0, {"bar": 456})]
    expected_nodes = {"a": pd.DataFrame(columns=range(0), index=[0])}
    expected_edges = {
        "x": pd.DataFrame([(0, 0)], columns=["source", "target"], index=[0]).assign(w=W)
    }

    g = nx.MultiDiGraph()
    g.add_nodes_from(raw_nodes)
    g.add_edges_from(raw_edges)
    nodes, edges = from_networkx_for_testing(g)
    assert_dataframe_dict_equal(nodes, expected_nodes)
    assert_dataframe_dict_equal(edges, expected_edges)


def test_from_networkx_heterogeneous_partial():
    # check that specifying node and edge types works, even when interleaved with unspecified types
    a_nodes = [0, (1, {"n": "a"})]
    b_nodes = [(2, {"n": "b"})]
    expected_nodes = {
        "a": pd.DataFrame(columns=range(0), index=[0, 1]),
        "b": pd.DataFrame(columns=range(0), index=[2]),
    }

    x_edges = [(0, 2, {"e": "x"}), (0, 2), (1, 2), (1, 2)]
    xs = len(x_edges)
    y_edges = [(0, 1, {"e": "y"})]
    ys = len(y_edges)
    expected_edges = {
        "x": pd.DataFrame(
            [t[:2] for t in x_edges], columns=["source", "target"], index=[0, 1, 3, 4],
        ).assign(w=W),
        "y": pd.DataFrame(
            [t[:2] for t in y_edges], columns=["source", "target"], index=[2],
        ).assign(w=W),
    }

    g = nx.MultiDiGraph()
    g.add_nodes_from(a_nodes)
    g.add_nodes_from(b_nodes)
    g.add_edges_from(x_edges)
    g.add_edges_from(y_edges)
    nodes, edges = from_networkx_for_testing(g)

    assert_dataframe_dict_equal(nodes, expected_nodes)
    assert_dataframe_dict_equal(edges, expected_edges)


def test_from_networkx_weights():
    expected_nodes = {
        "a": pd.DataFrame(columns=range(0), index=[0, 2, 1]),
    }

    x_edges = [(0, 2, {"w": 2.0}), (0, 2), (1, 2), (1, 2)]
    xs = len(x_edges)
    y_edges = [(0, 1, {"w": 3.0, "e": "y"})]
    ys = len(y_edges)

    def df_edge(edge_tuple):
        src, tgt = edge_tuple[:2]
        try:
            attrs = edge_tuple[2]
        except IndexError:
            attrs = {}
        weight = attrs.get("w", 1)
        return src, tgt, weight

    expected_edges = {
        "x": pd.DataFrame(
            [df_edge(t) for t in x_edges],
            columns=["source", "target", "w"],
            index=[0, 1, 3, 4],
        ),
        "y": pd.DataFrame(
            [df_edge(t) for t in y_edges], columns=["source", "target", "w"], index=[2]
        ),
    }

    g = nx.MultiDiGraph()
    g.add_edges_from(x_edges + y_edges)

    nodes, edges = from_networkx_for_testing(g)
    assert_dataframe_dict_equal(nodes, expected_nodes)
    assert_dataframe_dict_equal(edges, expected_edges)


@pytest.mark.parametrize(
    "feature_type",
    [
        "nodes",
        "dataframe no types",
        "dataframe types",
        "iterable no types",
        "iterable types",
    ],
)
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_from_networkx_homogeneous_features(feature_type, dtype):
    features = [[0, 1, 2], [3, 4, 5]]

    def node(i, **kwargs):
        feat = {"f": features[i]} if feature_type == "nodes" else {}
        return i, feat

    a_nodes = [node(0), node(1)]
    if feature_type == "nodes":
        node_features = "f"
    elif feature_type == "dataframe no types":
        node_features = pd.DataFrame(features)
    elif feature_type == "dataframe types":
        node_features = {"a": pd.DataFrame(features)}
    elif feature_type == "iterable no types":
        node_features = enumerate(features)
    elif feature_type == "iterable types":
        node_features = {"a": enumerate(features)}

    expected_nodes = {"a": pd.DataFrame(features, dtype=dtype)}

    g = nx.MultiDiGraph()
    g.add_nodes_from(a_nodes)
    nodes, edges = from_networkx_for_testing(g, node_features, dtype)
    assert_dataframe_dict_equal(nodes, expected_nodes)
    assert edges == {}


@pytest.mark.parametrize(
    "feature_type", ["nodes", "dataframe", "iterable no types", "iterable types"]
)
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_from_networkx_heterogeneous_features(feature_type, dtype):
    a_features = [[0, 1, 2], [3, 4, 5]]
    b_features = [[6, 7, 8, 9]]
    # node type c has no features

    def node(i, feats, **kwargs):
        feat = {"f": feats} if feature_type == "nodes" else {}
        return (i, {**kwargs, **feat})

    # make sure the default node type is applied correctly
    a_nodes = [node(0, a_features[0]), node(2, a_features[1], n="a")]
    b_nodes = [node(1, b_features[0], n="b")]
    c_nodes = [(3, {"n": "c"})]

    if feature_type == "nodes":
        node_features = "f"
    elif feature_type == "dataframe":
        node_features = {
            "a": pd.DataFrame(a_features, index=[0, 2]),
            "b": pd.DataFrame(b_features, index=[1]),
            # c is implied
        }
    elif feature_type == "iterable no types":
        node_features = zip([0, 2, 1, 3], a_features + b_features + [[]])
    elif feature_type == "iterable types":
        node_features = {
            "a": zip([0, 2], a_features),
            "b": zip([1], b_features),
            "c": [(3, [])],
        }

    expected_nodes = {
        "a": pd.DataFrame(a_features, index=[0, 2], dtype=dtype),
        "b": pd.DataFrame(b_features, index=[1], dtype=dtype),
        "c": pd.DataFrame(columns=range(0), index=[3], dtype=dtype),
    }

    g = nx.MultiDiGraph()
    g.add_nodes_from(a_nodes + b_nodes + c_nodes)
    nodes, edges = from_networkx_for_testing(g, node_features, dtype)
    assert_dataframe_dict_equal(nodes, expected_nodes)
    assert edges == {}


def test_from_networkx_default_features():
    abc = [1, 2, 3, 4]
    expected_nodes = {
        "a": pd.DataFrame([abc, [0] * len(abc)], index=["abc", "def"], dtype="float32")
    }
    g = nx.MultiDiGraph()
    g.add_node("abc", f=abc)
    g.add_node("def")

    with pytest.warns(UserWarning, match="type 'a'.*4-dimensional zero vector: 'def'$"):
        nodes, edges = from_networkx_for_testing(g, "f")

    assert_dataframe_dict_equal(nodes, expected_nodes)
    assert edges == {}


def test_from_networkx_errors():
    het = nx.MultiDiGraph()
    het.add_nodes_from([(0, {"n": "a"}), (1, {"n": "b"})])

    with pytest.raises(TypeError, match="more than one node type"):
        from_networkx_for_testing(het, node_features=pd.DataFrame())

    with pytest.raises(ValueError, match=r"\['a'\]:.*: missing from data \(0\)$"):
        from_networkx_for_testing(
            het, node_features={"a": pd.DataFrame(), "b": pd.DataFrame()}
        )

    with pytest.raises(ValueError, match=r"\['a'\]:.*: extra in data \(1\)$"):
        from_networkx_for_testing(
            het, node_features={"a": pd.DataFrame(index=[0, 1]), "b": pd.DataFrame()}
        )

    with pytest.raises(
        ValueError,
        match=r"\['a'\]:.*: missing from data \(0\) and extra in data \(1\)$",
    ):
        from_networkx_for_testing(
            het, node_features={"a": pd.DataFrame(index=[1]), "b": pd.DataFrame()}
        )

    with pytest.raises(TypeError, match=r"\['a'\]: .*, found NoneType"):
        from_networkx_for_testing(het, node_features={"a": None})

    attrs = nx.DiGraph()
    attrs.add_nodes_from([(0, {"f": [1]}), (1, {"f": [2, 3]})])
    with pytest.raises(
        ValueError, match="of type 'a' to have feature dimension 1, found dimension 2"
    ):
        from_networkx_for_testing(attrs, node_features="f")
