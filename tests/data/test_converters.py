# -*- coding: utf-8 -*-
#
# Copyright 2018-2019 Data61, CSIRO
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


from stellargraph.core.graph import *
from stellargraph.data.converter import (
    StellarAttributeConverter,
    CategoricalConverter,
    BinaryConverter,
    NumericConverter,
    OneHotCategoricalConverter,
    NodeAttributeSpecification,
)

import networkx as nx
import numpy as np
import pytest


def example_stellar_graph_1():
    G = nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (3, 2)]

    G.add_nodes_from([1, 2, 3, 4], label="")
    G.add_edges_from(elist, label="")

    # Add some node attributes
    G.node[1]["a1"] = 1
    G.node[3]["a1"] = 1
    G.node[1]["a2"] = 1
    G.node[4]["a2"] = 1
    return StellarGraph(G)


def test_converter_categorical():
    data = [1, 5, 5, 1, 6]
    conv = CategoricalConverter()
    converted_data = conv.fit_transform(data)

    assert isinstance(conv, StellarAttributeConverter)
    assert len(conv) == 1
    assert all(converted_data == [0, 1, 1, 0, 2])

    orig_data = conv.inverse_transform(converted_data)
    assert orig_data == data


def test_converter_categorical_mixed():
    data = [1, "a", "a", 1, "b", 2]
    conv = CategoricalConverter()
    converted_data = conv.fit_transform(data)

    assert isinstance(conv, StellarAttributeConverter)
    assert len(conv) == 1
    assert all(converted_data == [0, 2, 2, 0, 3, 1])

    orig_data = conv.inverse_transform(converted_data)
    assert orig_data == data


def test_converter_categorical_1hot():
    data = [1, 5, 5, 1, 6]
    conv = OneHotCategoricalConverter()
    converted_data = conv.fit_transform(data)

    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])

    assert isinstance(conv, StellarAttributeConverter)
    assert len(conv) == 3
    assert converted_data == pytest.approx(expected)

    orig_data = conv.inverse_transform(converted_data)
    assert orig_data == data

    conv = OneHotCategoricalConverter(without_first=True)
    converted_data = conv.fit_transform(data)

    expected = np.array([[0, 0], [1, 0], [1, 0], [0, 0], [0, 1]])

    assert len(conv) == 2
    assert converted_data == pytest.approx(expected)

    orig_data = conv.inverse_transform(converted_data)
    assert orig_data == data


def test_converter_binary():
    data = [1, "a", None, 0, "false"]
    conv = BinaryConverter()
    converted_data = conv.fit_transform(data)

    expected = np.array([1, 1, 0, 0, 1])

    assert isinstance(conv, StellarAttributeConverter)
    assert len(conv) == 1
    assert converted_data == pytest.approx(expected)

    # Note we can't recover the original for binary converters
    orig_data = conv.inverse_transform(converted_data)
    assert orig_data == [1, 1, None, None, 1]


def test_converter_categorical_1hot_binary():
    data = [1, 5, 5, 1, 5]
    conv = OneHotCategoricalConverter()
    converted_data = conv.fit_transform(data)

    expected = np.array([[1, 0], [0, 1], [0, 1], [1, 0], [0, 1]])

    assert isinstance(conv, StellarAttributeConverter)
    assert len(conv) == 2
    assert np.all(converted_data == expected)

    orig_data = conv.inverse_transform(converted_data)
    assert orig_data == data

    conv = OneHotCategoricalConverter(without_first=True)
    converted_data = conv.fit_transform(data)

    assert len(conv) == 1
    assert converted_data == pytest.approx([0, 1, 1, 0, 1])

    orig_data = conv.inverse_transform(converted_data)
    assert orig_data == data


def test_converter_numeric():
    data = np.array([2, 5, 5, 3, 5])
    conv = NumericConverter(normalize=False)
    converted_data = conv.fit_transform(data)

    assert isinstance(conv, StellarAttributeConverter)
    assert len(conv) == 1
    assert converted_data == pytest.approx(data)

    orig_data = conv.inverse_transform(converted_data)
    assert orig_data == pytest.approx(data)

    conv = NumericConverter(normalize="standard")
    converted_data = conv.fit_transform(data)
    expected = (np.array(data) - 4) / 1.26491
    assert isinstance(conv, StellarAttributeConverter)
    assert len(conv) == 1
    assert converted_data == pytest.approx(expected, rel=1e-3)

    orig_data = conv.inverse_transform(converted_data)
    assert orig_data == pytest.approx(data)


def test_attribute_spec():
    nfs = NodeAttributeSpecification()
    nfs.add_attribute("", "a1", NumericConverter, default_value=0, normalize=False)
    nfs.add_attribute("", "a2", NumericConverter, default_value=0, normalize=False)

    data = [{"a1": 1, "a2": 1}, {"a2": 1}, {"a1": 1}, {}]

    attr_list = nfs.get_attributes("")
    assert attr_list == ["a1", "a2"]

    converted_data = nfs.fit_transform("", data)
    expected = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
    assert converted_data == pytest.approx(expected)

    orig_data = nfs.inverse_transform("", converted_data)
    assert len(orig_data) == len(data)


def test_attribute_spec_add_list():
    nfs = NodeAttributeSpecification()
    nfs.add_attribute_list(
        "", ["a1", "a2"], NumericConverter, default_value=0, normalize=False
    )

    data = [{"a1": 1, "a2": 1}, {"a2": 1}, {"a1": 1}, {}]

    attr_list = nfs.get_attributes("")
    assert attr_list == ["a1", "a2"]

    converted_data = nfs.fit_transform("", data)
    expected = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])

    assert converted_data == pytest.approx(expected)


def test_attribute_spec_add_all():
    G = example_stellar_graph_1()

    nfs = NodeAttributeSpecification()
    nfs.add_all_attributes(G, "", NumericConverter, default_value=0, normalize=False)

    data = [{"a1": 1, "a2": 1}, {"a2": 1}, {"a1": 1}, {}]

    attr_list = nfs.get_attributes("")
    assert attr_list == ["a1", "a2"]

    converted_data = nfs.fit_transform("", data)
    expected = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])

    assert converted_data == pytest.approx(expected)


def test_attribute_spec_normalize_error():
    nfs = NodeAttributeSpecification()
    nfs.add_attribute("", "a1", NumericConverter, default_value=0)
    nfs.add_attribute("", "a2", NumericConverter, default_value=0)

    data = [{"a1": 1, "a2": 1}, {"a2": 1}, {"a1": 1}, {}]

    # We expect an error here as the normalization works before values have been
    # imputed with the default value, therefore the std dev will be zero.
    with pytest.raises(ValueError):
        nfs.fit_transform("", data)

    nfs = NodeAttributeSpecification()
    nfs.add_attribute("", "a", NumericConverter, default_value=0)

    data = [{"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}]
    with pytest.raises(ValueError):
        nfs.fit_transform("", data)


def test_attribute_spec_binary_conv():
    nfs = NodeAttributeSpecification()
    nfs.add_attribute("", "a1", BinaryConverter)
    nfs.add_attribute("", "a2", BinaryConverter)

    data = [{"a1": 1, "a2": 1}, {"a2": 1}, {"a1": 1}, {}]
    converted_data = nfs.fit_transform("", data)
    expected = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
    assert converted_data == pytest.approx(expected)

    orig_data = nfs.inverse_transform("", converted_data)
    assert len(orig_data) == len(data)


def test_attribute_spec_mixed():
    nfs = NodeAttributeSpecification()
    nfs.add_attribute("", "a1", OneHotCategoricalConverter)
    nfs.add_attribute("", "a2", NumericConverter, default_value="mean")

    data = [{"a1": 1, "a2": 0}, {"a1": "a", "a2": 1}, {"a1": 1}, {"a1": "a"}]

    converted_data = nfs.fit_transform("", data)
    expected = np.array([[1, 0, -1], [0, 1, 1], [1, 0, 0], [0, 1, 0]])

    assert converted_data == pytest.approx(expected)

    orig_data = nfs.inverse_transform("", converted_data)
    assert len(orig_data) == len(data)
