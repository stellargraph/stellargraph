# -*- coding: utf-8 -*-
#
# Copyright 2019 Data61, CSIRO
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
import pandas as pd

from stellargraph.data.node_data import NodeData


class TestNodeData:
    def test_empty_data(self):
        nd = NodeData()
        assert nd.num_nodes() == 0
        assert nd.node_type_set() == set()
        assert nd.default_node_type() == NodeData.DEFAULT_NODE_TYPE

    def test_id_index(self):
        node_ids = range(5, 15)
        df = pd.DataFrame({"x": node_ids}).set_index("x")
        nd = NodeData(df, node_id=NodeData.PANDAS_INDEX)
        assert nd.num_nodes() == len(node_ids)
        assert list(nd.node_ids()) == list(node_ids)
        assert nd.node_type_set() == {NodeData.DEFAULT_NODE_TYPE}
        assert list(nd.node_types()) == [NodeData.DEFAULT_NODE_TYPE] * len(node_ids)
        for node_id in range(5, 15):
            assert nd.has_node(node_id)
        assert not nd.has_node(42)
        assert not nd.has_node(None)

    def test_id_column(self):
        node_ids = range(5, 15)
        df = pd.DataFrame({"x": node_ids, "y": range(len(node_ids))})
        nd = NodeData(df, node_id="x")
        assert nd.num_nodes() == len(node_ids)
        assert list(nd.node_ids()) == list(node_ids)
        assert nd.node_type_set() == {NodeData.DEFAULT_NODE_TYPE}
        assert list(nd.node_types()) == [NodeData.DEFAULT_NODE_TYPE] * len(node_ids)

    def test_id_wrong_column(self):
        with pytest.raises(ValueError):
            df = pd.DataFrame({"x": range(5)})
            nd = NodeData(df, node_id="y")

    def test_id_type_columns(self):
        node_ids = ["x", "y", "z", 1, 2, 3]
        node_types = ["a", "b", "c", -1, -2, -3]
        df = pd.DataFrame({"x": node_ids, "y": node_types})
        nd = NodeData(df, node_id="x", node_type="y")
        assert nd.num_nodes() == len(node_ids)
        assert list(nd.node_ids()) == list(node_ids)
        assert nd.node_type_set() == set(node_types)
        assert list(nd.node_types()) == node_types

    def test_type_column(self):
        node_types = ["a", "b", "c", "a", "b", "c", "b", "a"]
        df = pd.DataFrame({"x": node_types})
        nd = NodeData(df, node_type="x")
        assert nd.num_nodes() == len(node_types)
        assert list(nd.node_ids()) == list(range(len(node_types)))
        assert nd.node_type_set() == set(node_types)
        assert list(nd.node_types()) == node_types

    def test_id_type_lookup(self):
        node_ids = [1, 5, 7, 3, 9, 11, 13, 0]
        node_types = ["a", "b", "c", "a", "b", "c", "b", "a"]
        df = pd.DataFrame({"x": node_ids, "y": node_types})
        nd = NodeData(df, node_id="x", node_type="y")
        assert nd.num_nodes() == len(node_ids)
        assert list(nd.node_ids()) == node_ids
        assert list(nd.node_types()) == node_types
        for i, node_id in enumerate(node_ids):
            node_idx = nd.node_index(node_id)
            assert node_idx == i
            node_type = nd.node_type(node_id)
            assert node_type == node_types[i]
            assert nd.node_id(node_idx) == node_id
        assert nd.node_index("blah") < 0
        assert nd.node_type("blah") is None
        assert nd.node_id(-1) is None
        assert nd.node_id(len(node_ids)) is None
        assert nd.node_id(len(node_ids) + 11) is None

    def test_features(self):
        num_rows = 20
        num_features = 15
        x = np.random.rand(num_rows, num_features + 2)
        feature_names = ["f{}".format(i) for i in range(num_features)]
        df = pd.DataFrame(x, columns=["x"] + feature_names + ["y"])
        nd = NodeData(df, node_features=feature_names)
        for i in range(num_rows):
            v = nd.node_features([i])
            assert v.shape == (1, num_features)
            assert list(v[0, :]) == list(x[i, 1:-1])

    def test_type_dict(self):
        type_data = {
            "a": pd.DataFrame([1, 2, 3], columns=["id"]),
            "b": pd.DataFrame([4, 5], columns=["id"]),
            "c": pd.DataFrame([6], columns=["id"]),
            "d": pd.DataFrame([], columns=["id"]),
        }
        nd = NodeData(type_data, node_id="id")
        assert nd.num_nodes() == 6
        assert nd.node_type_set() == {"a", "b", "c"}
        assert set(nd.node_ids()) == set(range(1, 7))
        assert len(list(nd.node_ids())) == 6
        assert set(nd.node_types()) == {"a", "b", "c"}
        for node_id in range(1, 7):
            assert nd.has_node(node_id)
        assert not nd.has_node(42)
        assert not nd.has_node(None)

    def test_type_dict_missing_ids(self):
        type_data = {
            "a": NodeData(pd.DataFrame([1, 2, 3], columns=["id"]), node_id="id"),
            "b": NodeData(pd.DataFrame([4, 5], columns=["id"])),
        }
        with pytest.raises(ValueError):
            NodeData(type_data, node_id="id")
