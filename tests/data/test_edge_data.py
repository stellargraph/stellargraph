# -*- coding: utf-8 -*-
#
# Copyright 2019 Data61, CSIRO
#
# Licensed ueder the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed ueder the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions aed
# limitations ueder the License.

import pytest

import numpy as np
import pandas as pd

from stellargraph.data.edge_data import EdgeData


class TestEdgeData:
    def create_data(self):
        src_ids = ["r", "r", "r", "r.2", "r.3", "r.3"]
        dst_ids = ["r.1", "r.2", "r.3", "r.2.1", "r.3.1", "r.3.2"]
        edge_ids = ["a", "b", "c", 1, 2, 3]
        edge_types = [1, 2, 3, "a", "b", "c"]
        return src_ids, dst_ids, edge_ids, edge_types

    def test_empty(self):
        ed = EdgeData()
        assert ed.num_edges() == 0
        assert ed.edge_type_set() == set()
        assert ed.default_edge_type() == EdgeData.DEFAULT_EDGE_TYPE

    def test_one_edge(self):
        df = pd.DataFrame({"s": ["from"], "d": ["to"]})
        ed = EdgeData(df, source_id="s", target_id="d")
        assert ed.num_edges() == 1
        assert list(ed.edge_ids()) == list(range(1))
        assert ed.edge_type_set() == {EdgeData.DEFAULT_EDGE_TYPE}
        assert list(ed.edge_types()) == [EdgeData.DEFAULT_EDGE_TYPE]
        for i, (src_id, dst_id) in enumerate(ed.edges()):
            assert i < 1
            assert src_id == "from"
            assert dst_id == "to"

    def test_edges(self):
        src_ids, dst_ids, _, _ = self.create_data()
        df = pd.DataFrame({"s": src_ids, "d": dst_ids})
        ed = EdgeData(df, source_id="s", target_id="d")
        assert ed.num_edges() == len(src_ids)
        assert list(ed.edge_ids()) == list(range(len(src_ids)))
        assert ed.edge_type_set() == {EdgeData.DEFAULT_EDGE_TYPE}
        assert list(ed.edge_types()) == [EdgeData.DEFAULT_EDGE_TYPE] * len(src_ids)
        for i, (src_id, dst_id) in enumerate(ed.edges()):
            assert src_id == src_ids[i]
            assert dst_id == dst_ids[i]

    def test_id_index(self):
        src_ids, dst_ids, edge_ids, _ = self.create_data()
        df = pd.DataFrame({"i": edge_ids, "s": src_ids, "d": dst_ids}).set_index("i")
        ed = EdgeData(df, source_id="s", target_id="d", edge_id=EdgeData.PANDAS_INDEX)
        assert ed.num_edges() == len(edge_ids)
        assert list(ed.edge_ids()) == list(edge_ids)
        assert ed.edge_type_set() == {EdgeData.DEFAULT_EDGE_TYPE}
        assert len(list(ed.edge_types())) == len(edge_ids)
        assert list(ed.edge_types()) == [EdgeData.DEFAULT_EDGE_TYPE] * len(edge_ids)
        for edge_id in edge_ids:
            assert ed.has_edge(edge_id)
        assert not ed.has_edge(None)

    def test_id_column(self):
        src_ids, dst_ids, edge_ids, _ = self.create_data()
        df = pd.DataFrame({"i": edge_ids, "s": src_ids, "d": dst_ids})
        ed = EdgeData(df, source_id="s", target_id="d", edge_id="i")
        assert ed.num_edges() == len(edge_ids)
        assert list(ed.edge_ids()) == list(edge_ids)
        assert ed.edge_type_set() == {EdgeData.DEFAULT_EDGE_TYPE}
        assert list(ed.edge_types()) == [EdgeData.DEFAULT_EDGE_TYPE] * len(edge_ids)
        for edge_id in edge_ids:
            assert ed.has_edge(edge_id)
        assert not ed.has_edge(None)

    def test_id_wrong_column(self):
        with pytest.raises(ValueError):
            df = pd.DataFrame({"x": range(5)})
            ed = EdgeData(df, edge_id="y")

    def test_id_type_columns(self):
        src_ids, dst_ids, edge_ids, edge_types = self.create_data()
        df = pd.DataFrame({"i": edge_ids, "s": src_ids, "d": dst_ids, "t": edge_types})
        ed = EdgeData(df, source_id="s", target_id="d", edge_id="i", edge_type="t")
        assert ed.num_edges() == len(edge_ids)
        assert list(ed.edge_ids()) == list(edge_ids)
        assert ed.edge_type_set() == set(edge_types)
        assert list(ed.edge_types()) == edge_types

    def test_type_column(self):
        src_ids, dst_ids, _, edge_types = self.create_data()
        df = pd.DataFrame({"s": src_ids, "d": dst_ids, "t": edge_types})
        ed = EdgeData(df, source_id="s", target_id="d", edge_type="t")
        assert ed.num_edges() == len(edge_types)
        assert list(ed.edge_ids()) == list(range(len(edge_types)))
        assert ed.edge_type_set() == set(edge_types)
        assert list(ed.edge_types()) == edge_types

    def test_id_type_lookup(self):
        src_ids, dst_ids, edge_ids, edge_types = self.create_data()
        df = pd.DataFrame({"i": edge_ids, "s": src_ids, "d": dst_ids, "t": edge_types})
        ed = EdgeData(df, source_id="s", target_id="d", edge_id="i", edge_type="t")
        assert ed.num_edges() == len(edge_ids)
        assert list(ed.edge_ids()) == edge_ids
        assert list(ed.edge_types()) == edge_types
        for i, edge_id in enumerate(edge_ids):
            edge_idx = ed.edge_index(edge_id)
            assert edge_idx == i
            edge_type = ed.edge_type(edge_id)
            assert edge_type == edge_types[i]
            # assert ed.edge_id(edge_idx) == edge_id
        assert ed.edge_index("blah") < 0
        assert ed.edge_type("blah") is None
        # assert ed.edge_id(-1) is None
        # assert ed.edge_id(len(edge_ids)) is None
        # assert ed.edge_id(len(edge_ids) + 11) is None

    def test_neighbours(self):
        src_ids, dst_ids, edge_ids, edge_types = self.create_data()
        df = pd.DataFrame({"i": edge_ids, "s": src_ids, "d": dst_ids, "t": edge_types})
        ed = EdgeData(df, source_id="s", target_id="d", edge_id="i", edge_type="t")
        assert set(ed.neighbour_nodes("r.2")) == {"r", "r.2.1"}
        assert set(ed.out_nodes("r")) == {"r.1", "r.2", "r.3"}
        assert set(ed.in_nodes("r.3.2")) == {"r.3"}

    def test_type_dict(self):
        type_data = {
            "a": pd.DataFrame({"i": [1, 2, 3], "s": [1, 2, 3], "d": [3, 2, 1]}),
            "b": pd.DataFrame({"i": [4, 5, 6], "s": [11, 12, 13], "d": [13, 12, 11]}),
            "c": pd.DataFrame([], columns=["i", "s", "d"]),
        }
        ne = EdgeData(type_data, edge_id="i", source_id="s", target_id="d")
        assert ne.num_edges() == 6
        assert ne.edge_type_set() == {"a", "b"}
        assert set(ne.edge_ids()) == set(range(1, 7))
        assert len(list(ne.edge_ids())) == 6
        assert set(ne.edge_types()) == {"a", "b"}
        for edge_id in range(1, 7):
            assert ne.has_edge(edge_id)
        assert not ne.has_edge(42)
        assert not ne.has_edge(None)

    def test_type_dict_missing_ids(self):
        type_data = {
            "a": pd.DataFrame({"i": [1, 2, 3], "s": [1, 2, 3], "d": [3, 2, 1]}),
            "b": pd.DataFrame({"i": [4, 5, 6], "s": [11, 12, 13], "d": [13, 12, 11]}),
        }
        with pytest.raises(ValueError):
            EdgeData(type_data, source_id="s", target_id="d")
