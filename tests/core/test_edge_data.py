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

# import pytest
import pandas as pd
import numpy as np

# Public interface:
from stellargraph.core.edge_data import to_edge_data, EdgeData, EdgeDatum

# Private interface:
from stellargraph.core.edge_data import (
    NoEdgeData,
    PandasEdgeData,
    TypeDictEdgeData,
    NumPyEdgeData,
    IterableEdgeData,
)


def test_no_edges():
    ed = NoEdgeData()
    assert ed.is_undirected()
    assert not ed.is_directed()
    assert ed.is_unidentified()
    assert not ed.is_identified()
    assert ed.is_untyped()
    assert not ed.is_typed()
    assert ed.default_edge_type() == EdgeData.DEFAULT_EDGE_TYPE
    assert ed.is_homogeneous()
    assert not ed.is_heterogeneous()
    assert ed.num_edges() == 0
    assert len(list(ed.edges())) == 0
    assert len(ed.edge_types()) == 0

    ed = to_edge_data(
        data=None,
        is_directed=True,
        default_edge_type="fred",
        edge_id="yes",
        edge_type="yes",
    )
    assert isinstance(ed, NoEdgeData)
    assert not ed.is_undirected()
    assert ed.is_directed()  # XXX Is this at all useful?
    assert ed.is_unidentified()
    assert not ed.is_identified()
    assert ed.is_untyped()
    assert not ed.is_typed()
    assert ed.default_edge_type() == "fred"
    assert ed.is_homogeneous()
    assert not ed.is_heterogeneous()
    assert ed.num_edges() == 0
    assert len(list(ed.edges())) == 0
    assert len(ed.edge_types()) == 0


def test_pandas_data_use_index():
    # No explicit edge id or type
    data = [(1, 2), (2, 1)]
    # Set edge id on index
    df = pd.DataFrame(data, columns=["source_id", "target_id"], index=["e1", "e2"])
    # Deliberately mix types of column identifiers; use index for id
    ed = to_edge_data(
        df,
        is_directed=True,
        source_id="source_id",
        target_id=1,
        edge_id=EdgeData.PANDAS_INDEX,
    )
    assert ed.is_directed()
    assert ed.is_homogeneous()
    assert ed.num_edges() == 2
    assert ed.edge_types() == {EdgeData.DEFAULT_EDGE_TYPE}
    it = iter(ed.edges())
    edge = next(it)
    assert edge.source_id == 1
    assert edge.target_id == 2
    assert edge.edge_id == "e1"
    assert edge.edge_type == EdgeData.DEFAULT_EDGE_TYPE
    edge = next(it)
    assert edge.source_id == 2
    assert edge.target_id == 1
    assert edge.edge_id == "e2"
    assert edge.edge_type == EdgeData.DEFAULT_EDGE_TYPE
    try:
        _ = next(it)
        assert False
    except StopIteration:
        assert True
    # XXX Edge types must be known after iteration!
    assert len(ed.edge_types()) == 1
    assert ed.edge_types() == {EdgeData.DEFAULT_EDGE_TYPE}


def test_pandas_data_not_use_index():
    # No explicit edge id but explicit type
    data = [(1, 2, "t1"), (2, 3, "t2"), (3, 4, "t3")]
    # Set edge id on index - we won't use it!
    df = pd.DataFrame(
        data, columns=["source_id", "target_id", "edge_type"], index=["e1", "e2", "e3"]
    )
    # Deliberately mix types of identifiers; do NOT use index for id
    ed = to_edge_data(
        df,
        is_directed=False,
        source_id=0,
        target_id=1,
        edge_id=None,
        edge_type="edge_type",
    )
    assert ed.is_undirected()
    # XXX Edge types should be unknown in advance.
    assert ed.is_heterogeneous()
    assert not ed.is_homogeneous()
    assert ed.num_edges() == 3
    # XXX Edge types should be unknown before iteration!
    assert ed.edge_types() is None
    count = -1
    for edge in ed.edges():
        count += 1
        src_id = edge.source_id
        assert src_id == count + 1
        assert edge.target_id == src_id + 1
        assert edge.edge_type == "t" + str(src_id)
        assert edge.edge_id == count
    # XXX Edge types should be known after iteration!
    assert len(ed.edge_types()) == 3
    assert ed.edge_types() == {"t1", "t2", "t3"}
    assert ed.is_heterogeneous()
    assert not ed.is_homogeneous()


def test_pandas_data_explicit():
    # Explicit edge id and type
    data = [
        (1, 2, "e1", "t1"),
        (2, 3, "e2", "t1"),
        (3, 4, "e3", "t1"),
        (4, 5, "e4", "t1"),
    ]
    df = pd.DataFrame(data, columns=["source_id", "target_id", "edge_id", "edge_type"])
    ed = to_edge_data(
        df, is_directed=True, source_id=0, target_id=1, edge_id=2, edge_type=3
    )
    assert ed.is_directed()
    # XXX Heterogeneity will change after iteration!
    assert ed.is_heterogeneous()
    assert not ed.is_homogeneous()
    assert ed.num_edges() == 4
    # XXX Edge types should be unknown before iteration!
    assert ed.edge_types() is None
    count = -1
    for edge in ed.edges():
        count += 1
        src_id = edge.source_id
        assert src_id == count + 1
        assert edge.target_id == src_id + 1
        assert edge.edge_type == "t1"
        assert edge.edge_id == "e" + str(src_id)
    # XXX Edge types should be known after iteration!
    assert len(ed.edge_types()) == 1
    assert ed.edge_types() == {"t1"}
    # We now know there is only 1 edge type!
    assert ed.is_homogeneous()
    assert not ed.is_heterogeneous()


def test_pandas_no_data():
    data = pd.DataFrame([], columns=["source_id", "target_id"])
    ed = PandasEdgeData(data, is_directed=True, source_id=0, target_id=1)
    assert ed.is_directed()
    assert ed.is_homogeneous()
    assert ed.num_edges() == 0
    assert ed.edge_types() == set()
    count = -1
    for _ in ed.edges():
        count += 1
    assert count == -1


def test_typed_pandas_data():
    # Edge data for type t1:
    edges1 = [(1, 2), (2, 3), (3, 4)]
    df1 = pd.DataFrame(edges1, columns=["source_id", "target_id"])
    # Edge data for type t2:
    edges2 = [(2, 1), (3, 2), (4, 3)]
    # XXX Column ordering is assumed same for all blocks of edge data!
    df2 = pd.DataFrame(edges2, columns=["source_id", "target_id"])
    data = {None: df1, "t2": df2}  # type will be mapped to default
    ed = to_edge_data(
        data, is_directed=True, source_id=0, target_id=1, default_edge_type="t1"
    )
    assert ed.is_directed()
    assert ed.is_heterogeneous()
    assert len(ed.edge_types()) == 2
    assert ed.edge_types() == {"t1", "t2"}
    assert ed.num_edges() == 6
    _count = -1
    for edge in ed.edges():
        _count += 1
        assert edge.edge_id == _count
        edge_type = edge.edge_type
        assert edge_type in ["t1", "t2"]
        if edge_type == "t1":
            assert edge.target_id == edge.source_id + 1
        else:
            assert edge.target_id == edge.source_id - 1
    assert _count == 5
    # XXX Size and heterogeneity should not have changed!
    assert ed.is_heterogeneous()
    assert ed.edge_types() == {"t1", "t2"}
    assert ed.num_edges() == 6


def test_typed_no_data():
    data = {}
    ed = TypeDictEdgeData(data, True, 0, 1)
    assert ed.is_directed()
    assert ed.is_homogeneous()
    assert ed.num_edges() == 0
    assert ed.edge_types() == set()
    count = -1
    for _ in ed.edges():
        count += 1
    assert count == -1


def test_typed_pandas_partial_data():
    # Edge data for type t1:
    edges1 = []
    df1 = pd.DataFrame(edges1, columns=["source_id", "target_id"])
    # Edge data for type t2:
    edges2 = [(2, 1), (3, 2), (4, 3)]
    # XXX Column ordering is assumed same for all blocks of edge data!
    df2 = pd.DataFrame(edges2, columns=["source_id", "target_id"])
    data = {"t1": df1, "t2": df2}
    ed = to_edge_data(data, is_directed=True, source_id=0, target_id=1)
    assert ed.is_directed()
    # Since data blocks are of known size, types should be known.
    assert not ed.is_heterogeneous()
    assert ed.is_homogeneous()
    assert len(ed.edge_types()) == 1
    assert ed.edge_types() == {"t2"}
    assert ed.num_edges() == 3
    _count = -1
    for edge in ed.edges():
        _count += 1
        assert edge.edge_id == _count
        edge_type = edge.edge_type
        assert edge_type == "t2"
        assert edge.target_id == edge.source_id - 1
    assert _count == 2
    # XXX Size and heterogeneity should not have changed!
    assert ed.is_homogeneous()
    assert len(ed.edge_types()) == 1
    assert ed.edge_types() == {"t2"}
    assert ed.num_edges() == 3


def test_numpy_data_explicit():
    # Explicit edge id and type
    data = np.array(
        [
            (1, 2, "e1", "t1"),
            (2, 3, "e2", "t1"),
            (3, 4, "e3", "t1"),
            (4, 5, "e4", "t1"),
        ],
        dtype="object",
    )
    ed = to_edge_data(
        data, is_directed=True, source_id=0, target_id=1, edge_id=2, edge_type=3
    )
    assert ed.is_directed()
    # XXX Heterogeneity will change after iteration!
    assert ed.is_heterogeneous()
    assert ed.num_edges() == 4
    # XXX Edge types should be unknown before iteration!
    assert ed.edge_types() is None
    count = -1
    for edge in ed.edges():
        count += 1
        src_id = edge.source_id
        assert src_id == count + 1
        assert edge.target_id == src_id + 1
        assert edge.edge_type == "t1"
        assert edge.edge_id == "e" + str(src_id)
    # XXX Edge types should be known after iteration!
    assert len(ed.edge_types()) == 1
    assert ed.edge_types() == {"t1"}
    # We now know there is only 1 edge type!
    assert ed.is_homogeneous()
    assert not ed.is_heterogeneous()


def test_numpy_data_implicit():
    # Explicit edge id and type
    data = np.array([(1, 2), (2, 3), (3, 4), (4, 5)])
    ed = to_edge_data(data, is_directed=True, source_id=0, target_id=1)
    assert ed.default_edge_type() == EdgeData.DEFAULT_EDGE_TYPE
    assert ed.is_directed()
    assert ed.is_homogeneous()
    assert ed.num_edges() == 4
    # XXX Edge types should be known before iteration!
    assert ed.edge_types() == {EdgeData.DEFAULT_EDGE_TYPE}
    count = -1
    for edge in ed.edges():
        count += 1
        src_id = edge.source_id
        assert src_id == count + 1
        assert edge.target_id == src_id + 1
        assert edge.edge_type == EdgeData.DEFAULT_EDGE_TYPE
        assert edge.edge_id == count


def test_numpy_no_data():
    data = np.reshape([], (0, 2))
    ed = NumPyEdgeData(data, is_directed=True, source_id=0, target_id=1)
    assert ed.default_edge_type() == EdgeData.DEFAULT_EDGE_TYPE
    assert ed.is_directed()
    assert ed.is_homogeneous()
    assert ed.num_edges() == 0
    assert ed.edge_types() == set()
    count = -1
    for _ in ed.edges():
        count += 1
    assert count == -1


def test_iterable_data_explicit_tuple():
    # Explicit edge id and type
    data = [
        (1, 2, "e1", "t1"),
        (2, 3, "e2", "t1"),
        (3, 4, "e3", "t1"),
        (4, 5, "e4", "t1"),
    ]
    ed = to_edge_data(data, True, *list(range(4)))
    assert ed.is_directed()
    # XXX Heterogeneity will change after iteration!
    assert ed.is_heterogeneous()
    assert ed.num_edges() == 4
    # XXX Edge types should be unknown before iteration!
    assert ed.edge_types() is None
    count = -1
    for edge in ed.edges():
        count += 1
        src_id = edge.source_id
        assert src_id == count + 1
        assert edge.target_id == src_id + 1
        assert edge.edge_type == "t1"
        assert edge.edge_id == "e" + str(src_id)
    # XXX Edge types should be known after iteration!
    assert len(ed.edge_types()) == 1
    assert ed.edge_types() == {"t1"}
    # We now know there is only 1 edge type!
    assert ed.is_homogeneous()
    assert not ed.is_heterogeneous()


def test_iterable_data_explicit_edge_datum():
    # Explicit edge id and type
    data = [
        EdgeDatum(1, 2, "e1", "t1"),
        EdgeDatum(2, 3, "e2", "t1"),
        EdgeDatum(3, 4, "e3", "t1"),
        EdgeDatum(4, 5, "e4", "t1"),
    ]
    # XXX EdgeDatum has "__getitem__" so is treated as indexable rather than
    # an object with fields! Hence we must use integer positions NOT
    # the field names.
    ed = to_edge_data(
        data, is_directed=True, source_id=0, target_id=1, edge_id=2, edge_type=3
    )
    assert ed.is_directed()
    # XXX Heterogeneity will change after iteration!
    assert ed.is_heterogeneous()
    assert ed.num_edges() == 4
    # XXX Edge types should be unknown before iteration!
    assert ed.edge_types() is None
    count = -1
    for edge in ed.edges():
        count += 1
        src_id = edge.source_id
        assert src_id == count + 1
        assert edge.target_id == src_id + 1
        assert edge.edge_type == "t1"
        assert edge.edge_id == "e" + str(src_id)
    # XXX Edge types should be known after iteration!
    assert len(ed.edge_types()) == 1
    assert ed.edge_types() == {"t1"}
    # We now know there is only 1 edge type!
    assert ed.is_homogeneous()
    assert not ed.is_heterogeneous()


def test_iterable_data_explicit_dict():
    # Explicit edge id and type
    data = [
        {"s": 1, "d": 2, "i": "e1", "t": "t1"},
        {"s": 2, "d": 3, "i": "e2", "t": "t1"},
        {"s": 3, "d": 4, "i": "e3", "t": "t1"},
        {"s": 4, "d": 5, "i": "e4", "t": "t1"},
    ]
    ed = to_edge_data(data, True, *["s", "d", "i", "t"])
    assert ed.is_directed()
    # XXX Heterogeneity will change after iteration!
    assert ed.is_heterogeneous()
    assert ed.num_edges() == 4
    # XXX Edge types should be unknown before iteration!
    assert ed.edge_types() is None
    count = -1
    for edge in ed.edges():
        count += 1
        src_id = edge.source_id
        assert src_id == count + 1
        assert edge.target_id == src_id + 1
        assert edge.edge_type == "t1"
        assert edge.edge_id == "e" + str(src_id)
    # XXX Edge types should be known after iteration!
    assert len(ed.edge_types()) == 1
    assert ed.edge_types() == {"t1"}
    # We now know there is only 1 edge type!
    assert ed.is_homogeneous()
    assert not ed.is_heterogeneous()


def test_iterable_data_implicit():
    # Explicit edge id and type
    data = [(1, 2), (2, 3), (3, 4), (4, 5)]
    ed = to_edge_data(data, is_directed=True, source_id=0, target_id=1)
    assert ed.default_edge_type() == EdgeData.DEFAULT_EDGE_TYPE
    assert ed.is_directed()
    assert ed.is_homogeneous()
    assert ed.num_edges() == 4
    # XXX Edge types should be known before iteration!
    assert ed.edge_types() == {EdgeData.DEFAULT_EDGE_TYPE}
    count = -1
    for edge in ed.edges():
        count += 1
        src_id = edge.source_id
        assert src_id == count + 1
        assert edge.target_id == src_id + 1
        assert edge.edge_type == EdgeData.DEFAULT_EDGE_TYPE
        assert edge.edge_id == count


def test_iterable_no_data():
    data = []
    ed = IterableEdgeData(data, True, 0, 1)
    assert ed.default_edge_type() == EdgeData.DEFAULT_EDGE_TYPE
    assert ed.is_directed()
    assert ed.is_homogeneous()
    assert ed.num_edges() == 0
    assert ed.edge_types() == set()
    count = -1
    for _ in ed.edges():
        count += 1
    assert count == -1


def test_iterable_data_explicit_obj():
    # Explicit edge id and type
    class MyObj:
        def __init__(self, source_id, target_id, edge_id, edge_type):
            self.src_id = source_id
            self.dst_id = target_id
            self.my_id = edge_id
            self.my_type = edge_type

    data = [
        MyObj(1, 2, "e1", "t1"),
        MyObj(2, 3, "e2", "t1"),
        MyObj(3, 4, "e3", "t1"),
        MyObj(4, 5, "e4", "t1"),
    ]
    ed = to_edge_data(data, True, "src_id", "dst_id", "my_id", "my_type")
    assert ed.is_directed()
    assert ed.is_identified()
    assert ed.is_typed()
    # XXX Heterogeneity will change after iteration!
    assert ed.is_heterogeneous()
    assert ed.num_edges() == 4
    # XXX Edge types should be unknown before iteration!
    assert ed.edge_types() is None
    count = -1
    for edge in ed.edges():
        count += 1
        src_id = edge.source_id
        assert src_id == count + 1
        assert edge.target_id == src_id + 1
        assert edge.edge_type == "t1"
        assert edge.edge_id == "e" + str(src_id)
    # XXX Edge types should be known after iteration!
    assert len(ed.edge_types()) == 1
    assert ed.edge_types() == {"t1"}
    # We now know there is only 1 edge type!
    assert ed.is_homogeneous()
    assert not ed.is_heterogeneous()


def test_iterable_data_implicit_obj():
    # Explicit edge id and type
    class MyObj:
        def __init__(self, source_id, target_id):
            self.src_id = source_id
            self.dst_id = target_id

    data = [MyObj(1, 2), MyObj(2, 3), MyObj(3, 4), MyObj(4, 5)]
    ed = to_edge_data(data, True, "src_id", "dst_id")
    assert ed.default_edge_type() == EdgeData.DEFAULT_EDGE_TYPE
    assert ed.is_directed()
    assert ed.is_unidentified()
    assert ed.is_untyped()
    assert ed.is_homogeneous()
    assert ed.num_edges() == 4
    # XXX Edge types should be known before iteration!
    assert ed.edge_types() == {EdgeData.DEFAULT_EDGE_TYPE}
    count = -1
    for edge in ed.edges():
        count += 1
        src_id = edge.source_id
        assert src_id == count + 1
        assert edge.target_id == src_id + 1
        assert edge.edge_type == EdgeData.DEFAULT_EDGE_TYPE
        assert edge.edge_id == count
