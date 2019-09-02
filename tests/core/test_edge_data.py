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
import pandas as pd
from stellargraph.core.edge_data import *
from stellargraph.globalvar import *


def test_no_edges():
    ed = NoEdgeData()
    assert ed.is_undirected()
    assert not ed.is_directed()
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
        edge_id=USE_PANDAS_INDEX,
    )
    assert ed.is_directed()
    assert ed.is_homogeneous()
    assert ed.num_edges() == 2
    # XXX Edge types should be unknown before iteration!
    assert ed.edge_types() is None
    it = iter(ed.edges())
    edge = next(it)
    assert edge.source_id == 1
    assert edge.target_id == 2
    assert edge.edge_id == "e1"
    assert edge.edge_type == EDGE_TYPE_DEFAULT
    edge = next(it)
    assert edge.source_id == 2
    assert edge.target_id == 1
    assert edge.edge_id == "e2"
    assert edge.edge_type == EDGE_TYPE_DEFAULT
    try:
        edge = next(it)
        assert False
    except StopIteration:
        assert True
    # XXX Edge types should be known after iteration!
    assert len(ed.edge_types()) == 1
    assert ed.edge_types() == set([EDGE_TYPE_DEFAULT])


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
    assert ed.is_heterogeneous()
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
    assert ed.edge_types() == set(["t1", "t2", "t3"])


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
    assert ed.edge_types() == set(["t1"])
    # We now know there is only 1 edge type!
    assert ed.is_homogeneous()
