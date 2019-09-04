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
import numpy as np
from stellargraph.core.node_data import *


def test_no_edges():
    nd = NoNodeData()
    assert nd.default_node_type() == DEFAULT_NODE_TYPE
    assert nd.is_untyped()
    assert not nd.is_typed()
    assert nd.is_unidentified()
    assert not nd.is_identified()
    assert nd.is_homogeneous()
    assert not nd.is_heterogeneous()
    assert len(nd.node_types()) == 0
    assert nd.node_types() == set()

    nd = NoNodeData(default_node_type="fred")
    assert nd.default_node_type() == "fred"


def test_pandas_data_use_index_no_type():
    # No explicit node id or type, just attributes
    data = [(1,), (2,)]
    # Set node id on index
    df = pd.DataFrame(data, columns=["info"], index=["n1", "n2"])
    # Use index for id
    nd = node_data(df, node_id=PANDAS_INDEX, default_node_type="fred")
    assert nd.is_homogeneous()
    assert nd.num_nodes() == 2
    assert nd.node_types() == set(["fred"])
    _iter = iter(nd.nodes())
    node = next(_iter)
    assert node.node_id == "n1"
    assert node.node_type == "fred"
    node = next(_iter)
    assert node.node_id == "n2"
    assert node.node_type == "fred"
    try:
        node = next(_iter)
        assert False
    except StopIteration:
        assert True
    # XXX Node types must be known after iteration!
    assert len(nd.node_types()) == 1
    assert nd.node_types() == set(["fred"])


def test_pandas_data_use_index_with_type():
    # Explicit node type plus junk
    data = [("t1", 1), ("t2", 2)]
    # Set node id on index
    df = pd.DataFrame(data, columns=["type", "info"], index=["n1", "n2"])
    # Use index for id
    nd = node_data(df, node_id=PANDAS_INDEX, node_type="type", default_node_type="fred")
    assert nd.num_nodes() == 2
    # XXX Type heterogeneity should be unknown until after iteration!
    assert nd.is_homogeneous() is None
    assert nd.is_heterogeneous() is None
    assert nd.node_types() is None
    _count = -1
    for node in nd.nodes():
        _count += 1
        assert node.node_id == "n" + str(_count + 1)
        assert node.node_type == "t" + str(_count + 1)
    assert _count == 1
    # XXX Node types must be known after iteration!
    assert not nd.is_homogeneous()
    assert nd.is_heterogeneous()
    assert len(nd.node_types()) == 2
    assert nd.node_types() == set(["t1", "t2"])


def test_pandas_data_not_use_index():
    # No explicit node id but explicit type
    data = [(1, 2, "t1"), (2, 3, "t2"), (3, 4, "t3")]
    # Set node id on index - we won't use it!
    df = pd.DataFrame(
        data, columns=["one", "two", "node_type"], index=["n1", "n2", "n3"]
    )
    # Do NOT use index for id
    nd = node_data(df, node_id=None, node_type=2)
    # XXX Node types should be unknown in advance.
    assert nd.is_heterogeneous() is None
    assert nd.is_homogeneous() is None
    assert nd.num_nodes() == 3
    # XXX Node types should be unknown before iteration!
    assert nd.node_types() is None
    count = -1
    for node in nd.nodes():
        count += 1
        assert node.node_id == count
        assert node.node_type == "t" + str(count + 1)
    # XXX Node types should be known after iteration!
    assert len(nd.node_types()) == 3
    assert nd.node_types() == set(["t1", "t2", "t3"])
    assert nd.is_heterogeneous()
    assert not nd.is_homogeneous()


def test_pandas_data_explicit():
    # Explicit node id and single type
    data = [
        (1, 2, "n1", "t1"),
        (2, 3, "n2", "t1"),
        (3, 4, "n3", "t1"),
        (4, 5, "n4", "t1"),
    ]
    df = pd.DataFrame(data, columns=["one", "two", "node_id", "node_type"])
    nd = node_data(df, node_id=2, node_type="node_type")
    # XXX Heterogeneity will change after iteration!
    assert nd.is_heterogeneous() is None
    assert nd.is_homogeneous() is None
    assert nd.num_nodes() == 4
    # XXX Node types should be unknown before iteration!
    assert nd.node_types() is None
    count = 0
    for node in nd.nodes():
        count += 1
        assert node.node_id == "n" + str(count)
        assert node.node_type == "t1"
    assert nd.num_nodes() == count
    # XXX Node types should be known after iteration!
    assert len(nd.node_types()) == 1
    assert nd.node_types() == set(["t1"])
    # We now know there is only 1 edge type!
    assert nd.is_homogeneous()
    assert not nd.is_heterogeneous()


def test_pandas_no_data():
    data = pd.DataFrame([], columns=["id", "type"])
    nd = PandasNodeData(data, node_id="id", node_type="type")
    # XXX This should be consistent with NoNodeData!
    assert nd.is_untyped()
    assert not nd.is_typed()
    assert nd.is_unidentified()
    assert not nd.is_identified()
    assert nd.is_homogeneous()
    assert not nd.is_heterogeneous()
    assert nd.num_nodes() == 0
    assert nd.node_types() == set()
    count = -1
    for node in nd.nodes():
        count += 1
    assert count == -1
