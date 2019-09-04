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


def test_no_nodes():
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
    # We now know there is only 1 node type!
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


######################################################


def test_typed_pandas_data():
    # Node data for type t1:
    nodes1 = [(1, 2), (2, 3), (3, 4)]
    df1 = pd.DataFrame(nodes1, columns=["node_id", "one"])
    # Node data for type t2:
    nodes2 = [(-4, 4, "x"), (-5, 5, "y")]
    # XXX Column ordering is assumed same for all blocks of node data!
    df2 = pd.DataFrame(nodes2, columns=["two", "node_id", "three"])
    data = {None: df1, "t2": df2}  # Type of df1 will be default
    nd = node_data(data, node_id="node_id", default_node_type="t1")
    assert nd.is_identified()
    assert nd.is_typed()
    assert nd.is_heterogeneous()
    assert len(nd.node_types()) == 2
    assert nd.node_types() == set(["t1", "t2"])
    assert nd.num_nodes() == len(df1) + len(df2)
    _node_ids = set()
    _count = 0
    for node in nd.nodes():
        _count += 1
        assert node.node_id in list(range(1, 6))
        _node_ids.add(node.node_id)
        assert node.node_type in ["t1", "t2"]
    assert nd.num_nodes() == _count
    assert _node_ids == set(range(1, 6))
    # XXX Size and heterogeneity should not have changed!
    assert nd.is_heterogeneous()
    assert nd.node_types() == set(["t1", "t2"])
    assert nd.num_nodes() == 5


def test_typed_no_data():
    nd = TypeDictNodeData({}, node_id="yes", default_node_type="fred")
    assert nd.is_homogeneous()
    assert nd.is_unidentified()
    assert nd.is_untyped()
    assert nd.num_nodes() == 0
    assert nd.node_types() == set()
    assert nd.default_node_type() == "fred"
    count = 0
    for node in nd.nodes():
        count += 1
    assert count == 0


def test_typed_pandas_partial_data():
    # Empty node data for type t1:
    df1 = pd.DataFrame([], columns=["a", "b"])
    # Node data for type t2:
    nodes2 = [(2, 1), (3, 2), (4, 3)]
    df2 = pd.DataFrame(nodes2, columns=["a", "b"])
    data = {"t1": df1, "t2": df2}
    nd = node_data(data)
    # Since data blocks are of known size, types should be known.
    assert not nd.is_heterogeneous()
    assert nd.is_homogeneous()
    assert len(nd.node_types()) == 1
    assert nd.node_types() == set(["t2"])
    assert nd.num_nodes() == len(df2)
    _count = -1
    for node in nd.nodes():
        _count += 1
        assert node.node_id == _count
        assert node.node_type == "t2"
    assert _count == len(df2) - 1


def test_typed_nested_data():
    # The basic idea behind this test is that
    # the dictionary keys are the types AND
    # these types override any types from
    # the blocks of node data (the dictionary values).
    data = [(i, "t" + str(i)) for i in range(2, 12)]
    df = pd.DataFrame(data, columns=["node_id", "node_type"])
    nd_inner = node_data(df, node_id="node_id", node_type="node_type")
    # Types should be those specified
    assert nd_inner.is_identified()
    _count = 1
    for node in nd_inner.nodes():
        _count += 1
        assert node.node_id == _count
        assert node.node_type == "t" + str(_count)

    # Now nest the node types:
    nd = node_data({"inner": nd_inner})
    # Types should be overridden to "inner"
    assert nd.is_unidentified()
    _count = -1
    for node in nd.nodes():
        _count += 1
        assert node.node_id == _count
        assert node.node_type == "inner"

    # Now doubly nest the types:
    nd = node_data({"outer": {"inner": nd_inner, "ignored": []}})
    # Types should be overridden to "outer"
    assert nd.is_unidentified()
    _count = -1
    for node in nd.nodes():
        _count += 1
        assert node.node_id == _count
        assert node.node_type == "outer"


######################################################
