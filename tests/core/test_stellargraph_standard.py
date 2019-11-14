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
from stellargraph.core.graph import *


def create_nodes_dict():
    """
    Creates nodes with identifiers and randomised features, in the
    form of a type map:

        { type: data_frame }

    Returns:
        dict: The type -> nodes mapping.
    """
    node_type_specs = [(6, 3, "a"), (4, 5, "b")]
    nodes = {
        node_type: pd.DataFrame(
            {
                **{"id": [f"n_{node_type}_{ii}" for ii in range(num_nodes)]},
                **{f"f{fk}": np.random.randn(num_nodes) for fk in range(num_features)},
            }
        )
        for num_nodes, num_features, node_type in node_type_specs
    }
    return nodes


def create_edges_dict(nodes):
    """
    Creates random edges with identifiers and randomised features, in the
    form of a type map:

        { type: data_frame }

    Args:
        nodes (dict): The type -> nodes mapping.

    Returns:
        dict: The type -> edges mapping.
    """
    edge_type_specs = [(10, 4, "x"), (8, 5, "y")]
    edges = {
        edge_type: pd.DataFrame(
            {
                **{
                    "id": [f"e_{edge_type}_{ii}" for ii in range(num_edges)],
                    "src": [
                        np.random.choice(nodes[k].id)
                        for k in np.random.choice(list(nodes), num_edges)
                    ],
                    "dst": [
                        np.random.choice(nodes[k].id)
                        for k in np.random.choice(list(nodes), num_edges)
                    ],
                },
                **{f"f{fk}": np.random.randn(num_edges) for fk in range(num_features)},
            }
        )
        for num_edges, num_features, edge_type in edge_type_specs
    }
    return edges


def test_typed_standard():
    nodes = create_nodes_dict()
    edges = create_edges_dict(nodes)
    g = StellarGraph(
        edges=edges,
        nodes=nodes,
        source_id="src",
        target_id="dst",
        edge_id="id",
        node_id="id",
    )
    assert not g.is_directed()
    node_ids = {_id for df in nodes.values() for _id in df["id"]}
    counter = 0
    for node_id in g.nodes():
        counter += 1
        assert node_id in node_ids
    assert len(node_ids) == counter
    edge_ids = {_id for df in edges.values() for _id in df["id"]}
    counter = 0
    for edge in g.edges(include_info=True):
        counter += 1
        assert edge[2] in edge_ids
    assert len(edge_ids) == counter


def create_homogenous_graph(has_features=False, feature_size=10):
    edge_data = pd.DataFrame([(1, 2), (2, 3), (1, 4), (3, 2)], columns=["src", "dst"])
    if has_features:
        nmat = np.zeros((4, 1 + feature_size))
        for i in range(4):
            nmat[i, 0] = i + 1
            nmat[i, 1:] = (i + 1) * np.ones(feature_size)
        feature_names = ["f{}".format(i) for i in range(feature_size)]
        node_data = pd.DataFrame(nmat, columns=["id"] + feature_names)
        return StellarGraph(
            edges=edge_data,
            source_id="src",
            target_id="dst",
            default_edge_type="default",
            nodes=node_data,
            node_id="id",
            node_features=feature_names,
            default_node_type="default",
        )
    else:
        return StellarGraph(
            edges=edge_data,
            source_id="src",
            target_id="dst",
            default_edge_type="default",
        )


def create_heterogenous_graph(has_features=False, feature_sizes={}):
    edge_data = pd.DataFrame(
        [(0, 4, "R"), (1, 4, "R"), (1, 5, "R"), (2, 4, "R"), (3, 5, "R"), (4, 5, "F")],
        columns=["src", "dst", "label"],
    )

    type_A_nodes = pd.DataFrame([0, 1, 2, 3], columns=["id"])
    type_B_nodes = pd.DataFrame([4, 5, 6], columns=["id"])
    node_data = {"A": type_A_nodes, "B": type_B_nodes}

    # Add some numeric node attributes
    if has_features:
        feature_names = {}
        for node_type in ["A", "B"]:
            num_features = feature_sizes.get(node_type, 10)
            nd = node_data[node_type]
            features = np.asarray([v * np.ones(num_features) for v in nd["id"].values])
            feature_names[node_type] = fn = [
                "f{}".format(i) for i in range(num_features)
            ]
            for i, feature_name in enumerate(fn):
                nd[feature_name] = features[:, i]
    else:
        feature_names = None

    return StellarGraph(
        edges=edge_data,
        source_id="src",
        target_id="dst",
        edge_type="label",
        nodes=node_data,
        node_id="id",
        node_features=feature_names,
    )


def test_null_node_feature():
    sg = create_homogenous_graph(has_features=True, feature_size=6)
    aa = sg.node_features([1, None, 2, None])
    assert aa.shape == (4, 6)
    assert aa[:, 0] == pytest.approx([1, 0, 2, 0])

    sg = create_heterogenous_graph(has_features=True, feature_sizes={"A": 4, "B": 2})
    # Test feature for null node, without node type
    ab = sg.node_features([None, 5, None])
    assert ab.shape == (3, 2)
    assert ab[:, 0] == pytest.approx([0, 5, 0])
