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

from collections import defaultdict, namedtuple
from typing import Iterable

import numpy as np
import pandas as pd

from ..globalvar import SOURCE, TARGET, WEIGHT
from .element_data import NodeData, EdgeData
from .validation import comma_sep, require_dataframe_has_columns


class ColumnarConverter:
    """
    Convert data from a columnar representation like Pandas and Numpy into values appropriate for
    element_data.py types.

    Args:
        name (str): the name of the argument for error messages
        default_type (hashable): the default type to use for data without a type
        column_defaults (dict of hashable to any): any default values for columns (using names before renaming!)
        selected_columns (dict of hashable to hashable): renamings for columns, mapping original name to new name
        allow_features (bool): if True, columns that aren't selected are returned as a numpy feature matrix
        dtype (str or numpy dtype): the data type to use for the feature matrices
    """

    def __init__(
        self,
        name,
        default_type,
        column_defaults,
        selected_columns,
        allow_features,
        dtype=None,
    ):
        self._parent_name = name
        self.column_defaults = column_defaults
        self.selected_columns = selected_columns
        self.default_type = default_type
        self.allow_features = allow_features
        self.dtype = dtype

    def name(self, type_name=None):
        if type_name is None:
            return self._parent_name
        return f"{self._parent_name}[{type_name!r}]"

    def _convert_single(self, type_name, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"{self.name(type_name)}: expected pandas DataFrame, found {type(data)}"
            )

        existing = set(self.selected_columns).intersection(data.columns)
        # split the dataframe based on the columns we know about
        known = data[existing]
        other = data.drop(columns=existing)

        # add the defaults to for columns that need one (this will not overwrite any existing ones
        # of the same name, because they'll be in `other`, not `known`)
        defaults_required = {
            name: value
            for name, value in self.column_defaults.items()
            if name not in existing
        }
        known = known.assign(**defaults_required)

        # now verify that all of the columns needed are there (that is, the defaults have filled out appropriately)
        require_dataframe_has_columns(
            self.name(type_name), known, self.selected_columns
        )

        if self.allow_features:
            features = other.to_numpy(dtype=self.dtype)
        elif len(other.columns) == 0:
            features = None
        else:
            raise ValueError(
                f"{self.name(type_name)}: expected zero feature columns, found {comma_sep(other.columns)}"
            )

        return known.rename(columns=self.selected_columns), features

    def convert(self, elements):
        if isinstance(elements, pd.DataFrame):
            elements = {self.default_type: elements}

        if not isinstance(elements, dict):
            raise TypeError(f"{self.name()}: expected dict, found {type(elements)}")

        singles = {
            type_name: self._convert_single(type_name, data)
            for type_name, data in elements.items()
        }
        return (
            {type_name: shared for type_name, (shared, _) in singles.items()},
            {type_name: features for type_name, (_, features) in singles.items()},
        )


def convert_nodes(data, *, name, default_type, dtype) -> NodeData:
    converter = ColumnarConverter(
        name,
        default_type,
        column_defaults={},
        selected_columns={},
        allow_features=True,
        dtype=dtype,
    )
    nodes, node_features = converter.convert(data)
    return NodeData(nodes, node_features)


DEFAULT_WEIGHT = 1


def convert_edges(
    data,
    node_data: NodeData,
    *,
    name,
    default_type,
    source_column,
    target_column,
    weight_column,
):
    converter = ColumnarConverter(
        name,
        default_type,
        column_defaults={weight_column: DEFAULT_WEIGHT},
        selected_columns={
            source_column: SOURCE,
            target_column: TARGET,
            weight_column: WEIGHT,
        },
        allow_features=False,
    )
    edges, edge_features = converter.convert(data)
    assert all(features is None for features in edge_features.values())

    return EdgeData(edges, node_data)


NodeInfo = namedtuple("NodeInfo", ["ids", "features"])


def _empty_node_info() -> NodeInfo:
    return NodeInfo([], [])


def _features_from_attributes(node_type, num_nodes, values, dtype):
    # the size is the first element that has a length, or None if there's only None elements.
    size = next((len(x) for x in values if x is not None), None)

    if size is None:
        # no features = zero-dimensional features, and skip the loop below
        return np.empty((num_nodes, 0), dtype)

    default_value = np.zeros(size, dtype)

    def compute_value(x):
        if x is None:
            return default_value
        elif len(x) != size:
            raise ValueError(
                f"inferred all nodes of type {node_type!r} to have feature dimension {size}, found dimension {len(x)}"
            )

        return x

    matrix = np.array([compute_value(x) for x in values], dtype)
    assert matrix.shape == (num_nodes, size)

    return matrix


def _features_from_node_data(nodes, data, dtype):
    if isinstance(data, dict):

        def single(node_type):
            node_info = nodes[node_type]
            this_data = data[node_type]

            if isinstance(this_data, pd.DataFrame):
                df = this_data.astype(dtype, copy=False)
            elif isinstance(this_data, (Iterable, list)):
                ids, values = zip(*this_data)
                df = pd.DataFrame(values, index=ids, dtype=dtype)

            graph_ids = set(node_info.ids)
            data_ids = set(df.index)
            if graph_ids != data_ids:
                parts = []
                missing = graph_ids - data_ids
                if missing:
                    parts.append(f"missing from data ({comma_sep(missing)})")
                extra = data_ids - graph_ids
                if extra:
                    parts.append(f"extra in data ({comma_sep(extra)})")
                message = " and ".join(parts)
                raise ValueError(
                    f"expected feature node IDs to exactly match nodes in graph; found: {message}"
                )

            return df

        return {node_type: single(node_type) for node_type in nodes.keys()}
    elif isinstance(data, pd.DataFrame):
        if len(nodes) > 1:
            raise TypeError(
                "When there is more than one node type, pass node features as a dictionary."
            )

        node_type = next(iter(nodes))
        return _features_from_node_data(nodes, {node_type: data}, dtype)
    elif isinstance(data, (Iterable, list)):
        id_to_data = dict(data)
        return {
            node_type: pd.DataFrame(
                (id_to_data[x] for x in node_info.ids), index=node_info.ids, dtype=dtype
            )
            for node_type, node_info in nodes.items()
        }


def _fill_or_assign(df, column, default):
    if column in df.columns:
        df.fillna({column: default}, inplace=True)
    else:
        df[column] = default


def from_networkx(
    graph,
    *,
    node_type_name,
    edge_type_name,
    node_type_default,
    edge_type_default,
    edge_weight_label,
    node_features,
    dtype,
):
    import networkx as nx

    nodes = defaultdict(_empty_node_info)

    features_in_node = isinstance(node_features, str)

    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get(node_type_name, node_type_default)
        node_info = nodes[node_type]
        node_info.ids.append(node_id)
        if features_in_node:
            node_info.features.append(node_data.get(node_features, None))

    if features_in_node or node_features is None:
        node_frames = {
            node_type: pd.DataFrame(
                _features_from_attributes(
                    node_type, len(node_info.ids), node_info.features, dtype
                ),
                index=node_info.ids,
            )
            for node_type, node_info in nodes.items()
        }
    else:
        node_frames = _features_from_node_data(nodes, node_features, dtype)

    edges = nx.to_pandas_edgelist(graph, source=SOURCE, target=TARGET)
    _fill_or_assign(edges, edge_type_name, edge_type_default)
    _fill_or_assign(edges, edge_weight_label, DEFAULT_WEIGHT)
    edges_limited_columns = edges[[SOURCE, TARGET, edge_type_name, edge_weight_label]]
    edge_frames = {
        edge_type: data.drop(columns=edge_type_name)
        for edge_type, data in edges_limited_columns.groupby(edge_type_name)
    }

    return node_frames, edge_frames
