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

import warnings

import numpy as np
import pandas as pd

from ..globalvar import SOURCE, TARGET, WEIGHT, TYPE_ATTR_NAME
from .element_data import NodeData, EdgeData
from .indexed_array import IndexedArray
from .validation import comma_sep, require_dataframe_has_columns
from .utils import (
    is_real_iterable,
    zero_sized_array,
    smart_array_concatenate,
    smart_array_index,
)


class ColumnarConverter:
    """
    Convert data from a columnar representation like Pandas and Numpy into values appropriate for
    element_data.py types.

    Args:
        name (str): the name of the argument for error messages
        default_type (hashable): the default type to use for data without a type
        type_column (hashable, optional): the name of the type column, if one is being used
        column_defaults (dict of hashable to any): any default values for columns (using names before renaming!)
        selected_columns (dict of hashable to hashable): renamings for columns, mapping original name to new name
        dtype (str or numpy dtype): the data type to use for the feature matrices
        transform_columns (dict of hashable to callable): column transformations, maps column name to transform
    """

    def __init__(
        self,
        name,
        default_type,
        type_column,
        column_defaults,
        selected_columns,
        transform_columns,
        dtype=None,
    ):
        if type_column is not None and type_column not in selected_columns:
            raise ValueError(
                f"selected_columns: expected type column ({type_column!r}) to be included when using, found only {comma_sep(list(selected_columns.keys()))}"
            )

        self._parent_name = name
        self.type_column = type_column
        self.column_defaults = column_defaults
        self.selected_columns = selected_columns
        self.default_type = default_type
        self.transform_columns = transform_columns
        self.dtype = dtype

    def name(self, type_name=None):
        if type_name is None:
            return self._parent_name
        return f"{self._parent_name}[{type_name!r}]"

    def _convert_single(self, type_name, data):
        if isinstance(data, pd.DataFrame):
            return self._convert_pandas(type_name, data)
        elif isinstance(data, (IndexedArray, np.ndarray)):
            return self._convert_rowframe(type_name, data)
        else:
            raise TypeError(
                f"{self.name(type_name)}: expected IndexedArray or pandas DataFrame, found {type(data).__name__}"
            )

    def _convert_pandas(self, type_name, data):
        assert isinstance(data, pd.DataFrame)
        existing = set(self.selected_columns).intersection(data.columns)

        ids = data.index

        # split the dataframe based on the columns we know about
        missing_columns = []

        def select_column(old_name):
            if old_name in data.columns:
                column = data[old_name].to_numpy()
            elif old_name in self.column_defaults:
                column = np.broadcast_to(self.column_defaults[old_name], len(ids))
            else:
                nonlocal missing_columns
                missing_columns.append(old_name)
                return None

            transform = self.transform_columns.get(old_name)
            if transform is not None:
                column = transform(column)

            return column

        columns = {
            new_name: select_column(old_name)
            for old_name, new_name in self.selected_columns.items()
        }
        if missing_columns:
            raise ValueError(
                f"{self.name(type_name)}: expected {comma_sep(self.selected_columns)} columns, found: {comma_sep(data.columns)}"
            )

        if len(existing) != len(data.columns):
            other = data.drop(columns=existing)

            # to_numpy returns an unspecified order but it's Fortran in practice. Row-level bulk
            # operations are more common (e.g. slicing out a couple of row, when sampling a few
            # nodes) than column-level ones so having rows be contiguous (C order) is much more
            # efficient.
            features = np.ascontiguousarray(other.to_numpy(dtype=self.dtype))
        else:
            # if there's no extra columns we can save some effort and some memory usage by entirely
            # avoiding the Pandas tricks
            features = zero_sized_array((len(data), 0), self.dtype)

        return ids, columns, features

    def _convert_rowframe(self, type_name, data):
        assert isinstance(data, (IndexedArray, np.ndarray))
        if self.selected_columns:
            raise ValueError(
                f"{self.name(type_name)}: expected a Pandas DataFrame when selecting columns {comma_sep(self.selected_columns)}, found {type(data).__name__}"
            )

        if isinstance(data, np.ndarray):
            try:
                data = IndexedArray(data)
            except Exception as e:
                raise ValueError(
                    f"{self.name(type_name)}: could not convert NumPy array to a IndexedArray, see other error"
                )

        return data.index, {}, data.values

    def _ids_columns_and_type_info_from_singles(self, singles):
        type_info = []
        type_ids = []
        type_columns = defaultdict(list)

        for type_name in sorted(singles.keys()):
            ids, columns, data = singles[type_name]

            type_info.append((type_name, data))
            type_ids.append(ids)
            for col_name, col_array in columns.items():
                type_columns[col_name].append(col_array)

        if type_ids:
            ids = smart_array_concatenate(type_ids)
            columns = {
                col_name: smart_array_concatenate(col_arrays)
                for col_name, col_arrays in type_columns.items()
            }
        else:
            # there was no input types and thus no input elements, so create a dummy set of columns,
            # that is maximally flexible by using a "minimal"/highly-promotable type
            ids = []
            columns = {
                name: zero_sized_array((0,), dtype=np.uint8)
                for name in self.selected_columns.values()
            }

        return ids, columns, type_info

    def _convert_with_type_column(self, data):
        # we've got a type column, so there's no dictionaries or separate dataframes. We just need
        # to make sure things are arranged right, i.e. nodes of each type are contiguous, and
        # 'range(...)' objects describing each one.
        ids, columns, features = self._convert_single(None, data)

        # the column we see in `known` is after being selected/renamed
        type_column_name = self.selected_columns[self.type_column]

        # the shared data doesn't use the type column; that info is encoded in `type_ranges` below
        type_column = columns.pop(type_column_name)

        sorting = np.argsort(type_column)

        # arrange everything to be sorted by type
        ids = ids[sorting]
        type_column = type_column[sorting]

        # For many graphs these end up with values for which actually indexing would be suboptimal
        # (require allocating a new array, in particular), e.g. default edge weights in columns, or
        # features.size == 0.
        columns = {
            name: smart_array_index(array, sorting) for name, array in columns.items()
        }
        features = smart_array_index(features, sorting)

        # deduce the type ranges based on the first index of each of the known values
        types, first_occurance = np.unique(type_column, return_index=True)
        last_occurance = np.append(first_occurance[1:], len(type_column))

        type_info = [
            (type_name, features[start:stop, :])
            for type_name, start, stop in zip(types, first_occurance, last_occurance)
        ]

        return ids, columns, type_info

    def convert(self, elements):
        if self.type_column is not None:
            return self._convert_with_type_column(elements)

        if isinstance(elements, (pd.DataFrame, IndexedArray, np.ndarray)):
            elements = {self.default_type: elements}

        if not isinstance(elements, dict):
            raise TypeError(
                f"{self.name()}: expected dict, found {type(elements).__name__}"
            )

        singles = {
            type_name: self._convert_single(type_name, data)
            for type_name, data in elements.items()
        }

        ids, columns, type_info = self._ids_columns_and_type_info_from_singles(singles)
        return (ids, columns, type_info)


def convert_nodes(data, *, name, default_type, dtype) -> NodeData:
    converter = ColumnarConverter(
        name,
        default_type,
        type_column=None,
        column_defaults={},
        selected_columns={},
        transform_columns={},
        dtype=dtype,
    )
    ids, columns, type_info = converter.convert(data)
    assert len(columns) == 0
    return NodeData(ids, type_info)


DEFAULT_WEIGHT = np.float32(1)


def convert_edges(
    data,
    *,
    name,
    default_type,
    source_column,
    target_column,
    weight_column,
    type_column,
    nodes,
    dtype,
):
    def _node_ids_to_iloc(node_ids):
        try:
            return nodes.ids.to_iloc(node_ids, strict=True)
        except KeyError as e:
            missing_values = e.args[0]
            if not is_real_iterable(missing_values):
                missing_values = [missing_values]
            missing_values = pd.unique(missing_values)

            raise ValueError(
                f"edges: expected all source and target node IDs to be contained in `nodes`, "
                f"found some missing: {comma_sep(missing_values)}"
            )

    selected = {
        source_column: SOURCE,
        target_column: TARGET,
        weight_column: WEIGHT,
    }

    if type_column is not None:
        selected[type_column] = TYPE_ATTR_NAME

    converter = ColumnarConverter(
        name,
        default_type,
        type_column=type_column,
        column_defaults={weight_column: DEFAULT_WEIGHT},
        selected_columns=selected,
        transform_columns={
            source_column: _node_ids_to_iloc,
            target_column: _node_ids_to_iloc,
        },
        dtype=dtype,
    )
    ids, columns, type_info = converter.convert(data)

    weight_col = columns[WEIGHT]
    if not pd.api.types.is_numeric_dtype(weight_col):
        raise TypeError(
            f"{converter.name()}: expected weight column {weight_column!r} to be numeric, found dtype '{weight_col.dtype}'"
        )

    return EdgeData(
        ids, columns[SOURCE], columns[TARGET], weight_col, type_info, len(nodes)
    )


SingleTypeNodeIdsAndFeatures = namedtuple(
    "SingleTypeNodeIdsAndFeatures", ["ids", "features"]
)


def _empty_node_info() -> SingleTypeNodeIdsAndFeatures:
    return SingleTypeNodeIdsAndFeatures([], [])


def _features_from_attributes(node_type, ids, values, dtype):
    # the size is the first element that has a length, or None if there's only None elements.
    size = next((len(x) for x in values if x is not None), None)

    num_nodes = len(ids)

    if size is None:
        # no features = zero-dimensional features, and skip the loop below
        return zero_sized_array((num_nodes, 0), dtype)

    default_value = np.zeros(size, dtype)

    missing = []

    def compute_value(node_id, x):
        if x is None:
            missing.append(node_id)
            return default_value
        elif len(x) != size:
            raise ValueError(
                f"inferred all nodes of type {node_type!r} to have feature dimension {size}, found dimension {len(x)}"
            )

        return x

    matrix = np.array(
        [compute_value(node_id, x) for node_id, x in zip(ids, values)], dtype
    )
    assert matrix.shape == (num_nodes, size)

    if missing:
        # user code is 5 frames above the warnings.warn call
        stacklevel = 5
        warnings.warn(
            f"found the following nodes (of type {node_type!r}) without features, using {size}-dimensional zero vector: {comma_sep(missing)}",
            stacklevel=stacklevel,
        )

    return matrix


def _features_from_node_data(nodes, node_type_default, data, dtype):
    if isinstance(data, dict):

        def single(node_type):
            node_info = nodes[node_type]
            try:
                this_data = data[node_type]
            except KeyError:
                # no data specified for this type, so len(feature vector) = 0 for each node (this
                # uses a range index for columns, to match the behaviour of the other feature
                # converters here, that build DataFrames from NumPy arrays even when there's no
                # data, i.e. array.shape = (num nodes, 0))
                this_data = pd.DataFrame(columns=range(0), index=node_info.ids)

            if isinstance(this_data, pd.DataFrame):
                df = this_data.astype(dtype, copy=False)
            elif isinstance(this_data, (Iterable, list)):
                # this functionality is a bit peculiar (Pandas is generally nicer), and is
                # undocumented. Consider deprecating and removing it.
                ids, values = zip(*this_data)
                df = pd.DataFrame(values, index=ids, dtype=dtype)
            else:
                raise TypeError(
                    f"node_features[{node_type!r}]: expected DataFrame or iterable, found {type(this_data).__name__}"
                )

            graph_ids = set(node_info.ids)
            data_ids = set(df.index)
            if graph_ids != data_ids:
                parts = []
                missing = graph_ids - data_ids
                if missing:
                    parts.append(f"missing from data ({comma_sep(list(missing))})")
                extra = data_ids - graph_ids
                if extra:
                    parts.append(f"extra in data ({comma_sep(list(extra))})")
                message = " and ".join(parts)
                raise ValueError(
                    f"node_features[{node_type!r}]: expected feature node IDs to exactly match nodes in graph; found: {message}"
                )

            return df

        return {node_type: single(node_type) for node_type in nodes.keys()}
    elif isinstance(data, pd.DataFrame):
        if len(nodes) > 1:
            raise TypeError(
                "When there is more than one node type, pass node features as a dictionary."
            )

        node_type = next(iter(nodes), node_type_default)
        return _features_from_node_data(
            nodes, node_type_default, {node_type: data}, dtype
        )
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
    node_type_attr,
    edge_type_attr,
    node_type_default,
    edge_type_default,
    edge_weight_attr,
    node_features,
    dtype,
):
    import networkx as nx

    nodes = defaultdict(_empty_node_info)

    features_in_node = isinstance(node_features, str)

    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get(node_type_attr, node_type_default)
        node_info = nodes[node_type]
        node_info.ids.append(node_id)
        if features_in_node:
            node_info.features.append(node_data.get(node_features, None))

    if features_in_node or node_features is None:
        node_frames = {
            node_type: pd.DataFrame(
                _features_from_attributes(
                    node_type, node_info.ids, node_info.features, dtype
                ),
                index=node_info.ids,
            )
            for node_type, node_info in nodes.items()
        }
    else:
        node_frames = _features_from_node_data(
            nodes, node_type_default, node_features, dtype
        )

    edges = nx.to_pandas_edgelist(graph, source=SOURCE, target=TARGET)
    _fill_or_assign(edges, edge_type_attr, edge_type_default)
    _fill_or_assign(edges, edge_weight_attr, DEFAULT_WEIGHT)
    edges_limited_columns = edges[[SOURCE, TARGET, edge_type_attr, edge_weight_attr]]
    edge_frames = {
        edge_type: data.drop(columns=edge_type_attr)
        for edge_type, data in edges_limited_columns.groupby(edge_type_attr)
    }

    return node_frames, edge_frames
