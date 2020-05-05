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
from .validation import comma_sep, require_dataframe_has_columns
from .utils import is_real_iterable


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
        allow_features (bool): if True, columns that aren't selected are returned as a numpy feature matrix
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
        allow_features,
        transform_columns,
        dtype=None,
    ):
        if type_column is not None:
            if allow_features:
                raise ValueError(
                    f"allow_features: expected no features when using a type column ({type_column!r}), found them to be allowed"
                )
            if type_column not in selected_columns:
                raise ValueError(
                    f"selected_columns: expected type column ({type_column!r}) to be included when using, found only {comma_sep(list(selected_columns.keys()))}"
                )

        self._parent_name = name
        self.type_column = type_column
        self.column_defaults = column_defaults
        self.selected_columns = selected_columns
        self.default_type = default_type
        self.allow_features = allow_features
        self.transform_columns = transform_columns
        self.dtype = dtype

    def name(self, type_name=None):
        if type_name is None:
            return self._parent_name
        return f"{self._parent_name}[{type_name!r}]"

    def _convert_single(self, type_name, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"{self.name(type_name)}: expected pandas DataFrame, found {type(data).__name__}"
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

        for column, transform in self.transform_columns.items():
            known[column] = transform(known[column])

        if self.allow_features:
            # to_numpy returns an unspecified order but it's Fortran in practice. Row-level bulk
            # operations are more common (e.g. slicing out a couple of row, when sampling a few
            # nodes) than column-level ones so having rows be contiguous (C order) is much more
            # efficient.
            features = np.ascontiguousarray(other.to_numpy(dtype=self.dtype))
        elif len(other.columns) == 0:
            features = None
        else:
            raise ValueError(
                f"{self.name(type_name)}: expected zero feature columns, found {comma_sep(other.columns)}"
            )

        return known.rename(columns=self.selected_columns), features

    def _shared_and_starts_from_singles(self, singles):
        rows_so_far = 0
        type_starts = []
        type_dfs = []

        for type_name in sorted(singles.keys()):
            type_data, _ = singles[type_name]

            type_starts.append((type_name, rows_so_far))
            type_dfs.append(type_data)

            rows_so_far += len(type_data)

        if type_dfs:
            shared = pd.concat(type_dfs)
        else:
            # there was no input types and thus no input elements, so create a dummy dataframe, that
            # is maximally flexible by using a "minimal"/highly-promotable type
            shared = pd.DataFrame(
                columns=self.selected_columns.values(), dtype=np.uint8
            )

        return shared, type_starts

    def _convert_with_type_column(self, data):
        # we've got a type column, so there's no dictionaries or separate dataframes. We just need
        # to make sure things are arranged right, i.e. nodes of each type are contiguous, and
        # 'range(...)' objects describing each one.
        known, features = self._convert_single(None, data)
        assert features is None

        # the column we see in `known` is after being selected/renamed
        selected_type_column = self.selected_columns[self.type_column]

        known.sort_values(selected_type_column, inplace=True)

        # the shared data doesn't use the type column; that info is encoded in `type_ranges` below
        shared = known.drop(columns=selected_type_column)

        # deduce the type ranges based on the first index of each of the known values
        types, first_occurance = np.unique(
            known[selected_type_column], return_index=True
        )

        type_starts = [
            (type_name, start) for type_name, start in zip(types, first_occurance)
        ]

        # per the assert above, the features are None for every type
        features = {type_name: None for type_name in types}

        return shared, type_starts, features

    def convert(self, elements):
        if self.type_column is not None:
            return self._convert_with_type_column(elements)

        if isinstance(elements, pd.DataFrame):
            elements = {self.default_type: elements}

        if not isinstance(elements, dict):
            raise TypeError(
                f"{self.name()}: expected dict, found {type(elements).__name__}"
            )

        singles = {
            type_name: self._convert_single(type_name, data)
            for type_name, data in elements.items()
        }

        shared, type_starts = self._shared_and_starts_from_singles(singles)
        features = {type_name: features for type_name, (_, features) in singles.items()}
        return (shared, type_starts, features)


def convert_nodes(data, *, name, default_type, dtype) -> NodeData:
    converter = ColumnarConverter(
        name,
        default_type,
        type_column=None,
        column_defaults={},
        selected_columns={},
        allow_features=True,
        transform_columns={},
        dtype=dtype,
    )
    nodes, type_starts, node_features = converter.convert(data)
    return NodeData(nodes, type_starts, node_features)


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
        allow_features=False,
    )
    edges, type_starts, edge_features = converter.convert(data)

    # validation:
    assert all(features is None for features in edge_features.values())

    weight_col = edges[WEIGHT]
    if not pd.api.types.is_numeric_dtype(weight_col):
        raise TypeError(
            f"{converter.name()}: expected weight column {weight_column!r} to be numeric, found dtype '{weight_col.dtype}'"
        )

    return EdgeData(edges, type_starts)


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
        return np.empty((num_nodes, 0), dtype)

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
