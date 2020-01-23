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

import pandas as pd

from ..globalvar import SOURCE, TARGET, WEIGHT
from .element_data import NodeData, EdgeData
from .validation import require_dataframe_has_columns


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
    """

    def __init__(
        self, name, default_type, column_defaults, selected_columns, allow_features
    ):
        self._parent_name = name
        self.column_defaults = column_defaults
        self.selected_columns = selected_columns
        self.default_type = default_type
        self.allow_features = allow_features

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

        # add the defaults to for columns that need one
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
            features = other.to_numpy()
        elif len(other.columns) == 0:
            features = None
        else:
            raise ValueError(
                f"{self.name(type_name)}: expected zero feature columns, found {other.columns}"
            )

        return known.rename(columns=self.selected_columns), features

    def convert(self, elements):
        if isinstance(elements, pd.DataFrame):
            elements = {self.default_type: elements}

        if not isinstance(elements, dict):
            raise TypeError(f"{name}: expected dict, found {type(elements)}")

        singles = {
            type_name: self._convert_single(type_name, data)
            for type_name, data in elements.items()
        }
        return (
            {type_name: shared for type_name, (shared, _) in singles.items()},
            {type_name: features for type_name, (_, features) in singles.items()},
        )


def convert_nodes(data, *, name, default_type) -> NodeData:
    converter = ColumnarConverter(
        name, default_type, column_defaults={}, selected_columns={}, allow_features=True
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
