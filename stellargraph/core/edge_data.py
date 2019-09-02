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

"""
Provides a standarised way to access edge data in a variety of formats.

The general principal is that the data are iterable, with one
edge defined per element. Each edge should provide at least the node
identifiers of the source and target, a unique edge identifier
and the type of the edge. The latter two attributes might
have to be inferred.

Preferably there should only be one pass through the edge data for the
purposes of efficiency, so the calculation of some properties might
have to be deferred.

Supported data input formats:
    - Pandas array
    - NumPy array
    - collection of objects with fields
    - collection of indexable objects
    - dictionary of edge type -> edge data pairs.

Required attributes:
    - is_directed: Indicates whether the supplied edges are to
        be interpreted as directed or undirected.
    - source_id: The position of the source node identifier.
    - target_id:  The position of the target node identifier.
    - edge_id:  The position of the edge identifier (which might
        be able to be inferred from the data type).
    - edge_type: For heterogeneous graphs, the position of the
        edge type; this will be inferred from a dictionary data type.

Attribute specification:
    - For a data type with columns, a 'position' can be a column name or column index.
    - For a collection of objects with fields, it will be a field name.
    - For a collection of dictionary objects, it will be a key value.
    - For a collection of other indexable objects (such as a list of lists
        or a list of tuples, etc.), the 'position' will be an integer index value.
    - For a edge-type dictionary of edge data objects, the source_id, target_id
        and edge_id positions are assumed to be the same for each data object.
"""

__all__ = ["NoEdgeData", "to_edge_data"]

import pandas as pd
import numpy as np

from .. import globalvar


#############################################
# Classes for edge data:


class EdgeDatum(tuple):
    def __new__(cls, source_id, target_id, edge_id, edge_type):
        return tuple.__new__(EdgeDatum, (source_id, target_id, edge_id, edge_type))

    def __repr__(self):
        return "{}(src={}, dst={}, id={}, type={})".format(
            self.__class__.__name__, *self
        )

    @property
    def source_id(self):
        return self[0]

    @property
    def target_id(self):
        return self[1]

    @property
    def edge_id(self):
        return self[2]

    @property
    def edge_type(self):
        return self[3]

    def with_id(self, edge_id):
        return EdgeDatum(self[0], self[1], edge_id, self[3])


class EdgeData:
    """
    The base class for all edge data wrappers.
    """

    def __init__(self, is_directed, is_typed, default_edge_type=None):
        """
        Args:
            is_directed: <bool> Indicates whether the supplied edges are to
                be interpreted as directed or undirected.
            is_typed: <bool> Indicates whether explicit edge types
                are available.
            default_edge_type: The optional type to assign to edges without an explicit type.
        """
        self._is_directed = is_directed
        self._default_edge_type = (
            default_edge_type
            if default_edge_type is not None
            else globalvar.EDGE_TYPE_DEFAULT
        )
        self._is_heterogeneous = is_typed  # This value might change
        self._edge_types = None
        self._num_edges = None

    def is_directed(self) -> bool:
        """
        Indicates whether the edges are directed (if True) or undirected (if False).

        Returns:
             <bool> The Boolean flag.
        """
        return self._is_directed

    def is_undirected(self) -> bool:
        """
        Indicates whether the edges are undirected (if True) or directed (if False).

        Returns:
             <bool> The Boolean flag.
        """
        return not self._is_directed

    def is_heterogeneous(self) -> bool:
        """
        Indicates whether the edge types are heterogeneous (if True) or homogeneous (if False).

        Note: If True, this value might become False after an explicit pass through all edges,
        if only one type of edge is observed.

        Returns:
             <bool> The Boolean flag.
        """
        return self._is_heterogeneous

    def is_homogeneous(self) -> bool:
        """
        Indicates whether the graph is homogeneous (if True) or heterogeneous (if False).

        Returns:
             <bool> The Boolean flag.
        """
        return not self._is_heterogeneous

    def _edge_type(self, edge_type=None):
        """
        Private method to supply the default edge type if
        one is not specified.

        Args:
            event_type: The optional event type.

        Returns:
            A defined edge type.
        """
        return self._default_edge_type if edge_type is None else edge_type

    def edge_types(self):
        """
        Obtains the (possibly empty) collection of distinct edge types.

        Note: The calculation of this might be deferred until after a full
        iteration through the edge data.

        Returns:
            The distinct edge types if known, or a value of None.
        """
        return self._edge_types

    def _set_edge_types(self, edge_types):
        """
        A private method for setting the distinct edge types,
        computed after initialisation (and possibly after one
        full pass through the data).

        Args:
            edge_types: The computed edge types.
        """
        self._edge_types = edge_types
        self._is_heterogeneous = len(edge_types) > 1

    def num_edges(self):
        """
        Obtains the number of edges in the edge data.

        Returns:
             The number of edges if known, or a value of None.
        """
        return self._num_edges

    def _set_num_edges(self, num_edges):
        """
        A private method for setting the number of
        edges after an explicit computation.

        Args:
            num_edges: <int> The number of edges.
        """
        self._num_edges = num_edges

    def edges(self):
        """
        Provides an iterator or generator over edge data.
        Each element so obtained will be an EdgeDatum object.

        Returns:
             The iterator or generator.
        """
        raise NotImplementedError


#############################################
# Default class for no edge data:


class NoEdgeData(EdgeData):
    """
    Wrapper for the special case of having no edges.
    """

    def __init__(self):
        super().__init__(globalvar.IS_UNDIRECTED, globalvar.IS_UNTYPED)
        super()._set_edge_types(set())
        super()._set_num_edges(0)

    def edges(self):
        return []


#############################################
# Dictionary of edge type -> edge data pairs:


class TypeDictEdgeData(EdgeData):
    """
    Wrapper for a dictionary of edge type -> edge data pairs.

    Note that if edge_id is specified, the edge identifiers are
    assumed to be globally unique across all blocks of edge data;
    otherwise, we assume the inferred identifiers for each block
     are local to that block.
    """

    def __init__(
        self, data, is_directed, source_id, target_id, edge_id, default_edge_type
    ):
        super().__init__(is_directed, globalvar.IS_TYPED, default_edge_type)
        self._is_local = edge_id is None
        self._data = _data = {}
        for edge_type, edge_data in data.items():
            edge_type = self._edge_type(edge_type)  # in case of None
            if edge_type in _data:  # in case of None and default
                raise ValueError(
                    "Edge types contain both None and default '{}'".format(edge_type)
                )
            if isinstance(edge_data, dict):
                raise ValueError("The type-specific edge data cannot be a dictionary")
            # Wrap type-specific data
            _data[edge_type] = to_edge_data(
                edge_data, is_directed, source_id, target_id, edge_id, None, edge_type
            )
        super()._set_edge_types(set(_data.keys()))

    def num_edges(self):
        _num_edges = super().num_edges()
        if _num_edges is None:
            _num_edges = 0
            for edge_data in self._data.values():
                _num_type_edges = edge_data.num_edges()
                if _num_type_edges is None:
                    return None
                _num_edges += _num_type_edges
            self._set_num_edges(_num_edges)
        return _num_edges

    def edges(self):
        _edge_id = -1
        for edge_data in self._data.values():
            for edge in edge_data.edges():
                _edge_id += 1
                if self._is_local:
                    edge = edge.with_id(_edge_id)
                yield edge
        self._set_num_edges(_edge_id + 1)


class PandasEdgeData(EdgeData):
    """
    Wrapper for a Pandas data-frame.

    The edge identifiers are explicit (cf implicit) if
    edge_id is (cf is not) specified. The edge identifiers
    are taken from the Pandas index if edge_id is set to
    USE_PANDAS_INDEX.
    """

    def __init__(
        self,
        data,
        is_directed,
        source_id,
        target_id,
        edge_id,
        edge_type,
        default_edge_type,
    ):
        super().__init__(is_directed, edge_type is not None, default_edge_type)
        self._set_num_edges(len(data))
        self._data = data
        col_names = list(data.columns)
        self._src_idx = self.__validate_position(
            "source_id", source_id, False, col_names
        )
        self._dst_idx = self.__validate_position(
            "target_id", target_id, False, col_names
        )
        self._id_idx = self.__validate_position("edge_id", edge_id, True, col_names)
        self._type_idx = self.__validate_position(
            "edge_type", edge_type, True, col_names
        )

    @staticmethod
    def __validate_position(name, value, is_nullable, col_names):
        if value is None:
            if is_nullable:
                return None
        elif isinstance(value, int):
            if name == "edge_id" and value == globalvar.USE_PANDAS_INDEX:
                return -1
            if 0 <= value < len(col_names):
                return value
        elif isinstance(value, str):
            idx = col_names.index(value)
            if idx >= 0:
                return idx
        # Everything else is invalid
        raise ValueError("Invalid {}: {}".format(name, value))

    def edges(self):
        # Offset indices to match Pandas row object
        source_idx = self._src_idx + 1
        target_idx = self._dst_idx + 1
        id_idx = -1 if self._id_idx is None else self._id_idx + 1
        type_idx = -1 if self._type_idx is None else self._type_idx + 1

        edge_types = set()
        _edge_id = -1
        for row in self._data.itertuples():
            _edge_id += 1
            source_id = row[source_idx]
            target_id = row[target_idx]
            edge_id = _edge_id if id_idx < 0 else row[id_idx]
            edge_type = self._edge_type(None if type_idx < 0 else row[type_idx])
            edge_types.add(edge_type)
            yield EdgeDatum(source_id, target_id, edge_id, edge_type)
        self._set_edge_types(edge_types)


#############################################
# Main data wrapper method:


def to_edge_data(
    data,
    is_directed,
    source_id,
    target_id,
    edge_id=None,
    edge_type=None,
    default_edge_type=None,
):
    """
    Args:
        data: The edge data in one of the standard formats.
        is_directed: <bool> Indicates whether the supplied edges are to
            be interpreted as directed or undirected.
        source_id: The position of the source node identifier.
        target_id: The position of the target node identifier.
        edge_id: The position of the edge identifier.
            This is optional if the edge_id can be inferred from the data type.
        edge_type: The optional position of the edge type. If not specified,
            all edges will be implicitly assigned the default edge type.
        default_edge_type: The optional type to assign to edges without an explicit type.

    Returns:
        An EdgeData instance.
    """
    if isinstance(data, dict):
        return TypeDictEdgeData(
            data, is_directed, source_id, target_id, edge_id, default_edge_type
        )
    if isinstance(data, pd.DataFrame):
        return PandasEdgeData(
            data,
            is_directed,
            source_id,
            target_id,
            edge_id,
            edge_type,
            default_edge_type,
        )
