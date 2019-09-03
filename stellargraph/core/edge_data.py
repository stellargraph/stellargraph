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
have to be deferred. In particular:
    - If the number of edges cannot be known in advance, then num_edges()
        will return None.
    - If the heterogeneity or homogeneity of the edge types cannot
        be determined in advance, then is_heterogenous() and is_homogeneous()
        will return None.
    - If the set of distinct edge types cannot be determined in advance,
        then edge_types() will return None.
After a full pass through the edge data, these method are guaranteed to
return correct values.

Supported data input formats:
    - Pandas array
    - NumPy array
    - [TODO] collection of objects with fields
    - [TODO] collection of indexable objects
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
    - For a data type with named columns, a 'position' can be a column
        name or column index.
    - For a collection of objects with fields, it will be a field name.
    - For a collection of dictionary objects, it will be a key value.
    - For a collection of other indexable objects (such as a list of lists
        or a list of tuples, etc.), the 'position' will be an integer index value.
    - For a edge-type dictionary of edge data objects, the source_id, target_id
        and edge_id positions are assumed to be the same for each block of
        edge data.
"""

import pandas as pd
import numpy as np


# Useful constants:
DEFAULT_EDGE_TYPE = "edge"
IS_DIRECTED = True
IS_UNDIRECTED = False
IS_TYPED = True
IS_UNTYPED = False
IS_HETEROGENEOUS = True
IS_HOMOGENEOUS = False
IS_UNDETERMINED = None
USE_PANDAS_INDEX = -1


#############################################
# Classes for edge data:


class EdgeDatum(tuple):
    def __new__(cls, source_id, target_id, edge_id, edge_type):
        return tuple.__new__(EdgeDatum, (source_id, target_id, edge_id, edge_type))

    def __repr__(self):
        return "{}(source={}, target={}, id={}, type={})".format(
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

    def __init__(self, is_directed, is_typed, default_edge_type):
        """
        Args:
            is_directed: <bool> Indicates whether the supplied edges are to
                be interpreted as directed (True) or undirected (False).
            is_typed: <bool> Indicates whether (True) or not (False)
                there are explicit edge types.
            default_edge_type: The optional type to assign to edges without an explicit type.
        """
        self._is_directed = is_directed
        self._is_typed = is_typed
        self._default_edge_type = (
            default_edge_type if default_edge_type is not None else DEFAULT_EDGE_TYPE
        )
        # Currently undetermined values
        self._is_heterogeneous = None
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
        Indicates whether the edge types are known to be heterogeneous
        (True) or homogeneous (False); if this status is currently unknown,
        then a value of None will be returned.

        Returns:
             The Boolean heterogeneity status if known, or a value of None.
        """
        return self._is_heterogeneous

    def is_homogeneous(self) -> bool:
        """
        Indicates whether the edge types are known to be homogeneous
        (True) or heterogeneous (False); if this status is currently unknown,
        then a value of None will be returned.

        Returns:
             The Boolean homogeneity status if known, or a value of None.
        """
        return None if self._is_heterogeneous is None else not self._is_heterogeneous

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

    def edge_types(self) -> set:
        """
        Obtains the (possibly empty) collection of distinct edge types.

        Note: The calculation of this might be deferred until after a full
        iteration through the edge data.

        Returns:
            The set of distinct edge types if known, or a value of None.
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

    def num_edges(self) -> int:
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
        if num_edges == 0:
            self._is_heterogeneous = False
            self._edge_types = set()
        elif not self._is_typed:
            self._is_heterogeneous = False
            self._edge_types = set([self._default_edge_type])

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

    def __init__(self, is_directed=False, default_edge_type=None):
        super().__init__(is_directed, IS_UNTYPED, default_edge_type)
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
        super().__init__(is_directed, IS_TYPED, default_edge_type)
        self._is_local = edge_id is None
        self._data = _data = {}
        is_determined = True
        edge_types = set()
        num_edges = 0
        for edge_type, edge_data in data.items():
            edge_type = self._edge_type(edge_type)  # in case of None
            if edge_type in _data:  # in case of None and default
                raise ValueError(
                    "Edge types contain both None and default '{}'".format(edge_type)
                )
            if isinstance(edge_data, dict):
                raise ValueError("The type-specific edge data cannot be a dictionary")
            # Wrap type-specific data
            _data[edge_type] = block_data = to_edge_data(
                edge_data, is_directed, source_id, target_id, edge_id, None, edge_type
            )
            block_size = block_data.num_edges()
            is_determined = is_determined and block_size is not None
            if is_determined and block_size > 0:
                num_edges += block_size
                edge_types.add(edge_type)
        if is_determined:
            self._set_num_edges(num_edges)
            self._set_edge_types(edge_types)

    def edges(self):
        is_uncertain = self.num_edges() is None
        if is_uncertain:
            edge_types = set()
        _edge_id = -1
        for edge_data in self._data.values():
            for edge in edge_data.edges():
                _edge_id += 1
                if self._is_local:
                    edge = edge.with_id(_edge_id)
                if is_uncertain:
                    edge_types.add(edge.edge_type)
                yield edge
        if is_uncertain:
            self._set_num_edges(_edge_id + 1)
            self._set_edge_types(edge_types)


class PandasEdgeData(EdgeData):
    """
    Wrapper for a Pandas data-frame.

    The edge identifiers are explicit (cf implicit) if
    edge_id is (cf is not) specified. The edge identifiers
    are taken from the Pandas index if edge_id is set to
    USE_PANDAS_INDEX.

    The edge types are explicit (cf implicit) if
    edge_type is (cf is not) specified.
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
            if name == "edge_id" and value == USE_PANDAS_INDEX:
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

        if type_idx >= 0:
            edge_types = set()
        _edge_id = -1
        for row in self._data.itertuples():
            _edge_id += 1
            source_id = row[source_idx]
            target_id = row[target_idx]
            edge_id = _edge_id if id_idx < 0 else row[id_idx]
            if type_idx >= 0:
                edge_type = self._edge_type(row[type_idx])
                edge_types.add(edge_type)
            else:
                edge_type = self._default_edge_type
            yield EdgeDatum(source_id, target_id, edge_id, edge_type)
        if type_idx >= 0:
            self._set_edge_types(edge_types)


class NumPyEdgeData(EdgeData):
    """
    Wrapper for a NumPy array.

    The edge identifiers are explicit (cf implicit) if
    edge_id is (cf is not) specified.

    The edge types are explicit (cf implicit) if
    edge_type is (cf is not) specified.
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
        if len(data.shape) != 2:
            raise ValueError("Only two-dimensional arrays are supported!")
        is_heterogeneous = edge_type is not None and len(data) > 1
        super().__init__(is_directed, edge_type is not None, default_edge_type)
        self._set_num_edges(len(data))
        self._data = data
        num_cols = data.shape[1]
        self._src_idx = self.__validate_position(
            "source_id", source_id, False, num_cols
        )
        self._dst_idx = self.__validate_position(
            "target_id", target_id, False, num_cols
        )
        self._id_idx = self.__validate_position("edge_id", edge_id, True, num_cols)
        self._type_idx = self.__validate_position(
            "edge_type", edge_type, True, num_cols
        )

    @staticmethod
    def __validate_position(name, value, is_nullable, num_cols):
        if value is None:
            if is_nullable:
                return -1
        elif isinstance(value, int):
            if 0 <= value < num_cols:
                return value
        # Everything else is invalid
        raise ValueError("Invalid {}: {}".format(name, value))

    def edges(self):
        source_idx = self._src_idx
        target_idx = self._dst_idx
        id_idx = self._id_idx
        type_idx = self._type_idx
        data = self._data

        if type_idx >= 0:
            edge_types = set()
        for _edge_id in range(len(data)):
            source_id = data[_edge_id, source_idx]
            target_id = data[_edge_id, target_idx]
            edge_id = _edge_id if id_idx < 0 else data[_edge_id, id_idx]
            if type_idx >= 0:
                edge_type = self._edge_type(data[_edge_id, type_idx])
                edge_types.add(edge_type)
            else:
                edge_type = self._default_edge_type
            yield EdgeDatum(source_id, target_id, edge_id, edge_type)
        if type_idx >= 0:
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
            be interpreted as directed (True) or undirected (False).
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
    # Shortcut for empty data:
    if hasattr(data, "__len__") and len(data) == 0:
        return NoEdgeData(is_directed, default_edge_type)
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
    if isinstance(data, np.ndarray):
        return NumPyEdgeData(
            data,
            is_directed,
            source_id,
            target_id,
            edge_id,
            edge_type,
            default_edge_type,
        )


def no_edge_data(is_directed=False, default_edge_type=None):
    """
    Args:
        is_directed: <bool> Optionally indicates whether the supplied edges are to
            be interpreted as directed (True) or undirected (False; default).
        default_edge_type: The optional type to assign to edges without an explicit type.

    Returns:
        An EdgeData instance.
    """
    return NoEdgeData(is_directed, default_edge_type)
