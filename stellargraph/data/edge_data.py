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

The values of some properties might not be known until after a full
pass through all of the data. For the purposes of efficiency, there
should be only one such pass, so:
    - If the number of edges cannot be known in advance, then num_edges()
        will return -1, and is_typed() and is_identified(), and their
        complements, will take on initial values as determined by the
        input parameters - these values might change.
    - If the set of distinct edge types cannot be determined in advance,
        then edge_types() will return None, and is_heterogenous() and
        is_homogeneous() will take on initial values as determined by the
        input parameters - these values might change.
After a full pass through the data, these methods are guaranteed to
return correct values.

Supported data input formats:
    - Pandas data-frame
    - NumPy array
    - dictionary of edge-type -> edge-data pairs
    - collection of indexable objects,
        e.g. list, tuple, dict, etc.
    - collection of objects with fields.

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
    - For an edge-type dictionary of edge data objects, the source_id, target_id
        and edge_id positions are assumed to be the same for each block of
        edge data.
"""
__all__ = ["to_edge_data", "EdgeData", "EdgeDatum"]

from typing import Sized, Iterable, Optional, Union, Any
from sys import maxsize as NO_POSITION
import operator

import pandas as pd
import numpy as np


#############################################
# Classes for edge data:


class EdgeDatum(tuple):
    """
    Encapsulates a single edge containing:
        - source_id: The identifier of the node from which the edge commences.
        - target_id: The identifier of the node at which the edge terminates.
        - edge_id: The identifier of the edge.
        - edge_type: The type of the edge.
    """

    def __new__(cls, source_id, target_id, edge_id, edge_type):
        return tuple.__new__(EdgeDatum, (source_id, target_id, edge_id, edge_type))

    def __repr__(self):
        return "{}(source_id={}, target_id={}, edge_id={}, edge_type={})".format(
            self.__class__.__name__, *self
        )

    @property
    def source_id(self):
        """
        Obtains the identifier of the source node.

        Returns:
             The source node identifier.
        """
        return self[0]

    @property
    def target_id(self):
        """
        Obtains the identifier of the target node.

        Returns:
             The target node identifier.
        """
        return self[1]

    @property
    def edge_id(self):
        """
        Obtains the identifier of this edge.

        Returns:
             The edge identifier.
        """
        return self[2]

    @property
    def edge_type(self):
        """
        Obtains the type of this edge.

        Returns:
             The edge type.
        """
        return self[3]

    def with_id(self, edge_id):
        """
        Helper method to replace the edge identifier.

        Args:
            edge_id: The new edge identifier.

        Returns:
            A new EdgeDatum object.
        """
        return EdgeDatum(self[0], self[1], edge_id, self[3])

    def with_type(self, edge_type):
        """
        Helper method to replace the edge type.

        Args:
            edge_type: The new edge type.

        Returns:
            A new EdgeDatum object.
        """
        return EdgeDatum(self[0], self[1], self[2], edge_type)


class EdgeData:
    """
    The base class for all edge data wrappers.
    """

    # Useful constants:
    DEFAULT_EDGE_TYPE = "edge"
    PANDAS_INDEX = -1

    def __init__(self, is_directed, is_identified, is_typed, default_edge_type):
        """
        Initialises the base edge data structure.

        Args:
            is_directed: <bool> Indicates whether the edges are to
                be interpreted as directed (True) or undirected (False).
            is_identified: <bool> Indicates whether the edges have
                explicit identifiers (True), or will be assigned implicit
                identifiers (False).
            is_typed: <bool> Indicates whether the edges have explicit
                types (True), or will be assigned the default edge type (False).
            default_edge_type: The optional type to assign to edges without an explicit type
                (defaults to the constant DEFAULT_EDGE_TYPE).
        """
        self._is_directed = is_directed
        self._default_edge_type = (
            default_edge_type
            if default_edge_type is not None
            else self.DEFAULT_EDGE_TYPE
        )
        # These values depend upon the number of edges.
        self._is_identified = is_identified
        self._is_typed = is_typed
        # This value also depends upon the data.
        self._is_heterogeneous = is_typed
        # Currently undetermined values
        self._edge_types = None
        self._num_edges = -1

    def is_directed(self) -> bool:
        """
        Indicates whether the edges are directed (True) or undirected (False).

        Returns:
             <bool> The Boolean flag.
        """
        return self._is_directed

    def is_undirected(self) -> bool:
        """
        Indicates whether the edges are undirected (True) or directed (False).

        Returns:
             <bool> The Boolean flag.
        """
        return not self._is_directed

    def is_identified(self) -> bool:
        """
        Indicates whether the node identifiers are explicit (True) or implicit (False).
        If True, this result will only change if the number of edges is initially unknown
        but later found to be zero.

        Returns:
             <bool> The Boolean flag.
        """
        return self._is_identified

    def is_unidentified(self) -> bool:
        """
        Indicates whether the node identifiers are implicit (True) or explicit (False).
        If False, this result will only change if the number of edges is initially unknown
        but later found to be zero.

        Returns:
             <bool> The Boolean flag.
        """
        return not self._is_identified

    def is_typed(self) -> bool:
        """
        Indicates whether the nodes have explicit (True) or implicit (False) types.
        If True, this result will only change if the number of edges is initially unknown
        but later found to be zero.

        Returns:
             <bool> The Boolean flag.
        """
        return self._is_typed

    def is_untyped(self) -> bool:
        """
        Indicates whether the nodes have implicit (True) or explicit (False) types.
        If False, this result will only change if the number of edges is initially unknown
        but later found to be zero.

        Returns:
             <bool> The Boolean flag.
        """
        return not self._is_typed

    def is_heterogeneous(self) -> bool:
        """
        Indicates whether the node types are heterogeneous (True) or homogeneous (False).
        If True, this result might change after a complete pass through all of the edges.

        Returns:
             The Boolean heterogeneity status.
        """
        return self._is_heterogeneous

    def is_homogeneous(self) -> bool:
        """
        Indicates whether the node types are homogeneous (True) or heterogeneous (False).
        If False, this result might change after a complete pass through all of the edges.

        Returns:
             The Boolean homogeneity status.
        """
        return not self._is_heterogeneous

    def default_edge_type(self, edge_type=None):
        """
        Helper method to supply the default edge type if
        one is not specified.

        Args:
            edge_type: The optional edge type.

        Returns:
            A defined edge type.
        """
        return self._default_edge_type if edge_type is None else edge_type

    def edge_types(self) -> Optional[set]:
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
        num_types = len(edge_types)
        self._is_heterogeneous = num_types > 1
        self._is_typed = self._is_typed and num_types > 0
        self._is_identified = self._is_identified and num_types > 0

    def num_edges(self) -> int:
        """
        Obtains the number of edges in the edge data.

        Returns:
             The number of edges if known, or a value of -1.
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
            self._set_edge_types(set())
        elif not self._is_typed:
            self._set_edge_types({self._default_edge_type})

    def edges(self):
        """
        Provides an iterator or generator over edge data.
        Each element so obtained will be an EdgeDatum object.

        Returns:
             The iterator or generator.
        """
        raise NotImplementedError


#############################################
# Base class for iterable collections of edges:


class RowEdgeData(EdgeData):
    """
    Abstract wrapper for an iterable collection of edges.
    It is assumed that it is more efficient here to
    iterate over all edges, rather than just look up edges
    by an implicit index.
    """

    def edges(self):
        is_uncertain = self.num_edges() < 0 or self.is_typed()
        if is_uncertain:
            edge_types = set()
        _edge_id = -1
        for row in self._iter_rows():
            _edge_id += 1
            edge = self._get_edge(row)
            if self.is_unidentified():
                edge = edge.with_id(_edge_id)
            if is_uncertain:
                edge_types.add(edge.edge_type)
            yield edge
        if is_uncertain:
            self._set_num_edges(_edge_id + 1)
            self._set_edge_types(edge_types)

    def _iter_rows(self):
        """
        A private method for obtaining an iterator or generator
        over the collection of raw edge data.

        Returns:
            The iterator or generator.
        """
        raise NotImplementedError

    def _get_edge(self, row):
        """
        A private method for obtaining an EdgeDatum
        from the raw row of data.

        Args:
            row: The row of attributes for the given edge.

        Returns:
             An EdgeDatum representation of the edge.
        """
        raise NotImplementedError


#############################################
# Default class for no edge data:


class NoEdgeData(EdgeData):
    """
    Wrapper for the special case of having no edges.
    """

    def __init__(self, is_directed=False, default_edge_type=None):
        super().__init__(
            is_directed,
            is_identified=False,
            is_typed=False,
            default_edge_type=default_edge_type,
        )
        super()._set_num_edges(0)

    def edges(self):
        return []


#############################################
# Dictionary of edge type -> edge data pairs:


class TypeDictEdgeData(RowEdgeData):
    """
    Wrapper for a dictionary of edge-type -> edge-data pairs.

    Note that if edge_id is specified, the edge identifiers are
    assumed to be globally unique across all blocks of edge data;
    otherwise, we assume the inferred identifiers for each block
     are local to that block, and will be discarded in favour of
     globally unique identifiers.
    """

    def __init__(
        self,
        data,
        is_directed,
        source_id,
        target_id,
        edge_id=None,
        default_edge_type=None,
    ):
        _is_identified = edge_id is not None
        super().__init__(
            is_directed,
            _is_identified,
            is_typed=True,
            default_edge_type=default_edge_type,
        )
        self._data = _data = {}
        is_determined = True
        edge_types = set()
        num_edges = 0
        for _edge_type, block_data in data.items():
            edge_type = self.default_edge_type(_edge_type)  # in case of None
            if edge_type in _data:  # in case of None and default
                raise ValueError(
                    "Edge types contain both None and default '{}'".format(edge_type)
                )
            # Wrap type-specific data
            _data[edge_type] = block_data = to_edge_data(
                block_data, is_directed, source_id, target_id, edge_id, None, edge_type
            )
            block_size = block_data.num_edges()
            is_determined = is_determined and block_size >= 0
            if block_size > 0:
                if _is_identified and block_data.is_unidentified():
                    raise ValueError(
                        "Edge data for type '{}' has local identifiers!".format(
                            _edge_type
                        )
                    )
                num_edges += block_size
                edge_types.add(edge_type)
        if is_determined:
            self._set_num_edges(num_edges)
            self._set_edge_types(edge_types)

    def _iter_rows(self):
        # XXX The dictionary values are EdgeData objects.
        for edge_type, block_data in self._data.items():
            for edge in block_data.edges():
                # XXX Cannot guarantee the inner edge type
                if edge.edge_type != edge_type:
                    edge = edge.with_type(edge_type)
                yield edge

    def _get_edge(self, row):
        # XXX A row is already an EdgeDatum object.
        return row


#############################################
# Pandas data-frame of edge data:


class PandasEdgeData(RowEdgeData):
    """
    Wrapper for a Pandas data-frame.

    The edge identifiers are taken from the Pandas index
    if edge_id is set to PANDAS_INDEX; otherwise,
    if edge_id is defined then the identifiers are taken from
    the specified column, or else are enumerated for each edge.
    """

    def __init__(
        self,
        data,
        is_directed,
        source_id,
        target_id,
        edge_id=None,
        edge_type=None,
        default_edge_type=None,
    ):
        super().__init__(
            is_directed, edge_id is not None, edge_type is not None, default_edge_type
        )
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
        # XXX The Pandas tuple has the index value at position 0,
        # so we have to offset all column positions by +1.
        if value is None:
            if is_nullable:
                return -1
        elif isinstance(value, int):
            if name == "edge_id" and value == EdgeData.PANDAS_INDEX:
                return 0
            if 0 <= value < len(col_names):
                return value + 1
        elif isinstance(value, str):
            idx = col_names.index(value)
            if idx >= 0:
                return idx + 1
        # Everything else is invalid
        raise ValueError("Invalid {}: {}".format(name, value))

    def _iter_rows(self):
        return self._data.itertuples()

    def _get_edge(self, row):
        # XXX Row is a Pandas tuple.
        source_id = row[self._src_idx]
        target_id = row[self._dst_idx]
        edge_id = None if self._id_idx < 0 else row[self._id_idx]
        edge_type = self.default_edge_type(
            None if self._type_idx < 0 else row[self._type_idx]
        )
        return EdgeDatum(source_id, target_id, edge_id, edge_type)


#############################################
# NumPy array of edge data:


class NumPyEdgeData(RowEdgeData):
    """
    Wrapper for a NumPy array.

    If edge_id is defined then the identifiers are taken from the
    specified column; otherwise they are enumerated for each edge.
    """

    def __init__(
        self,
        data,
        is_directed,
        source_id,
        target_id,
        edge_id=None,
        edge_type=None,
        default_edge_type=None,
    ):
        if len(data.shape) != 2:
            raise ValueError("Only two-dimensional arrays are supported!")
        super().__init__(
            is_directed, edge_id is not None, edge_type is not None, default_edge_type
        )
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

    def _iter_rows(self):
        # Pretend to iterate over the rows.
        return range(len(self._data))

    def _get_edge(self, row):
        # XXX The row is just the implicit edge index.
        source_id = self._data[row, self._src_idx]
        target_id = self._data[row, self._dst_idx]
        edge_id = None if self._id_idx < 0 else self._data[row, self._id_idx]
        edge_type = self.default_edge_type(
            None if self._type_idx < 0 else self._data[row, self._type_idx]
        )
        return EdgeDatum(source_id, target_id, edge_id, edge_type)


#############################################
# Iterable collection of edge data:


class IterableEdgeData(RowEdgeData):
    """
    Wrapper for an iterable collection of edges that might or might
    not have a predefined length.

    The individual edge objects are assumed to have attributes that
    are individually extractable via an index or a field.
    """

    def __init__(
        self,
        data,
        is_directed,
        source_id,
        target_id,
        edge_id=None,
        edge_type=None,
        default_edge_type=None,
    ):
        super().__init__(
            is_directed, edge_id is not None, edge_type is not None, default_edge_type
        )
        if hasattr(data, "__len__"):
            self._set_num_edges(len(data))
        self._data = data
        self._src_idx = source_id
        self._dst_idx = target_id
        self._id_idx = edge_id
        self._type_idx = edge_type

    def _iter_rows(self):
        # XXX Assumes iterable edge data
        return self._data

    def _get_edge(self, row):
        get = operator.getitem if hasattr(row, "__getitem__") else getattr
        source_id = get(row, self._src_idx)
        target_id = get(row, self._dst_idx)
        edge_id = None if self._id_idx is None else get(row, self._id_idx)
        edge_type = self.default_edge_type(
            None if self._type_idx is None else get(row, self._type_idx)
        )
        return EdgeDatum(source_id, target_id, edge_id, edge_type)


#############################################
# Main data wrapper method:


def to_edge_data(
    data: Any,
    is_directed: bool,
    source_id: Union[int, str] = NO_POSITION,
    target_id: Union[int, str] = NO_POSITION,
    edge_id: Optional[Union[int, str]] = None,
    edge_type: Optional[Union[int, str]] = None,
    default_edge_type: Optional[Any] = None,
) -> EdgeData:
    """
    Wraps the edge data with an EdgeData object.
    Has no effect if 'data' is already EdgeData.

    Args:
        data: any
            The edge data in one of the standard formats. This may be None or empty
            if there are no edge attributes (in which case parameters other than
            is_directed and default_edge_type are then ignored).
        is_directed: bool
            Indicates whether the supplied edges are to
            be interpreted as directed (True) or undirected (False).
        source_id: int or str
            The position of the source node identifier. This is only
            optional if there are no data.
        target_id: int or str
            The position of the target node identifier. This is only
            optional if there are no data.
        edge_id: int or str, optional
            The optional position of the edge identifier. If not specified,
            all edge identifiers will be obtained via enumeration.
        edge_type: int or str, optional
            The optional position of the edge type. If not specified,
            all edges will be implicitly assigned the default edge type.
        default_edge_type: any, optional
            The optional type to assign to edges without an explicit type,
            or to any edges with an edge type of None.

    Returns:
        An EdgeData instance.
    """
    # Check for known data:
    if data is None:
        return NoEdgeData(is_directed, default_edge_type)
    if isinstance(data, EdgeData):
        return data
    # Check for empty data:
    if isinstance(data, Sized) and len(data) == 0:
        return NoEdgeData(is_directed, default_edge_type)
    if source_id is NO_POSITION:
        raise ValueError("Must specify source_id")
    if target_id is NO_POSITION:
        raise ValueError("Must specify target_id")
    # Check for dictionary of edge-type -> edge-data pairs:
    if isinstance(data, dict):
        return TypeDictEdgeData(
            data, is_directed, source_id, target_id, edge_id, default_edge_type
        )
    # Check for Pandas data-frame:
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
    # Check for NumPy array:
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
    # Check for arbitrary collection:
    if isinstance(data, Iterable) or hasattr(data, "__getitem__"):
        return IterableEdgeData(
            data,
            is_directed,
            source_id,
            target_id,
            edge_id,
            edge_type,
            default_edge_type,
        )
    # Don't know this data type!
    raise ValueError("Unknown edge data type: {}".format(type(data)))


# Add helper method:
EdgeData.from_data = to_edge_data
