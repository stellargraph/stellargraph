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

The general principal is that the edge data are iterable, with one
edge defined (vian edge attributes) per element.

Attributes:
    - source_id: The name or position of the source node identifiers.
    - target_id: The name or position of the target node identifiers.
    - edge_id: The name or position of the edge identifiers. If not specified
        then the identifiers will be implicitly assigned.
    - edge_type: The name or position of the edge types. If not specified then
        the default edge type will be assumed (except for a dictionary
        data type, which defines the types via its keys).
    - edge_weight: The name or position of the edge weights. If not specified then
        the default edge weight will be assumed.

Supported data input formats:
    - Pandas data-frame
    - dictionary of edge-type -> edge-data pairs

Future data input formats:
    - NumPy array
    - collection of indexable objects, e.g. list, tuple, dict, etc.
    - collection of objects with fields.

Attribute specification:
    - For a data type with named columns, an attribute can specified by column name or position.
    - For a data type with indexable columns, an attribute can specified by column position.
    - For a collection of objects with fields, an attribute can specified by field name.
    - For a collection of dictionary objects, an attribute can specified by key value.
    - For a dictionary of edge-type -> edge-data pairs, the specified attributes are assumed to be
        identical and consistent for every block of edge data (although the data types of the blocks may vary).
"""

__all__ = ["EdgeData"]

from typing import Sized, Iterable, Optional, Any, Set, Union, Dict
from numbers import Number
import itertools

import pandas as pd
import numpy as np


class EdgeDataFactory(type):
    """
    Private class for instantiating the edge data interface from
    user-supplied information.
    """

    def __call__(cls, *args, **kwargs):
        if cls is EdgeData:
            data = EdgeDataFactory.process_arguments(args, kwargs)
            return EdgeDataFactory.instantiate(data, kwargs)
        else:
            return type.__call__(cls, *args, **kwargs)

    @staticmethod
    def process_arguments(args, kwargs):
        known_params = [
            "data",
            "source_id",
            "target_id",
            "edge_id",
            "edge_type",
            "edge_weight",
            "default_edge_type",
            "default_edge_weight",
        ]
        if len(args) > len(known_params):
            raise ValueError("Too many positional arguments")
        for i, arg in enumerate(args):
            param = known_params[i]
            if param in kwargs:
                raise ValueError("Multiple arguments for parameter: {}".format(param))
            kwargs[param] = arg
        for param in kwargs:
            if param not in known_params:
                raise ValueError("Unknown parameter: {}".format(param))
        return kwargs.pop(known_params[0], None)

    @staticmethod
    def instantiate(data, kwargs):
        # Check for null data:
        if data is None:
            return DefaultEdgeData(
                kwargs.get("default_edge_type"), kwargs.get("default_edge_weight")
            )
        # Check for pre-processed edge data:
        if isinstance(data, EdgeData):
            return data
        # Check for empty data:
        if isinstance(data, Sized) and len(data) == 0:
            return DefaultEdgeData(
                kwargs.get("default_edge_type"), kwargs.get("default_edge_weight")
            )
        # Check for required parameters:
        for param in ["source_id", "target_id"]:
            if param not in kwargs:
                raise ValueError(
                    "No argument supplied for required parameter: {}".format(param)
                )
        # Check for dictionary of edge-type -> edge-data pairs:
        if isinstance(data, dict):
            return TypeDictEdgeData(data, **kwargs)
        # Check for Pandas data-frame:
        if isinstance(data, pd.DataFrame):
            return PandasEdgeData(data, **kwargs)
        # Cannot (yet) handle this data type!
        raise TypeError("Unsupported edge data type: {}".format(type(data)))


class EdgeData(metaclass=EdgeDataFactory):
    """
    Encapsulation of user-supplied edge data.

    Args:
        data (any):
            The edge data in one of the standard formats. This may
            be None or empty if there are no edge attributes
            (parameters other than default_edge_type and default_edge_weight are then ignored).
        source_id (str or int):
            The name or position of the source node identifier (this is only optional
            if there are no edges specified).
        target_id (str or int):
            The name or position of the target node identifier (this is only optional
            if there are no edges specified).
        edge_id (str or int, optional):
            The name or position of the edge identifier. If not specified,
            all edge identifiers will be obtained via enumeration.
            Use the constant PANDAS_INDEX if the edge data are specified
            by a Pandas data-frame and its index provides the edge identifiers.
        edge_type (str or int, optional):
            The name or position of the edge type. If not specified,
            all edges will be implicitly assigned the default edge type,
            unless the edge data are specified by a type -> edges mapping.
        edge_weight (str or int, optional):
            The name or position of the edge weight. If not specified,
            all edges will be implicitly assigned the default edge weight.
        default_edge_type (any, optional):
            The implicit type to assign to any edge without an explicit type.
            Defaults to the constant DEFAULT_EDGE_TYPE.
        default_edge_weight (number, optional):
            The implicit weight to assign to any edge without an explicit weight.
            Defaults to the constant DEFAULT_EDGE_WEIGHT.
    """

    # Useful constants:
    DEFAULT_EDGE_TYPE = "edge"
    DEFAULT_EDGE_WEIGHT = 1.0
    PANDAS_INDEX = -1

    def is_identified(self) -> bool:
        """
        Indicates whether or not the edge identifiers have been
        explicitly given.

        Returns:
            bool: A value of True (cf False) if edge identifiers
            are explicit (cf implicit).
        """
        raise NotImplementedError

    def num_edges(self) -> int:
        """
        Obtains the number of edges in the graph.

        Returns:
             The number of edges.
        """
        raise NotImplementedError

    def edge_ids(self) -> Iterable[Any]:
        """
        Obtains the collection of edge identifiers.

        Returns:
             The edge identifiers.
        """
        raise NotImplementedError

    def source_ids(self) -> Iterable[Any]:
        """
        Obtains the collection of source node identifiers.

        Returns:
             The source node identifiers.
        """
        raise NotImplementedError

    def target_ids(self) -> Iterable[Any]:
        """
        Obtains the collection of target node identifiers.

        Returns:
             The target node identifiers.
        """
        raise NotImplementedError

    def edge_types(self) -> Iterable[Any]:
        """
        Obtains the collection of edge types.

        Returns:
             The edge types.
        """
        raise NotImplementedError

    def edge_weights(self) -> Iterable[Number]:
        """
        Obtains the collection of edge weights.

        Returns:
             The edge weights.
        """
        raise NotImplementedError

    def edge_type_set(self) -> Set[Any]:
        """
        Obtains the (possibly empty) collection of distinct edge types.

        Returns:
            set: The distinct edge types.
        """
        raise NotImplementedError

    def edge_type(self, edge_id: Any) -> Optional[Any]:
        """
        Obtains the type of an edge in the graph.

        Args:
            edge_id (any): The edge identifier.

        Returns:
            any: The edge type, or a value of None if the
            edge is not in the graph.
        """
        raise NotImplementedError

    def has_edge(self, edge_id: Any) -> bool:
        """
        Determines whether or not the given edge is in the graph.

        Args:
            edge_id (any): The edge identifier.

        Returns:
            bool: A value of True (cd False) if the edge
            is (cd is not) in the graph.
        """
        raise NotImplementedError

    def add_edge(self, source_id: Any, target_id: Any, edge_id: Any):
        """
        Adds a new edge with default attributes to the graph.

        Args:
            source_id (any): The source node identifier.
            target_id (any): The target node identifier.
            edge_id (any): The edge identifier.
        """
        raise NotImplementedError

    def default_edge_type(self) -> Any:
        """
        Obtains the default edge type.

        Returns:
            any: The default edge type.
        """
        raise NotImplementedError

    def edge_weight(self, edge_id: Any) -> Optional[Number]:
        """
        Obtains the weight of an edge in the graph.

        Args:
            edge_id (any): The edge identifier.

        Returns:
            number: The edge weight, or a value of None if the
            edge is not in the graph.
        """
        raise NotImplementedError

    def default_edge_weight(self) -> Number:
        """
        Obtains the default edge weight.

        Returns:
            number: The default edge weight.
        """
        raise NotImplementedError

    ##################################################
    # Edge relationships:

    def edges(self, include_info: bool = False) -> Iterable[tuple]:
        """
        Obtains the edges in the graph, where each edge
        is represented by a tuple containing (at least)
        the source and target node identifiers.

        Args:
            include_info (bool):
                If True, then the edge information also contains
                the edge identifier, type and weight.
        Returns:
             The collection of edges.
        """
        if include_info:
            return zip(
                self.source_ids(),
                self.target_ids(),
                self.edge_ids(),
                self.edge_types(),
                self.edge_weights(),
            )
        return zip(self.source_ids(), self.target_ids())

    def neighbour_nodes(self, node_id: Any) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes connected
        to the given node, without regards to directionality.
        Nodes may be repeated if there are multiple edges.

        Args:
            node_id (any): The identifier of the node in question.

        Returns:
            iterable: The identifiers of the neighbouring nodes.
        """
        raise NotImplementedError

    def in_nodes(self, node_id: Any) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes with edges
        directed to the given node. Nodes may be repeated if there
        are multiple edges.

        Args:
            node_id (any): The identifier of the node in question.

        Returns:
            iterable: The identifiers of the neighbouring in-nodes.
        """
        raise NotImplementedError

    def out_nodes(self, node_id: Any) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes with edges
        directed from the given node. Nodes may be repeated if there
        are multiple edges.

        Args:
            node_id (any): The identifier of the node in question.

        Returns:
            iterable: The identifiers of the neighbouring out-nodes.
        """
        raise NotImplementedError


#############################################
# Base class for an extensible collection of edges:


class DefaultIterable:
    def __init__(self, size, value):
        self.size = size
        self.value = value

    def __getitem__(self, item):
        if item < 0 or item >= self.size:
            raise IndexError("Index out of range")
        return self.value

    def __len__(self):
        return self.size


class DefaultEdgeData(EdgeData):
    """
    Base class for an initially empty but extensible collection of edges with
    default attributes.

    Args:
        default_edge_type (any, optional):
            The implicit type to assign to any edge without an explicit type.
            Defaults to the constant DEFAULT_EDGE_TYPE.
        default_edge_weight (number, optional):
            The implicit weight to assign to any edge without an explicit weight.
            Defaults to the constant DEFAULT_EDGE_WEIGHT.
        implicit_id_offset (int, optional):
            If specified, then the implicit identifiers corresponding to
            external identifiers are offset by this amount.
    """

    def __init__(
        self,
        default_edge_type: Optional[Any] = None,
        default_edge_weight: Optional[Number] = None,
        implicit_id_offset: int = 0,
    ):
        self._default_edge_type = (
            default_edge_type
            if default_edge_type is not None
            else EdgeData.DEFAULT_EDGE_TYPE
        )
        self._default_edge_weight = (
            default_edge_weight
            if default_edge_weight is not None
            else EdgeData.DEFAULT_EDGE_WEIGHT
        )
        self._offset = implicit_id_offset
        self._external_edges = []
        self._external_ids = {}

    def is_identified(self) -> bool:
        # Edges can only be added with explicit identifiers
        return True

    def default_edge_type(self) -> Any:
        return self._default_edge_type

    def default_edge_weight(self) -> Number:
        return self._default_edge_weight

    def num_edges(self) -> int:
        return len(self._external_ids)

    def source_ids(self) -> Iterable[Any]:
        return [e[0] for e in self._external_edges]

    def target_ids(self) -> Iterable[Any]:
        return [e[0] for e in self._external_edges]

    def edge_ids(self) -> Iterable[Any]:
        return [e[2] for e in self._external_edges]

    def edge_types(self) -> Iterable[Any]:
        return DefaultIterable(len(self._external_ids), self._default_edge_type)

    def edge_weights(self) -> Iterable[Number]:
        return DefaultIterable(len(self._external_ids), self._default_edge_weight)

    def add_edge(self, source_id: Any, target_id: Any, edge_id: Any):
        if self.has_edge(edge_id):
            raise ValueError("An edge with that identifier already exists")
        # Add external edge with explicit identifier
        self._external_ids[edge_id] = len(self._external_edges)
        self._external_edges.append((source_id, target_id, edge_id))

    def edge_index(self, edge_id: Any) -> int:
        """
        Obtains the ordered edge index corresponding to the given identifier.

        Args:
            edge_id (any): The edge identifier.

        Returns:
            int: The edge index, or a value of -1 if the edge is not in the graph.
        """
        edge_idx = self._external_ids.get(edge_id, -1)
        return -1 if edge_idx < 0 else self._offset + edge_idx

    def has_edge(self, edge_id: Any) -> bool:
        return self.edge_index(edge_id) >= 0

    def edge_type_set(self) -> Set:
        return set() if len(self._external_ids) == 0 else {self._default_edge_type}

    def edge_type(self, edge_id: Any) -> Optional[Any]:
        return self._default_edge_type if self.has_edge(edge_id) else None

    def edge_weight(self, edge_id: Any) -> Optional[Number]:
        return self._default_edge_weight if self.has_edge(edge_id) else None

    def edges(self, include_info: bool = False) -> Iterable[tuple]:
        if include_info:
            return [
                (e[0], e[1], e[2], self._default_edge_type, self._default_edge_weight)
                for e in self._external_edges
            ]
        else:
            return [e[0:2] for e in self._external_edges]

    def neighbour_nodes(self, node_id: Any) -> Iterable[Any]:
        # TODO Use node caches
        for src_id, dst_id, edge_id in self._external_edges:
            if src_id == node_id:
                yield dst_id
            elif dst_id == node_id:
                yield src_id

    def in_nodes(self, node_id: Any) -> Iterable[Any]:
        # TODO Use node caches
        for src_id, dst_id, edge_id in self._external_edges:
            if dst_id == node_id:
                yield src_id

    def out_nodes(self, node_id: Any) -> Iterable[Any]:
        # TODO Use node caches
        for src_id, dst_id, edge_id in self._external_edges:
            if src_id == node_id:
                yield dst_id


#############################################
# Dictionary of edge-type -> edge-data pairs:


class TypeDictEdgeData(DefaultEdgeData):
    """
    Wrapper for a dictionary of edge-type -> edge-data pairs.

    Args:
        data (dict):
            The edge data in the form of a type -> block mapping.
        source_id (str or int):
            The name or position of the source node identifier.
        target_id (str or int):
            The name or position of the target node identifier.
        edge_id (str or int, optional):
            The name or position of the edge identifier; this is assumed
            to be consistent across all blocks of edge data.
            Use the constant PANDAS_INDEX if the edge data are specified
            by a Pandas data-frame and its index provides the edge identifiers.
            The edge identifiers must be explicitly given, and are assumed to be
            globally unique across all blocks of edge data.
        edge_weight (str or int, optional):
            The name or position of the edge weight. If not specified,
            all edges will be implicitly assigned the default edge weight.
        default_edge_type (any, optional):
            The implicit type to assign to any edge without an explicit type.
            Defaults to the constant DEFAULT_EDGE_TYPE.
        default_edge_weight (number, optional):
            The implicit weight to assign to any edge without an explicit weight.
            Defaults to the constant DEFAULT_EDGE_WEIGHT.
    """

    def __init__(
        self,
        data: Dict,
        source_id: Union[str, int],
        target_id: Union[str, int],
        edge_id: Optional[Union[str, int]] = None,
        edge_weight: Optional[Union[str, int]] = None,
        default_edge_type: Optional[Any] = None,
        default_edge_weight: Optional[Number] = None,
    ):
        # Analyse the data
        _data = {}
        _num_edges = 0
        for edge_type, block_data in data.items():
            ed = EdgeData(
                block_data,
                source_id,
                target_id,
                edge_id,
                edge_weight=edge_weight,
                default_edge_type=default_edge_type,
                default_edge_weight=default_edge_weight,
            )
            ne = ed.num_edges()
            if ne > 0:
                if not ed.is_identified():
                    raise ValueError(
                        "Edge data for type {} do not have explicit identifiers".format(
                            edge_type
                        )
                    )
                _num_edges += ne
                _data[edge_type] = ed
        self._data = _data
        self._num_edges = _num_edges
        super().__init__(default_edge_type, default_edge_weight, _num_edges)

    def num_edges(self) -> int:
        return self._num_edges + super().num_edges()

    def source_ids(self) -> Iterable[Any]:
        source_ids = [ed.source_ids() for ed in self._data.values()]
        return itertools.chain(*source_ids, super().source_ids())

    def target_ids(self) -> Iterable[Any]:
        target_ids = [ed.target_ids() for ed in self._data.values()]
        return itertools.chain(*target_ids, super().target_ids())

    def edge_ids(self) -> Iterable[Any]:
        edge_ids = [ed.edge_ids() for ed in self._data.values()]
        return itertools.chain(*edge_ids, super().edge_ids())

    def edge_types(self) -> Iterable[Any]:
        edge_types = [
            DefaultIterable(ed.num_edges(), et) for et, ed in self._data.items()
        ]
        return itertools.chain(*edge_types, super().edge_types())

    def edge_weights(self) -> Iterable[Any]:
        edge_weights = [ed.edge_weights() for ed in self._data.values()]
        return itertools.chain(*edge_weights, super().edge_weights())

    def has_edge(self, edge_id: Any) -> bool:
        for ne in self._data.values():
            if ne.has_edge(edge_id):
                return True
        return super().has_edge(edge_id)

    def edge_type(self, edge_id: Any) -> Optional[Any]:
        # Check explicit identifiers
        for et, ed in self._data.items():
            if ed.has_edge(edge_id):
                return et
        # Check external identifiers
        return super().edge_type(edge_id)

    def edge_weight(self, edge_id: Any) -> Optional[Number]:
        # Check explicit identifiers
        for ed in self._data.values():
            weight = ed.edge_weight(edge_id)
            if weight is not None:
                return weight
        # Check external identifiers
        return super().edge_weight(edge_id)

    def edges(self, include_info: bool = False) -> Iterable[tuple]:
        edges = [ed.edges(include_info) for ed in self._data.values()]
        return itertools.chain(*edges, super().edges(include_info))

    def edge_type_set(self) -> Set[Any]:
        return self._data.keys() | super().edge_type_set()


#############################################
# Base class for an extensible collection of edges:


class MappedEdgeData(DefaultEdgeData):
    """
    Base class for an arbitrary collection of edges with
    default attributes.

    Args:
        default_edge_type (any, optional):
            The implicit type to assign to any edge without an explicit type.
            Defaults to the constant DEFAULT_EDGE_TYPE.
        default_edge_weight (number, optional):
            The implicit weight to assign to any edge without an explicit weight.
            Defaults to the constant DEFAULT_EDGE_WEIGHT.
        num_implicit_edges (int, optional):
            The number N of edges to be assigned implicit identifiers
            in {0, 1, ..., N-1}; defaults to N = 0.
        edge_ids (iterable, optional):
            The collection of explicit edge identifiers to be indexed.
        edge_types (iterable, optional):
            The collection of explicit edge types to be summarised.

    Note:
        If any implicit edge identifiers are assigned, then these will
        always be processed before any explicit edge identifiers.
    """

    def __init__(
        self,
        default_edge_type: Optional[Any] = None,
        default_edge_weight: Optional[Number] = None,
        num_implicit_edges: int = 0,
        edge_ids: Optional[Iterable[Any]] = None,
        edge_types: Optional[Iterable[Any]] = None,
    ):
        self._num_implicit_ids = _offset = num_implicit_edges
        if edge_ids is None:
            self._explicit_ids = {}
        else:
            self._explicit_ids = {
                edge_id: (edge_idx + _offset)
                for edge_idx, edge_id in enumerate(edge_ids)
            }
        _offset += len(self._explicit_ids)
        super().__init__(default_edge_type, default_edge_weight, _offset)
        if edge_types is None:
            self._edge_types = set() if _offset == 0 else {self.default_edge_type()}
        else:
            self._edge_types = set(edge_types)

    def is_identified(self) -> bool:
        return self._num_implicit_ids == 0

    def num_edges(self) -> int:
        return self._num_implicit_ids + len(self._explicit_ids) + super().num_edges()

    def edge_ids(self) -> Iterable[Any]:
        return itertools.chain(
            range(self._num_implicit_ids), self._explicit_ids.keys(), super().edge_ids()
        )

    def edge_index(self, edge_id: Any) -> int:
        if isinstance(edge_id, int) and 0 <= edge_id < self._num_implicit_ids:
            # Implicit identifier is the index
            return edge_id
        # Lookup explicit identifier
        edge_idx = self._explicit_ids.get(edge_id, -1)
        if edge_idx >= 0:
            return edge_idx
        # Lookup external identifier
        return super().edge_index(edge_id)

    def edge_type_set(self) -> Set:
        return self._edge_types | super().edge_type_set()

    def edges(self, include_info: bool = False) -> Iterable[tuple]:
        if include_info:
            return zip(
                self.source_ids(),
                self.target_ids(),
                self.edge_ids(),
                self.edge_types(),
                self.edge_weights(),
            )
        return zip(self.source_ids(), self.target_ids())


#############################################
# Pandas data-frame of edge data:


class PandasEdgeData(MappedEdgeData):
    """
    Wrapper for a Pandas data-frame.

    Args:
        data (DataFrame):
            The edge data in the form of a Pandas data-frame.
        source_id (str or int):
            The name or position of the source node identifier.
        target_id (str or int):
            The name or position of the target node identifier.
        edge_id (str or int, optional):
            The name or position of the edge identifier.
            If not specified,
            all edge identifiers will be obtained via enumeration.
            Use the constant PANDAS_INDEX if the data-frame index
            provides the edge identifiers.
        edge_type (str or int, optional):
            The name or position of the edge type. If not specified,
            all edges will be implicitly assigned the default edge type.
        edge_weight (str or int, optional):
            The name or position of the edge weight. If not specified,
            all edges will be implicitly assigned the default edge weight.
        default_edge_type (any, optional):
            The implicit type to assign to any edge without an explicit type.
            Defaults to the constant DEFAULT_EDGE_TYPE.
        default_edge_weight (number, optional):
            The implicit weight to assign to any edge without an explicit weight.
            Defaults to the constant DEFAULT_EDGE_WEIGHT.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        source_id: Union[str, int],
        target_id: Union[str, int],
        edge_id: Optional[Union[str, int]] = None,
        edge_type: Optional[Union[str, int]] = None,
        edge_weight: Optional[Union[str, int]] = None,
        default_edge_type: Optional[Any] = None,
        default_edge_weight: Optional[Number] = None,
    ):
        self._data = data
        # Check source and target
        columns = list(data.columns)
        self._source_id_col = self.__validate_column("source_id", source_id, columns)
        self._target_id_col = self.__validate_column("target_id", target_id, columns)
        # Check edge weights
        self._edge_weight_col = (
            None
            if edge_weight is None
            else self.__validate_column("edge_weight", edge_weight, columns)
        )
        # Check edge types
        if edge_type is None:
            # No explicit types
            self._edge_type_col = None
            _edge_types = None
        else:
            # Use data-frame column
            self._edge_type_col = self.__validate_column(
                "edge_type", edge_type, columns
            )
            _edge_types = data[self._edge_type_col].values
        # Check edge identifiers
        if edge_id is None:
            # Assign implicit identifiers
            self._edge_id_col = None
            super().__init__(
                default_edge_type,
                default_edge_weight,
                num_implicit_edges=len(data),
                edge_types=_edge_types,
            )
        elif edge_id == EdgeData.PANDAS_INDEX:
            # Use data-frame index
            self._edge_id_col = EdgeData.PANDAS_INDEX
            super().__init__(
                default_edge_type,
                default_edge_weight,
                edge_ids=data.index,
                edge_types=_edge_types,
            )
        else:
            # Use data-frame column
            self._edge_id_col = col = self.__validate_column(
                "edge_id", edge_id, columns
            )
            super().__init__(
                default_edge_type,
                default_edge_weight,
                edge_ids=data[col].values,
                edge_types=_edge_types,
            )

    @staticmethod
    def __validate_column(name, value, col_names) -> str:
        if isinstance(value, int):
            if 0 <= value < len(col_names):
                return col_names[value]
        elif isinstance(value, str):
            if value in col_names:
                return value
        # Everything else is invalid
        raise ValueError("Invalid data-frame column for {}: {}".format(name, value))

    def edge_type(self, edge_id: Any) -> Optional[Any]:
        edge_idx = self.edge_index(edge_id)
        if edge_idx < 0:
            # Not in graph
            return None
        elif edge_idx < len(self._data):
            if self._edge_type_col is None:
                # No explicit types
                return self.default_edge_type()
            else:
                # Get explicit type
                edge_types = self._data[self._edge_type_col].values
                return edge_types[edge_idx]
        else:
            # Default type
            return self.default_edge_type()

    def edge_weight(self, edge_id: Any) -> Optional[Number]:
        edge_idx = self.edge_index(edge_id)
        if edge_idx < 0:
            # Not in graph
            return None
        elif edge_idx < len(self._data):
            if self._edge_weight_col is None:
                # No explicit weights
                return self.default_edge_weight()
            else:
                # Get explicit weight
                edge_weights = self._data[self._edge_weight_col].values
                return edge_weights[edge_idx]
        else:
            # Default weight
            return self.default_edge_weight()

    def source_ids(self) -> Iterable[Any]:
        source_ids = self._data[self._source_id_col].values
        return itertools.chain(source_ids, super().source_ids())

    def target_ids(self) -> Iterable[Any]:
        target_ids = self._data[self._target_id_col].values
        return itertools.chain(target_ids, super().target_ids())

    def edge_types(self) -> Iterable[Any]:
        if self._edge_type_col is None:
            # No explicit types
            edge_types = DefaultIterable(len(self._data), self.default_edge_type())
        else:
            # Get explicit types
            edge_types = self._data[self._edge_type_col].values
        return itertools.chain(edge_types, super().edge_types())

    def edge_weights(self) -> Iterable[Number]:
        if self._edge_weight_col is None:
            # No explicit weights
            edge_weights = DefaultIterable(len(self._data), self.default_edge_weight())
        else:
            # Get explicit weights
            edge_weights = self._data[self._edge_weight_col].values
        return itertools.chain(edge_weights, super().edge_weights())

    def neighbour_nodes(self, node_id: Any) -> Iterable[Any]:
        # XXX Self nodes will be repeated in this implementation!
        return itertools.chain(
            self._data.loc[
                self._data[self._target_id_col] == node_id, self._source_id_col
            ].values,
            self._data.loc[
                self._data[self._source_id_col] == node_id, self._target_id_col
            ].values,
            super().neighbour_nodes(node_id),
        )

    def in_nodes(self, node_id: Any) -> Iterable[Any]:
        return itertools.chain(
            self._data.loc[
                self._data[self._target_id_col] == node_id, self._source_id_col
            ].values,
            super().in_nodes(node_id),
        )

    def out_nodes(self, node_id: Any) -> Iterable[Any]:
        return itertools.chain(
            self._data.loc[
                self._data[self._source_id_col] == node_id, self._target_id_col
            ].values,
            super().out_nodes(node_id),
        )
