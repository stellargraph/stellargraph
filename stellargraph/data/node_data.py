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
Provides a standarised way to access node data in a variety of formats.

The general principal is that the node data are iterable, with one
node defined (via node attributes) per element.

Attributes:
    - node_id: The name or position of the node identifiers. If not specified
        then the identifiers will be implicitly assigned.
    - node_type: The name or position of the node types. If not specified then
        the default node type will be assumed (except for a dictionary
        data type, which defines the types via its keys).

Supported data input formats:
    - Pandas data-frame
    - dictionary of node-type -> node-data pairs

Future data input formats:
    - NumPy array
    - collection of indexable objects, e.g. list, tuple, dict, etc.
    - collection of objects with fields.

Attribute specification:
    - For a data type with named columns, an attribute can specified by column name or position.
    - For a data type with indexable columns, an attribute can specified by column position.
    - For a collection of objects with fields, an attribute can specified by field name.
    - For a collection of dictionary objects, an attribute can specified by key value.
    - For a dictionary of node-type -> node-data pairs, the node_id attribute is assumed to be
        identical and consistent for every block of node data (although the data types of the blocks may vary).
"""

__all__ = ["NodeData"]

from typing import Sized, Iterable, Optional, Any, Set, Union, Mapping, List
import itertools
from collections import defaultdict

import pandas as pd
import numpy as np


class NodeDataFactory(type):
    """
    Private class for instantiating the node data interface from
    user-supplied information.
    """

    def __call__(cls, *args, **kwargs):
        if cls is NodeData:
            data = NodeDataFactory.process_arguments(args, kwargs)
            return NodeDataFactory.instantiate(data, kwargs)
        else:
            return type.__call__(cls, *args, **kwargs)

    @staticmethod
    def process_arguments(args, kwargs):
        known_params = [
            "data",
            "node_id",
            "node_type",
            "node_features",
            "default_node_type",
        ]
        if len(args) > len(known_params):
            raise TypeError(
                "NodeData takes {} positional arguments but {} were given".format(
                    len(known_params), len(args)
                )
            )
        for i, arg in enumerate(args):
            param = known_params[i]
            if param in kwargs:
                raise TypeError(
                    "NodeData got multiple values for argument '{}'".format(param)
                )
            kwargs[param] = arg
        for param in kwargs:
            if param not in known_params:
                raise TypeError(
                    "NodeData got an unexpected keyword argument '{}'".format(param)
                )
        return kwargs.pop(known_params[0], None)

    @staticmethod
    def instantiate(data, kwargs):
        # Check for null data:
        if data is None:
            return DefaultNodeData(kwargs.get("default_node_type"))
        # Check for pre-processed node data:
        if isinstance(data, NodeData):
            return data
        # Check for empty data:
        if isinstance(data, Sized) and len(data) == 0:
            return DefaultNodeData(kwargs.get("default_node_type"))
        # Check for dictionary of node-type -> node-data pairs:
        if isinstance(data, dict):
            return TypeDictNodeData(data, **kwargs)
        # Check for Pandas data-frame:
        if isinstance(data, pd.DataFrame):
            return PandasNodeData(data, **kwargs)
        # Check for NumPy array:
        if isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                # One-dimensional
                raise NotImplementedError(
                    "One-dimensional NumPy arrays are not yet supported"
                )
            elif len(data.shape) == 2:
                # Two-dimensional
                raise NotImplementedError(
                    "Two-dimensional NumPy arrays are not yet supported"
                )
            else:
                raise TypeError("Expected a one- or two-dimensional NumPy array")
        # Check for arbitrary collection:
        if isinstance(data, Iterable) or hasattr(data, "__getitem__"):
            if NodeDataFactory.is_single_column(kwargs):
                # One-dimensional
                raise NotImplementedError(
                    "One-dimensional collections are not yet supported"
                )
            # Two-dimensional
            raise NotImplementedError(
                "Two-dimensional collections are not yet supported"
            )
        # Don't know this data type!
        raise TypeError("Unknown node data type: {}".format(type(data)))

    @staticmethod
    def is_single_column(kwargs):
        return (
            kwargs.get("node_id") == NodeData.SINGLE_COLUMN
            or kwargs.get("node_type") == NodeData.SINGLE_COLUMN
        )


class NodeData(metaclass=NodeDataFactory):
    """
    Encapsulation of user-supplied node data.

    Args:
        data (any):
            The node data in one of the standard formats. This may
            be None or empty if there are no node attributes
            (parameters other than default_node_type are then ignored).
        node_id (str or int, optional):
            The name or position of the node identifier. If not specified,
            all node identifiers will be obtained via enumeration.
            Use the constant SINGLE_COLUMN if the node data are specified
            by a one-dimensional collection of node identifiers.
            Use the constant PANDAS_INDEX if the node data are specified
            by a Pandas data-frame and its index provides the node identifiers.
        node_type (str or int, optional):
            The name or position of the node type. If not specified,
            all nodes will be implicitly assigned the default node type,
            unless the node data are specified by a type -> nodes mapping.
            Use the constant SINGLE_COLUMN if the node data form a
            one-dimensional collection of node types.
        node_features (iterable or dict, optional):
            Either a collection of column names or positions that comprise
            the node features, or a mapping from node type to node feature columns.
        default_node_type (any, optional):
            The implicit type to assign to any node without an explicit type.
            Defaults to the constant DEFAULT_NODE_TYPE.
    """

    # Useful constants:
    DEFAULT_NODE_TYPE = "node"
    PANDAS_INDEX = -1
    SINGLE_COLUMN = -2

    def num_nodes(self) -> int:
        """
        Obtains the number of nodes in the graph.

        Returns:
             The number of nodes.
        """
        raise NotImplementedError

    def node_ids(self) -> Iterable[Any]:
        """
        Obtains the collection of node identifiers.

        Returns:
             The node identifiers.
        """
        raise NotImplementedError

    def nodes(
        self, include_info: bool = False
    ) -> Union[Iterable[Any], Iterable[tuple]]:
        """
        Obtains the collection of nodes.

        Args:
            include_info (bool): If False (default), then only the node identifiers
            are provided; otherwise, both node identifiers and node types are
            provided.
        Returns:
             The collection of node information.
        """
        if include_info:
            return zip(self.node_ids(), self.node_types())
        return self.node_ids()

    def node_index(self, node_id: Any) -> int:
        """
        Obtains the ordered node index corresponding to the given identifier.

        Args:
            node_id (any): The node identifier.

        Returns:
            int: The node index, or a value of -1 if the node is not in the graph.
        """
        raise NotImplementedError

    def node_id(self, node_idx: int) -> Optional[Any]:
        """
        Obtains the node identifier corresponding to the given ordered index.

        Args:
            node_idx (int): The node index.

        Returns:
            any: The node identifier, or a value of None if the node is not in the graph.
        """
        raise NotImplementedError

    def has_node(self, node_id: Any) -> bool:
        """
        Determines whether or not the given node is in the graph.

        Args:
            node_id (any): The node identifier.

        Returns:
            bool: A value of True (cd False) if the node
            is (cd is not) in the graph.
        """
        return self.node_index(node_id) >= 0

    def add_node(self, node_id: Any):
        """
        Adds a new node with default attributes to the graph, if the node
        identifier has not previously been seen.

        Args:
            node_id (any): The node identifier.
        """
        raise NotImplementedError

    ###############################################
    # Methods relating to node types:

    def default_node_type(self) -> Any:
        """
        Obtains the default node type.

        Returns:
            any: The default node type.
        """
        raise NotImplementedError

    def node_types(self) -> Iterable[Any]:
        """
        Obtains the collection of node types.

        Returns:
             The node types.
        """
        raise NotImplementedError

    def node_type_set(self) -> Set[Any]:
        """
        Obtains the (possibly empty) collection of distinct node types.

        Returns:
            set: The distinct node types.
        """
        raise NotImplementedError

    def node_type(self, node_id: Any) -> Optional[Any]:
        """
        Obtains the type of a node in the graph.

        Args:
            node_id (any): The node identifier.

        Returns:
            any: The node type, or a value of None if the
            node is not in the graph.
        """
        raise NotImplementedError

    ###############################################
    # Methods relating to node features:

    def node_feature_sizes(self) -> Mapping[Any, int]:
        """
        Obtains a mapping from each node type to the number
        of node features for that type.

        Returns:
             The node-type -> node-features mapping; this is
             empty if node features are unavailable.
        """
        raise NotImplementedError

    def node_features(self, node_ids: Iterable[Any], node_type: Any = None):
        """
        Obtains the numeric feature vectors for the specified nodes as a NumPy array.

        Args:
            node_ids (iterable): A collection of node identifiers.
            node_type (any, optional): The common type of the nodes.

        Returns:
            Numpy array containing the node features for the requested nodes.
        """
        raise NotImplementedError


#############################################
# Base class for an extensible collection of nodes:


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


class DefaultNodeData(NodeData):
    """
    Base class for an initially empty but extensible collection of nodes with
    default attributes.

    Args:
        default_node_type (any, optional):
            The implicit type to assign to any node without an explicit type.
            Defaults to the constant DEFAULT_NODE_TYPE.
        implicit_id_offset (int, optional):
            If specified, then the implicit identifiers corresponding to
            external identifiers are offset by this amount.
    """

    def __init__(
        self, default_node_type: Optional[Any] = None, implicit_id_offset: int = 0
    ):
        self._default_node_type = (
            default_node_type
            if default_node_type is not None
            else NodeData.DEFAULT_NODE_TYPE
        )
        self._offset = implicit_id_offset
        self._external_ids = {}

    def default_node_type(self) -> Any:
        return self._default_node_type

    def num_nodes(self) -> int:
        return len(self._external_ids)

    def node_ids(self) -> Iterable[Any]:
        return self._external_ids.keys()

    def node_types(self) -> Iterable[Any]:
        return DefaultIterable(len(self._external_ids), self._default_node_type)

    def node_index(self, node_id: Any) -> int:
        return self._external_ids.get(node_id, -1)

    def node_id(self, node_idx: int) -> Optional[Any]:
        # Could optimise this by having an ordered list of ids.
        for node_id, _node_idx in self._external_ids.items():
            if _node_idx == node_idx:
                return node_id
        return None

    def add_node(self, node_id: Any):
        if not self.has_node(node_id):
            self._external_ids[node_id] = self._offset + len(self._external_ids)

    def node_type_set(self) -> Set[Any]:
        return set() if len(self._external_ids) == 0 else {self._default_node_type}

    def node_type(self, node_id: Any) -> Optional[Any]:
        return self._default_node_type if node_id in self._external_ids else None

    def type_features(self) -> Mapping[Any, List]:
        return {}


#############################################
# Dictionary of node-type -> node-data pairs:


class TypeDictNodeData(DefaultNodeData):
    """
    Wrapper for a dictionary of node-type -> node-data pairs.

    Args:
        data (any):
            The node data in the form of a type -> block mapping.
        node_id (str or int, optional):
            The name or position of the node identifier.
            If specified, the node identifiers are
            assumed to be globally unique across all blocks of node data.
            If not specified,
            all node identifiers will be obtained via enumeration.
            Use the constant SINGLE_COLUMN if the node data are specified
            by a one-dimensional collection of node identifiers.
            Use the constant PANDAS_INDEX if the node data are specified
            by a Pandas data-frame and its index provides the node identifiers.
        node_features (iterable or dict, optional):
            Either a collection of column names or positions that comprise
            the node features, or a mapping from node type to node feature columns.
        default_node_type (any, optional):
            The implicit type to assign to any node without an explicit type.
            Defaults to the constant DEFAULT_NODE_TYPE.
    """

    def __init__(self, data, node_id=None, node_features=None, default_node_type=None):
        # Check the feature structure
        if isinstance(node_features, dict):
            _node_features = node_features
            for nt, nf in node_features.items():
                if not isinstance(nf, Iterable):
                    raise TypeError(
                        "Node type {} must have iterable features".format(nt)
                    )
        elif isinstance(node_features, Iterable):
            _node_features = defaultdict(lambda: node_features)
        elif node_features is not None:
            raise TypeError(
                "Expected node_features to be None or of type dict or iterable"
            )
        # Analyse the data
        _data = {}
        self._node_feature_sizes = nfs = {}
        self._offsets = _offsets = {}
        _num_nodes = 0
        for node_type, block_data in data.items():
            nf = None if node_features is None else _node_features[node_type]
            nd = NodeData(block_data, node_id=node_id, node_features=nf)
            # TODO Add is_identified() method and check for consistency.
            nn = nd.num_nodes()
            if nn <= 0:
                # Ignore empty blocks
                continue
            _offsets[node_type] = _num_nodes
            _num_nodes += nn
            _data[node_type] = nd
            if node_features is not None:
                nfs[node_type] = len(nf)
        self._data = _data
        self._num_nodes = _num_nodes
        super().__init__(default_node_type, _num_nodes)
        # Determine if nodes have explicit identifiers
        self._has_implicit_ids = node_id is None

    def num_nodes(self) -> int:
        return self._num_nodes + super().num_nodes()

    def node_ids(self) -> Iterable[Any]:
        if self._has_implicit_ids:
            # Global implicit identifiers
            return itertools.chain(range(self._num_nodes), super().node_ids())
        # Get explicit identifiers
        node_ids = [nd.node_ids() for nd in self._data.values()]
        return itertools.chain(*node_ids, super().node_ids())

    def node_types(self) -> Iterable[Any]:
        node_types = [
            DefaultIterable(nd.num_nodes(), nt) for nt, nd in self._data.items()
        ]
        return itertools.chain(*node_types, super().node_types())

    def node_type(self, node_id: Any) -> Optional[Any]:
        if self._has_implicit_ids:
            # Check implicit identifiers
            if isinstance(node_id, int) and 0 <= node_id < self._num_nodes:
                local_node_id = node_id
                for nt, nd in self._data.items():
                    if local_node_id < nd.num_nodes():
                        return nt
                    local_node_id -= nd.num_nodes()
        else:
            # Check explicit identifiers
            for nt, nd in self._data.items():
                if nd.has_node(node_id):
                    return nt
        # Check external identifiers
        return super().node_type(node_id)

    def node_index(self, node_id: Any) -> int:
        if self._has_implicit_ids:
            # Check implicit identifiers
            if isinstance(node_id, int) and 0 <= node_id < self._num_nodes:
                return node_id
        else:
            # Check explicit identifiers
            _offset = 0
            for nd in self._data.values():
                node_idx = nd.node_index(node_id)
                if node_idx >= 0:
                    return _offset + node_idx
                _offset += nd.num_nodes()
        # Check external identifiers
        return super().node_index(node_id)

    def node_id(self, node_idx: int) -> Optional[Any]:
        if 0 <= node_idx < self._num_nodes:
            if self._has_implicit_ids:
                # Index equals implicit identifier.
                return node_idx
            local_node_idx = node_idx
            for nd in self._data.values():
                if local_node_idx < nd.num_nodes():
                    return nd.node_id(local_node_idx)
                local_node_idx -= nd.num_nodes()
        return super().node_id(node_idx)

    def node_type_set(self) -> Set[Any]:
        return self._data.keys() | super().node_type_set()

    def node_feature_sizes(self) -> Mapping[Any, int]:
        return self._node_feature_sizes

    def node_features(self, node_ids: Iterable[Any], node_type: Any = None):
        if node_type is None:
            node_types = {self.node_type(node_id) for node_id in node_ids}
            if None in node_types:
                raise ValueError("Unidentified node(s)")
            if len(node_types) != 1:
                raise ValueError("Ambiguous node type")
            node_type = node_types.pop()
        nd = self._data.get(node_type)
        if nd is None:
            raise ValueError("Unknown node type: {}".format(node_type))
        if self._has_implicit_ids:
            _offset = self._offsets[node_type]
            if _offset > 0:
                node_ids = [node_id - _offset for node_id in node_ids]
            for nt, nd in self._data.items():
                if nt == node_type:
                    break
                _offset += nd.num_nodes()
        return nd.node_features(node_ids)


#############################################
# Base class for an extensible collection of nodes
# that is pre-initialised from supplied data:


class MappedNodeData(DefaultNodeData):
    """
    Base class for a collection of nodes with
    a mapping from explicit to implicit node identifiers.

    Args:
        default_node_type (any, optional):
            The implicit type to assign to any node without an explicit type.
            Defaults to the constant DEFAULT_NODE_TYPE.
        num_implicit_nodes (int, optional):
            The number N of nodes to be assigned implicit identifiers
            in {0, 1, ..., N-1}; defaults to N = 0.
        node_ids (iterable, optional):
            The known collection of explicit node identifiers to be indexed.
        node_types (iterable, optional):
            The collection of explicit node types to be summarised.

    Note:
        If any implicit node identifiers are assigned, then these will
        always be processed before any explicit node identifiers.
    """

    def __init__(
        self,
        default_node_type: Optional[Any] = None,
        num_implicit_nodes: int = 0,
        node_ids: Optional[Iterable[Any]] = None,
        node_types: Optional[Iterable[Any]] = None,
    ):
        self._num_implicit_ids = _offset = num_implicit_nodes
        if node_ids is None:
            self._explicit_ids = {}
        else:
            self._explicit_ids = {
                node_id: (node_idx + _offset)
                for node_idx, node_id in enumerate(node_ids)
            }
        _offset += len(self._explicit_ids)
        super().__init__(default_node_type, _offset)
        if node_types is None:
            self._node_types = set() if _offset == 0 else {self.default_node_type()}
        else:
            self._node_types = set(node_types)

    def num_nodes(self) -> int:
        return self._num_implicit_ids + len(self._explicit_ids) + super().num_nodes()

    def node_ids(self) -> Iterable[Any]:
        return itertools.chain(
            range(self._num_implicit_ids), self._explicit_ids.keys(), super().node_ids()
        )

    def node_index(self, node_id: Any) -> int:
        """
        Obtains the ordered node index corresponding to the given identifier.

        Args:
            node_id (any): The node identifier.

        Returns:
            int: The node index, or a value of -1 if the node is not in the graph.
        """
        if isinstance(node_id, int) and 0 <= node_id < self._num_implicit_ids:
            # Implicit identifier is the index
            return node_id
        # Lookup explicit identifier
        node_idx = self._explicit_ids.get(node_id, -1)
        if node_idx >= 0:
            return node_idx
        # Lookup external identifier
        return super().node_index(node_id)

    def node_type_set(self) -> Set[Any]:
        return self._node_types | super().node_type_set()


#############################################
# Pandas data-frame of node data:


class HashableList(list):
    def __hash__(self):
        val = 0
        for elem in self:
            val = 31 * val + hash(elem)
        return val


class PandasNodeData(MappedNodeData):
    """
    Wrapper for a Pandas data-frame.

    Args:
        data (any):
            The node data in the form of a Pandas data-frame.
        node_id (str or int, optional):
            The name or position of the node identifier.
            If not specified,
            all node identifiers will be obtained via enumeration.
            Use the constant PANDAS_INDEX if the data-frame index
            provides the node identifiers.
        node_type (str or int, optional):
            The name or position of the node type. If not specified,
            all nodes will be implicitly assigned the default node type.
        node_features (iterable or dict, optional):
            Either a collection of column names or positions that comprise
            the node features, or a mapping from node type to node feature columns.
        default_node_type (any, optional):
            The implicit type to assign to any node without an explicit type.
            Defaults to the constant DEFAULT_NODE_TYPE.
    """

    def __init__(
        self,
        data,
        node_id=None,
        node_type=None,
        node_features=None,
        default_node_type=None,
    ):
        self._data = data
        _node_types = self.check_node_type(node_type)
        self.check_node_identifier(node_id, _node_types, default_node_type)
        self.check_node_features(node_features)

    def check_node_type(self, node_type):
        if node_type is None:
            # No explicit types
            self._node_type_col = None
            _node_types = None
        else:
            # Use data-frame column
            self._node_type_col = self.__validate_column(
                "node_type", node_type, list(self._data.columns)
            )
            _node_types = self._data.iloc[:, self._node_type_col].values
        return _node_types

    def check_node_identifier(self, node_id, _node_types, default_node_type):
        # Check node identifiers
        if node_id is None:
            # Assign implicit identifiers
            self._node_id_col = None
            super().__init__(
                default_node_type,
                num_implicit_nodes=len(self._data),
                node_types=_node_types,
            )
        elif node_id == NodeData.PANDAS_INDEX:
            # Use data-frame index
            self._node_id_col = NodeData.PANDAS_INDEX
            super().__init__(
                default_node_type, node_ids=self._data.index, node_types=_node_types
            )
        else:
            # Use data-frame column
            self._node_id_col = col = self.__validate_column(
                "node_id", node_id, list(self._data.columns)
            )
            super().__init__(
                default_node_type,
                node_ids=self._data.iloc[:, col].values,
                node_types=_node_types,
            )

    def check_node_features(self, node_features):
        if node_features is None:
            # No features specified
            self._type_features = None
            self._node_features = None
        elif isinstance(node_features, dict):
            # Features specified by node type
            if self._node_type_col is None:
                raise ValueError(
                    "Use a feature list when there are no explicit node types"
                )
            if len(node_features) == 0:
                raise ValueError("Do not specify empty node type features")
            if len(node_features) == 1:
                raise ValueError(
                    "Use a feature list when there is only one explicit node type"
                )
            self._type_features = tf = {}
            _feature_positions = []  # Holds global column positions
            for nt, nf in node_features.items():
                tf[nt] = nf = self.__validate_features(nf)
                _feature_positions.extend(nf)
            _feature_positions = sorted(set(_feature_positions))
            # Map to local feature positions
            for nt, nf in tf.items():
                tf[nt] = [_feature_positions.index(pos) for pos in tf]
            # Cache the feature matrix
            self._node_features = self._data.iloc[:, _feature_positions].values
        elif isinstance(node_features, Iterable):
            # Features specified by single 'list'
            self._type_features = None
            _feature_positions = self.__validate_features(node_features)
            self._node_features = self._data.iloc[:, _feature_positions].values
        else:
            raise TypeError(
                "Expected node_features to be None or of type dict or iterable"
            )

    def __validate_features(self, feature_cols):
        if not isinstance(feature_cols, Iterable):
            raise TypeError("Expected a collection of feature columns")
        col_names = list(self._data.columns)
        node_features = HashableList()
        for i, feature_col in enumerate(feature_cols):
            name = "node_features[{}]".format(i)
            node_features.append(self.__validate_column(name, feature_col, col_names))
        return node_features

    @staticmethod
    def __validate_column(name, value, col_names) -> int:
        if isinstance(value, int):
            if 0 <= value < len(col_names):
                return value
        elif isinstance(value, str):
            idx = col_names.index(value)
            if idx >= 0:
                return idx
        # Everything else is invalid
        raise ValueError("Invalid data-frame column for {}: {}".format(name, value))

    def node_types(self) -> Iterable[Any]:
        if self._node_type_col is None:
            # No explicit types
            node_types = DefaultIterable(len(self._data), self.default_node_type())
        else:
            # Get explicit types
            node_types = self._data.iloc[:, self._node_type_col].values
        return itertools.chain(node_types, super().node_types())

    def node_type(self, node_id: Any) -> Optional[Any]:
        node_idx = self.node_index(node_id)
        if node_idx < 0:
            # Not in graph
            return None
        elif node_idx < len(self._data):
            if self._node_type_col is None:
                # No explicit types
                return self.default_node_type()
            else:
                # Get explicit type
                return self._data.iloc[node_idx, self._node_type_col]
        else:
            # Default type
            return self.default_node_type()

    def node_id(self, node_idx: int) -> Optional[Any]:
        if 0 <= node_idx < len(self._data):
            if self._node_id_col is None:
                # Index equals implicit identifier.
                return node_idx
            elif self._node_id_col == NodeData.PANDAS_INDEX:
                return self._data.index[node_idx]
            else:
                return self._data.iloc[node_idx, self._node_id_col]
        return super().node_id(node_idx)

    ###################################
    # Node features

    def node_feature_sizes(self) -> Mapping[Any, int]:
        if self._type_features is not None:
            return {nt: len(nf) for nt, nf in self._type_features}
        if self._node_features is None:
            return {}
        num_features = self._node_features.shape[1]
        return {nt: num_features for nt in self.node_type_set()}

    def node_features(self, node_ids: Iterable[Any], node_type: Any = None):
        if self._type_features is not None:
            # Features vary by node type
            if node_type is None:
                # Compute node type
                for node_id in node_ids:
                    nt = self.node_type(node_id)
                    if node_type is None:
                        node_type = nt
                    elif node_type != nt:
                        raise ValueError(
                            "Cannot obtain compatible features for multiple node types"
                        )
            feature_cols = self._type_features[node_type]
            node_idxs = [self.node_index(node_id) for node_id in node_ids]
            return self._node_features[node_idxs, feature_cols]
        if self._node_features is not None:
            # Same features for all node types
            node_idxs = [self.node_index(node_id) for node_id in node_ids]
            return self._node_features[node_idxs, :]
        raise ValueError("No node features have been defined")
