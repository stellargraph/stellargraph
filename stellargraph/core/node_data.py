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

The general principal is that the data are iterable, with one
node defined per element. Each node element should provide at least the node
identifier and the type of the node (although one of these may be inferred).

Preferably there should only be one pass through the node data for the
purposes of efficiency, so the calculation of some properties might
have to be deferred. In particular:
    - If the number of nodes cannot be known in advance, then num_nodes()
        will return None.
    - If the heterogeneity or homogeneity of the node types cannot
        be determined in advance, then is_heterogenous() and is_homogeneous()
        will return None.
    - If the set of distinct node types cannot be determined in advance,
        then node_types() will return None.
After a full pass through the node data, these method are guaranteed to
return correct values.

Supported data input formats:
    - Pandas data-frame
    - NumPy array
    - dictionary of node-type -> node-data pairs
    - collection of indexable objects,
        e.g. list, tuple, dict, etc.
    - collection of objects with fields.

Required attributes:
    - node_id:  The position of the node identifier (which might
        be able to be inferred from the data type).
    - node_type: For heterogeneous graphs, the position of the
        node type; this will be inferred for a dictionary data type.

Attribute specification:
    - For a data type with named columns, a 'position' can be a column
        name or column index.
    - For a collection of objects with fields, it will be a field name.
    - For a collection of dictionary objects, it will be a key value.
    - For a collection of other indexable objects (such as a list of lists
        or a list of tuples, etc.), the 'position' will be an integer index value.
    - For a node-type dictionary of node-data objects, the node_id
        position is assumed to be the same for each block of node data.
"""

import pandas as pd
import numpy as np


# Useful constants:
DEFAULT_NODE_TYPE = "node"
IS_IDENTIFIED = True
IS_UNIDENTIFIED = False
IS_TYPED = True
IS_UNTYPED = False
IS_HETEROGENEOUS = True
IS_HOMOGENEOUS = False
IS_UNDETERMINED = None
PANDAS_INDEX = -1
SINGLE_COLUMN = -2


#############################################
# Classes for node data:


class NodeDatum(tuple):
    """
    Encapsulates a single node containing:
        - node_id: The identifier of the node.
        - node_type: The type of the node.
    """

    def __new__(cls, node_id, node_type):
        return tuple.__new__(NodeDatum, (node_id, node_type))

    def __repr__(self):
        return "{}(node_id={}, node_type={})".format(self.__class__.__name__, *self)

    @property
    def node_id(self):
        """
        Obtains the identifier of this node.

        Returns:
             The node identifier.
        """
        return self[0]

    @property
    def node_type(self):
        """
        Obtains the type of this node.

        Returns:
             The node type.
        """
        return self[1]

    def with_id(self, node_id):
        """
        Helper method to replace the node identifier.

        Args:
            node_id: The new node identifier.

        Returns:
            A new NodeDatum object.
        """
        return NodeDatum(node_id, self[1])

    def with_type(self, node_type):
        """
        Helper method to replace the node type.

        Args:
            node_type: The new node type.

        Returns:
            A new NodeDatum object.
        """
        return NodeDatum(self[0], node_type)


class NodeData:
    """
    The base class for all node data wrappers.
    """

    def __init__(self, is_identified, is_typed, default_node_type):
        """
        Initialises the base node data structure.

        Args:
            is_identified: <bool> Indicates whether the nodes have
                explicit identifiers (True), or will be assigned implicit
                identifiers (False).
            is_typed: <bool> Indicates whether the nodes have explicit
                types (True), or will be assigned the default node type (False).
            default_node_type: The optional type to assign to nodes without an explicit type
                (defaults to the constant DEFAULT_NODE_TYPE).
        """
        self._is_identified = is_identified
        self._is_typed = is_typed
        self._default_node_type = (
            default_node_type if default_node_type is not None else DEFAULT_NODE_TYPE
        )
        # Currently undetermined values
        self._is_heterogeneous = None
        self._node_types = None
        self._num_nodes = None

    def is_identified(self) -> bool:
        """
        Indicates whether the node identifiers are explicit (True) or implicit (False).

        Returns:
             <bool> The Boolean flag.
        """
        return self._is_identified

    def is_unidentified(self) -> bool:
        """
        Indicates whether the node identifiers are implicit (True) or explicit (False).

        Returns:
             <bool> The Boolean flag.
        """
        return not self._is_identified

    def is_heterogeneous(self) -> bool or None:
        """
        Indicates whether the node types are known to be heterogeneous
        (True) or homogeneous (False); if this status is currently unknown,
        then a value of None will be returned.

        Returns:
             The Boolean heterogeneity status if known, or a value of None.
        """
        return self._is_heterogeneous

    def is_homogeneous(self) -> bool or None:
        """
        Indicates whether the node types are known to be homogeneous
        (True) or heterogeneous (False); if this status is currently unknown,
        then a value of None will be returned.

        Returns:
             The Boolean homogeneity status if known, or a value of None.
        """
        return None if self._is_heterogeneous is None else not self._is_heterogeneous

    def is_typed(self) -> bool:
        """
        Indicates whether the nodes have explicit (True) or implicit (False) types.

        Returns:
             <bool> The Boolean flag.
        """
        return self._is_typed

    def is_untyped(self) -> bool:
        """
        Indicates whether the nodes have implicit (True) or explicit (False) types.

        Returns:
             <bool> The Boolean flag.
        """
        return not self._is_typed

    def default_node_type(self, node_type=None):
        """
        Helper method to supply the default node type if
        one is not specified.

        Args:
            node_type: The optional node type.

        Returns:
            A defined node type.
        """
        return self._default_node_type if node_type is None else node_type

    def node_types(self) -> set or None:
        """
        Obtains the (possibly empty) collection of distinct node types.

        Note: The calculation of this might be deferred until after a full
        iteration through the node data.

        Returns:
            The set of distinct node types if known, or a value of None.
        """
        return self._node_types

    def _set_node_types(self, node_types):
        """
        A private method for setting the distinct node types,
        computed after initialisation (and possibly after one
        full pass through the data).

        Args:
            node_types: The computed node types.
        """
        self._node_types = node_types
        self._is_heterogeneous = len(node_types) > 1

    def num_nodes(self) -> int or None:
        """
        Obtains the number of nodes in the node data.

        Returns:
             The number of nodes if known, or a value of None.
        """
        return self._num_nodes

    def _set_num_nodes(self, num_nodes):
        """
        A private method for setting the number of
        nodes after an explicit computation.

        Args:
            num_nodes: <int> The number of nodes.
        """
        self._num_nodes = num_nodes
        if num_nodes == 0:
            self._is_heterogeneous = False
            self._node_types = set()
            # Override the following for consistency with NoNodeData:
            self._is_typed = False
            self._is_identified = False
        elif not self._is_typed:
            self._is_heterogeneous = False
            self._node_types = set([self._default_node_type])

    def nodes(self):
        """
        Provides an iterator or generator over node data.
        Each element so obtained will be a NodeDatum object.

        Returns:
             The iterator or generator.
        """
        raise NotImplementedError


#############################################
# Base class for iterable collections of nodes:


class RowNodeData(NodeData):
    """
    Abstract wrapper for an iterable collection of nodes.
    It is assumed that it is more efficient here to
    iterate over all nodes, rather than just look up nodes
    by an implicit index.
    """

    def nodes(self):
        is_uncertain = self.num_nodes() is None or self.is_typed()
        if is_uncertain:
            node_types = set()
        _node_id = -1
        for row in self._iter_rows():
            _node_id += 1
            node = self._get_node(row)
            if self.is_unidentified():
                node = node.with_id(_node_id)
            if is_uncertain:
                node_types.add(node.node_type)
            yield node
        if is_uncertain:
            self._set_num_nodes(_node_id + 1)
            self._set_node_types(node_types)

    def _iter_rows(self):
        """
        A private method for obtaining an iterator or generator
        over the collection of raw node data.

        Returns:
            The iterator or generator.
        """
        raise NotImplementedError

    def _get_node(self, row):
        """
        A private method for obtaining a NodeDatum
        from the raw row of data.

        Args:
            row: The row of attributes for the given node.

        Returns:
             A NodeDatum representation of the node.
        """
        raise NotImplementedError


#############################################
# Default class for no node data:


class NoNodeData(NodeData):
    """
    Wrapper for the special case of having no node attributes.

    This might be useful if all nodes are explicitly identified
    by a corresponding EdgeData object.
    """

    def __init__(self, default_node_type=None):
        super().__init__(IS_UNIDENTIFIED, IS_UNTYPED, default_node_type)
        super()._set_num_nodes(0)

    def nodes(self):
        return []


#############################################
# Dictionary of node type -> node data pairs:


class TypeDictNodeData(RowNodeData):
    """
    Wrapper for a dictionary of node-type -> node-data pairs.

    Note that if node_id is specified, the node identifiers are
    assumed to be globally unique across all blocks of node data;
    otherwise, we assume the inferred identifiers for each block
     are local to that block, and will be discarded in favour of
     globally unique identifiers.
    """

    def __init__(self, data, node_id=None, default_node_type=None):
        _is_identified = node_id is not None
        super().__init__(_is_identified, IS_TYPED, default_node_type)
        self._data = _data = {}
        is_determined = True
        node_types = set()
        num_nodes = 0
        for _node_type, block_data in data.items():
            node_type = self.default_node_type(_node_type)  # in case of None
            if node_type in _data:  # in case of None and default
                raise ValueError(
                    "Node types contain both None and default '{}'".format(node_type)
                )
            # Wrap type-specific data
            _data[node_type] = block_data = node_data(
                block_data, node_id, None, node_type
            )
            block_size = block_data.num_nodes()
            is_determined = is_determined and block_size is not None
            if block_size is not None and block_size > 0:
                if _is_identified and block_data.is_unidentified():
                    raise ValueError(
                        "Node data for type '{}' has local identifiers!".format(
                            _node_type
                        )
                    )
                num_nodes += block_size
                node_types.add(node_type)
        if is_determined:
            self._set_num_nodes(num_nodes)
            self._set_node_types(node_types)

    def _iter_rows(self):
        # XXX The dictionary values are NodeData objects.
        for node_type, block_data in self._data.items():
            for node in block_data.nodes():
                # XXX Cannot guarantee the inner node type
                if node.node_type != node_type:
                    node = node.with_type(node_type)
                yield node

    def _get_node(self, row):
        # XXX A row is already a NodeDatum object.
        return row


#############################################
# Pandas data-frame of node data:


class PandasNodeData(RowNodeData):
    """
    Wrapper for a Pandas data-frame.

    The node identifiers are taken from the Pandas index
    if node_id is set to PANDAS_INDEX; otherwise,
    if node_id is defined the identifiers are taken from the
    specified column, or else are enumerated for each node.
    """

    def __init__(self, data, node_id=None, node_type=None, default_node_type=None):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("A Pandas DataFrame is required!")
        super().__init__(node_id is not None, node_type is not None, default_node_type)
        self._set_num_nodes(len(data))
        self._data = data
        col_names = list(data.columns)
        self._id_idx = self.__validate_position("node_id", node_id, True, col_names)
        self._type_idx = self.__validate_position(
            "node_type", node_type, True, col_names
        )

    @staticmethod
    def __validate_position(name, value, is_nullable, col_names):
        # XXX The Pandas tuple has the index value at position 0,
        # so we have to offset all column positions by +1.
        if value is None:
            if is_nullable:
                return -1
        elif isinstance(value, int):
            if name == "node_id" and value == PANDAS_INDEX:
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

    def _get_node(self, row):
        # XXX Row is a Pandas tuple.
        node_id = None if self._id_idx < 0 else row[self._id_idx]
        node_type = self.default_node_type(
            None if self._type_idx < 0 else row[self._type_idx]
        )
        return NodeDatum(node_id, node_type)


#############################################
# Two-dimensional NumPy array of node data:


class NumPy2NodeData(RowNodeData):
    """
    Wrapper for a two-dimensional NumPy array.

    If node_id is defined then the identifiers are taken from the
    specified column; otherwise they are enumerated for each node.
    """

    def __init__(self, data, node_id=None, node_type=None, default_node_type=None):
        if not isinstance(data, np.ndarray):
            raise ValueError("A NumPy ndarray is required!")
        if len(data.shape) != 2:
            raise ValueError("Only two-dimensional arrays are supported!")
        super().__init__(node_id is not None, node_type is not None, default_node_type)
        self._set_num_nodes(len(data))
        self._data = data
        num_cols = data.shape[1]
        self._id_idx = self.__validate_position("node_id", node_id, True, num_cols)
        self._type_idx = self.__validate_position(
            "node_type", node_type, True, num_cols
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

    def _get_node(self, row):
        # XXX The row is just the implicit node index.
        node_id = None if self._id_idx < 0 else self._data[row, self._id_idx]
        node_type = self.default_node_type(
            None if self._type_idx < 0 else self._data[row, self._type_idx]
        )
        return NodeDatum(node_id, node_type)


#############################################
# One-dimensional NumPy array of node data:


class NumPy1NodeData(RowNodeData):
    """
    Wrapper for a one-dimensional NumPy array of node identifier
    or node type values.

    One of node_id or node_type must be None and the
    other must be the constant SINGLE_COLUMN.
    """

    def __init__(self, data, node_id=None, node_type=None, default_node_type=None):
        if not isinstance(data, np.ndarray):
            raise ValueError("A NumPy ndarray is required!")
        if len(data.shape) != 1:
            raise ValueError("Only one-dimensional arrays are supported!")
        if node_id == SINGLE_COLUMN and node_type is None:
            self._is_id = True
        elif node_type == SINGLE_COLUMN and node_id is None:
            self._is_id = False
        else:
            raise ValueError(
                "Cannot obtain both node_id and node_type from one-dimensional data!"
            )

        super().__init__(node_id is not None, node_type is not None, default_node_type)
        self._set_num_nodes(len(data))
        self._data = data

    def _iter_rows(self):
        # Pretend to iterate over the rows.
        return range(len(self._data))

    def _get_node(self, row):
        # XXX The row is just the implicit node index.
        if self._is_id:
            node_id = self._data[row]
            node_type = self.default_node_type()
        else:
            node_id = None
            node_type = self.default_node_type(self._data[row])
        return NodeDatum(node_id, node_type)


#############################################
# Iterable collection of multi-attribute node data:


class Iterable2NodeData(RowNodeData):
    """
    Wrapper for an iterable collection of nodes that might or might
    not have a predefined length.

    The individual node objects are assumed to have attributes that
    are individually extractable via an index or a field.
    """

    def __init__(self, data, node_id=None, node_type=None, default_node_type=None):
        super().__init__(node_id is not None, node_type is not None, default_node_type)
        if hasattr(data, "__len__"):
            self._set_num_nodes(len(data))
        self._data = data
        self._id_idx = node_id
        self._type_idx = node_type

    def _iter_rows(self):
        # XXX Assumes iterable node data
        return self._data

    def _get_node(self, row):
        if hasattr(row, "__getitem__"):
            return self._get_values_by_index(row)
        return self._get_values_by_field(row)

    def _get_values_by_index(self, row):
        node_id = None if self._id_idx is None else row[self._id_idx]
        node_type = self.default_node_type(
            None if self._type_idx is None else row[self._type_idx]
        )
        return NodeDatum(node_id, node_type)

    def _get_values_by_field(self, row):
        node_id = None if self._id_idx is None else getattr(row, self._id_idx)
        node_type = self.default_node_type(
            None if self._type_idx is None else getattr(row, self._type_idx)
        )
        return NodeDatum(node_id, node_type)


#############################################
# One-dimensional iterable of node data:


class Iterable1NodeData(RowNodeData):
    """
    Wrapper for a one-dimensional collection of node identifier
    or node type values.

    One of node_id or node_type must be None and the
    other must be the constant SINGLE_COLUMN.
    """

    def __init__(self, data, node_id=None, node_type=None, default_node_type=None):
        if node_id == SINGLE_COLUMN and node_type is None:
            self._is_id = True
        elif node_type == SINGLE_COLUMN and node_id is None:
            self._is_id = False
        else:
            raise ValueError(
                "Cannot obtain both node_id and node_type from one-dimensional data!"
            )

        super().__init__(node_id is not None, node_type is not None, default_node_type)
        self._set_num_nodes(len(data))
        self._data = data

    def _iter_rows(self):
        # XXX Assumes iterable node data
        return self._data

    def _get_node(self, row):
        # XXX The row is just the node value.
        if self._is_id:
            node_id = row
            node_type = self.default_node_type()
        else:
            node_id = None
            node_type = self.default_node_type(row)
        return NodeDatum(node_id, node_type)


#############################################
# Main data wrapper method:


def node_data(
    data=None, node_id=None, node_type=None, default_node_type=None
) -> NodeData:
    """
    Wraps the node data with a NodeData object.
    Has no effect if 'data' is already NodeData.

    Args:
        data: The node data in one of the standard formats. This may
            be None or empty if there are no node attributes
            (parameters other than default_node_type are then ignored).
        node_id: The position of the node identifier.
            This is optional if the node_id can be inferred from the data type.
            Use the constant SINGLE_COLUMN if 'data' is a
            one-dimensional collection of node identifiers.
        node_type: The optional position of the node type. If not specified,
            all nodes will be implicitly assigned the default node type.
            Use the constant SINGLE_COLUMN if 'data' is a
            one-dimensional collection of node types.
        default_node_type: The optional type to assign to nodes without an explicit type.

    Returns:
        A NodeData instance.
    """
    # Check for known data:
    if data is None:
        return NoNodeData(default_node_type)
    if isinstance(data, NodeData):
        return data
    # Check for empty data:
    if hasattr(data, "__len__") and len(data) == 0:
        return NoNodeData(default_node_type)
    # Check for dictionary of node-type -> node-data pairs:
    if isinstance(data, dict):
        return TypeDictNodeData(data, node_id, default_node_type)
    # Check for Pandas data-frame:
    if isinstance(data, pd.DataFrame):
        return PandasNodeData(data, node_id, node_type, default_node_type)
    # Check for NumPy array:
    if isinstance(data, np.ndarray):
        if node_id == SINGLE_COLUMN or node_type == SINGLE_COLUMN:
            # One-dimensional
            return NumPy1NodeData(data, node_id, node_type, default_node_type)
        # Two-dimensional
        return NumPy2NodeData(data, node_id, node_type, default_node_type)
    # Check for arbitrary collection:
    if hasattr(data, "__iter__") or hasattr(data, "__getitem__"):
        if node_id == SINGLE_COLUMN or node_type == SINGLE_COLUMN:
            # One-dimensional
            return Iterable1NodeData(data, node_id, node_type, default_node_type)
        # Two-dimensional
        return Iterable2NodeData(data, node_id, node_type, default_node_type)
    # Don't know this data type!
    raise ValueError("Unknown node data type: {}".format(type(data)))
