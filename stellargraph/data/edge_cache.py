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
Provides a standarised way to access graph neighbours via edge data,
possibly using optimised caching.
"""

__all__ = ["EdgeCache", "EdgeCacheType"]

from enum import Enum
from typing import Iterable, Any
import itertools

from .edge_data import EdgeData
from .node_data import NodeData


class EdgeCacheType(Enum):
    # No caching of edges
    NONE = 1
    # Cache edges, ignoring types
    UNTYPED = 2
    # TYPED = 3


class EdgeCacheFactory(type):
    """
    Private class for instantiating the edge cache interface from
    user-supplied information.
    """

    def __call__(cls, *args, **kwargs):
        if cls is EdgeCache:
            if len(args) > 0:
                cache_type = args[0]
                args = args[1:]
            else:
                cache_type = kwargs.pop("cache_type", None)
            if cache_type is None:
                # Supply default
                cache_type = EdgeCacheType.UNTYPED
            if not isinstance(cache_type, EdgeCacheType):
                raise TypeError("Invalid cache type: {}".format(cache_type))
            if cache_type == EdgeCacheType.NONE:
                return NoEdgeCache(*args, **kwargs)
            if cache_type == EdgeCacheType.UNTYPED:
                return UntypedEdgeCache(*args, **kwargs)
            raise ValueError("Unknown cache type: {}".format(cache_type))
        else:
            return type.__call__(cls, *args, **kwargs)


class EdgeCache(metaclass=EdgeCacheFactory):
    """
    Encapsulation of node and edge data, for neighbourhood
    queries.

    Args:
        cache_type (EdgeCacheType): The type of cache to use.
        edge_data (EdgeData): The edge data object.
        node_data (NodeData): The node data object.
    """

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
# Class for direct access to the edge data:


class NoEdgeCache(EdgeCache):
    """
    Delegates to the underlying edge data.

    Args:
        edge_data (EdgeData): The edge data object.
        node_data (NodeData): The node data object.
    """

    def __init__(self, edge_data: EdgeData, node_data: NodeData):
        self._edge_data = edge_data
        # Ensure all nodes are in the graph
        for source_id, target_id in edge_data.edges():
            node_data.add_node(source_id)
            node_data.add_node(target_id)

    def neighbour_nodes(self, node_id: Any) -> Iterable[Any]:
        return self._edge_data.neighbour_nodes(node_id)

    def in_nodes(self, node_id: Any) -> Iterable[Any]:
        return self._edge_data.in_nodes(node_id)

    def out_nodes(self, node_id: Any) -> Iterable[Any]:
        return self._edge_data.out_nodes(node_id)


#############################################
# Class for cached access to the edge data, ignoring
# node and edge type information.


class UntypedEdgeCache(EdgeCache):
    """
    Copies the underlying edge data to a
    node adjacency cache, without regard to
    node or edge types.

    Args:
        edge_data (EdgeData): The edge data object.
        node_data (NodeData): The node data object.
    """

    def __init__(self, edge_data: EdgeData, node_data: NodeData):
        self._edge_data = edge_data
        self._in_edges = _in_edges = {}
        self._out_edges = _out_edges = {}
        self._self_edges = _self_edges = {}

        def _insert(cache, key, value):
            values = cache.get(key)
            if values is None:
                cache[key] = {value}
            else:
                values.add(value)

        for src_id, dst_id, edge_id, _, _ in edge_data.edges(include_info=True):
            # Ensure all nodes are in the graph
            node_data.add_node(src_id)
            node_data.add_node(dst_id)
            # Add edge to cache
            _insert(_out_edges, src_id, (dst_id, edge_id))
            if dst_id != src_id:
                _insert(_in_edges, dst_id, (src_id, edge_id))

    def neighbour_nodes(self, node_id: Any) -> Iterable[Any]:
        return itertools.chain(
            [n[0] for n in self._in_edges.get(node_id, [])],
            [n[0] for n in self._out_edges.get(node_id, [])],
            [n[0] for n in self._self_edges.get(node_id, [])],
        )

    def in_nodes(self, node_id: Any) -> Iterable[Any]:
        return itertools.chain(
            [n[0] for n in self._in_edges.get(node_id, [])],
            [n[0] for n in self._self_edges.get(node_id, [])],
        )

    def out_nodes(self, node_id: Any) -> Iterable[Any]:
        return itertools.chain(
            [n[0] for n in self._out_edges.get(node_id, [])],
            [n[0] for n in self._self_edges.get(node_id, [])],
        )
